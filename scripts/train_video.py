import argparse
from datetime import datetime

from catalyst import dl, utils
from catalyst.contrib.data import AllTripletsSampler
from catalyst.contrib.losses import TripletMarginLossWithSampler
from catalyst.data import BatchBalanceClassSampler, BatchPrefetchLoaderWrapper
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader

from introspection.datasets import TemporalDataset
from introspection.modules import TemporalResNet
from introspection.settings import DATA_ROOT, LOGS_ROOT


class CustomRunner(dl.Runner):
    def handle_batch(self, batch) -> None:
        images, targets = batch
        embeddings, logits = self.model(images)

        # batch size, length, feature size
        bs, ln, fs = embeddings.shape
        t_embeddings = embeddings.view(bs * ln, fs)
        t_targets = targets.repeat_interleave(ln)

        self.batch = {
            "temporal_embeddings": t_embeddings,
            "temporal_targets": t_targets,
            "targets": targets,
            "logits": logits,
        }


def main(use_ml: bool = False, freeze_encoder: bool = False):
    # data
    train_csv = pd.read_csv(
        f"{DATA_ROOT}/UCF11_updated_mpg_clean/train.csv",
        header=None,
        names=["path", "class", "length"],
    )
    valid_csv = pd.read_csv(
        f"{DATA_ROOT}/UCF11_updated_mpg_clean/valid.csv",
        header=None,
        names=["path", "class", "length"],
    )

    num_segments = 5
    segment_len = 5
    train_dataset = TemporalDataset(
        train_csv,
        f"{DATA_ROOT}/UCF11_updated_mpg",
        num_segments=num_segments,
        segment_len=segment_len,
    )
    valid_dataset = TemporalDataset(
        valid_csv,
        f"{DATA_ROOT}/UCF11_updated_mpg",
        num_segments=num_segments,
        segment_len=segment_len,
    )

    # loaders
    labels = train_dataset.get_labels()
    sampler = BatchBalanceClassSampler(labels=labels, num_classes=6, num_samples=2)
    bs = sampler.batch_size
    loaders = {
        "train": DataLoader(train_dataset, batch_sampler=sampler, num_workers=8),
        "valid": DataLoader(valid_dataset, batch_size=bs, num_workers=8, shuffle=False),
    }
    loaders = {k: BatchPrefetchLoaderWrapper(v) for k, v in loaders.items()}

    # model
    model = TemporalResNet(
        emb_features=256,
        out_features=train_csv["class"].nunique(),
        arch="resnet18",
        pretrained=True,
        freeze_encoder=freeze_encoder,
    )
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.3)

    criterion_ce = nn.CrossEntropyLoss()
    sampler_inbatch = AllTripletsSampler()
    criterion_ml = TripletMarginLossWithSampler(
        margin=0.5, sampler_inbatch=sampler_inbatch
    )
    criterion = {"ce": criterion_ce, "ml": criterion_ml}

    # runner
    runner = CustomRunner()

    # callbacks
    callbacks = [
        dl.CriterionCallback(
            input_key="logits",
            target_key="targets",
            metric_key="loss_ce",
            criterion_key="ce",
        ),
        dl.AccuracyCallback(input_key="logits", target_key="targets", topk=(1, 3, 5)),
        dl.BackwardCallback(metric_key="loss" if use_ml else "loss_ce"),
        dl.OptimizerCallback(metric_key="loss" if use_ml else "loss_ce"),
        dl.SchedulerCallback(),
    ]
    if use_ml:
        callbacks.extend(
            [
                dl.ControlFlowCallbackWrapper(
                    base_callback=dl.CriterionCallback(
                        input_key="temporal_embeddings",
                        target_key="temporal_targets",
                        metric_key="loss_ml",
                        criterion_key="ml",
                    ),
                    loaders=["train"],
                ),
                dl.ControlFlowCallbackWrapper(
                    base_callback=dl.MetricAggregationCallback(
                        metric_key="loss",
                        metrics=["loss_ce", "loss_ml"],
                        mode="mean",
                    ),
                    loaders=["train"],
                ),
            ]
        )

    # train
    strtime = datetime.now().strftime("%Y%m%d-%H%M%S")
    ml_flag = int(use_ml)
    encoder_flag = int(not freeze_encoder)
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        num_epochs=20,
        callbacks=callbacks,
        logdir=f"{LOGS_ROOT}/video-ml{ml_flag}-encoder{encoder_flag}-{strtime}",
        valid_loader="valid",
        valid_metric="accuracy01",
        minimize_valid_metric=False,
        verbose=True,
        load_best_on_end=True,
    )

    # evaluate
    metrics = runner.evaluate_loader(
        loader=loaders["valid"],
        callbacks=[
            dl.AccuracyCallback(
                input_key="logits", target_key="targets", topk=(1, 3, 5)
            ),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=11
            ),
        ],
    )
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    utils.boolean_flag(parser, "use-ml", default=False)
    utils.boolean_flag(parser, "freeze-encoder", default=False)
    args = parser.parse_args()
    main(args.use_ml, args.freeze_encoder)
