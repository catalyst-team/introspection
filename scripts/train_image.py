import argparse
from datetime import datetime
import os

from catalyst import dl, utils
from catalyst.contrib.data import AllTripletsSampler, Compose, ImageToTensor, NormalizeImage
from catalyst.contrib.datasets import CIFAR10
from catalyst.contrib.losses import TripletMarginLossWithSampler
from catalyst.data import BatchBalanceClassSampler
from torch import nn, optim
from torch.utils.data import DataLoader

from introspection.modules import resnet9
from introspection.settings import LOGS_ROOT


class CustomRunner(dl.Runner):
    def handle_batch(self, batch) -> None:
        images, targets = batch
        embeddings, logits = self.model(images)

        self.batch = {
            "embeddings": embeddings,
            "targets": targets,
            "logits": logits,
        }


def main(use_ml: bool = False):
    # data
    transform = Compose([ImageToTensor(), NormalizeImage((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = CIFAR10(os.getcwd(), train=True, download=True, transform=transform)
    valid_dataset = CIFAR10(os.getcwd(), train=False, download=True, transform=transform)

    # loaders
    labels = train_dataset.targets
    sampler = BatchBalanceClassSampler(labels=labels, num_classes=10, num_samples=20)
    bs = sampler.batch_size
    loaders = {
        "train": DataLoader(train_dataset, batch_sampler=sampler, num_workers=8),
        "valid": DataLoader(valid_dataset, batch_size=bs, num_workers=8, shuffle=False),
    }

    # model
    model = resnet9(in_channels=3, num_classes=10)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [5, 8], gamma=0.3)

    criterion_ce = nn.CrossEntropyLoss()
    sampler_inbatch = AllTripletsSampler()
    criterion_ml = TripletMarginLossWithSampler(margin=0.5, sampler_inbatch=sampler_inbatch)
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
        dl.AccuracyCallback(input_key="logits", target_key="targets", topk_args=(1, 3, 5)),
        dl.OptimizerCallback(metric_key="loss" if use_ml else "loss_ce"),
        dl.SchedulerCallback(),
    ]
    if use_ml:
        callbacks.extend(
            [
                dl.ControlFlowCallback(
                    base_callback=dl.CriterionCallback(
                        input_key="embeddings",
                        target_key="targets",
                        metric_key="loss_ml",
                        criterion_key="ml",
                    ),
                    loaders=["train"],
                ),
                dl.ControlFlowCallback(
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
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        num_epochs=20,
        callbacks=callbacks,
        logdir=f"{LOGS_ROOT}/image-ml{ml_flag}-{strtime}",
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
            dl.AccuracyCallback(input_key="logits", target_key="targets", topk_args=(1, 3, 5)),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=10
            ),
        ],
    )
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    utils.boolean_flag(parser, "use-ml", default=False)
    args = parser.parse_args()
    main(args.use_ml)
