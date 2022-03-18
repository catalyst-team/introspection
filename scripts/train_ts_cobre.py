import argparse
from datetime import datetime

from catalyst import dl, utils
from catalyst.contrib.data import HardTripletsSampler
from catalyst.contrib.losses import TripletMarginLossWithSampler
from catalyst.contrib.utils.torch import get_optimal_inner_init, outer_init
from catalyst.data import BatchBalanceClassSampler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from introspection.settings import LOGS_ROOT
from introspection.ts_cobre import load_COBRE


class TSQuantileTransformer:
    def __init__(self, *args, n_quantiles: int, **kwargs):
        self.n_quantiles = n_quantiles
        self._args = args
        self._kwargs = kwargs
        self.transforms = {}

    def fit(self, features: np.ndarray):
        for i in range(features.shape[1]):
            self.transforms[i] = QuantileTransformer(
                *self._args, n_quantiles=self.n_quantiles, **self._kwargs
            ).fit(features[:, i, :])
        return self

    def transform(self, features: np.ndarray):
        result = np.empty_like(features, dtype=np.int32)
        for i in range(features.shape[1]):
            result[:, i, :] = (
                self.transforms[i].transform(features[:, i, :]) * self.n_quantiles
            ).astype(np.int32)
        return result


class TemporalDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        num_segments: int = None,
        segment_len: int = None,
    ):
        super().__init__()
        self.features = features
        self.targets = targets
        self.num_segments = num_segments
        self.segment_len = segment_len

    def __getitem__(self, idx):
        target = self.targets[idx]
        images = np.swapaxes(self.features[idx], 0, 1)  # [time, features]

        if self.num_segments is not None:
            images = np.array_split(images, self.num_segments)

        if self.segment_len is not None:
            sampled_images = []
            for images_segment in images:
                idxs = np.random.choice(
                    len(images_segment),
                    self.segment_len,
                    replace=len(images_segment) < self.segment_len,
                )
                sampled_images.extend(images_segment[idxs])
        else:
            sampled_images = images

        features = np.vstack([im[np.newaxis] for im in sampled_images])
        return features, target

    def __len__(self):
        return len(self.targets)

    def get_labels(self):
        return list(self.targets)


class TemporalModel(nn.Module):
    def __init__(
        self,
        in_features: int,
        emb_features: int,
        n_channels: int,
        hid_features: int,
        out_features: int,
    ):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Embedding(in_features, emb_features),
            nn.Flatten(start_dim=2),
            nn.Linear(n_channels * emb_features, hid_features),
            nn.ReLU(),
        )
        # self.attention = nn.Sequential(nn.Linear(in_features, 1), nn.Sigmoid())
        self.classifier = nn.Linear(hid_features, out_features)
        self.embedder.apply(get_optimal_inner_init(nn.ReLU))
        self.classifier.apply(outer_init)

    def forward(self, x):
        embeddings = self.embedder(x)
        # x_a = self.attention(x.view(bs, sl, -1))
        # x = x_r * x_a
        logits = self.classifier(embeddings).mean(1)
        return embeddings, logits


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

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "wandb": dl.WandbLogger(project="wandb_test", name="cobre_experiment"),
        }


def main(use_ml: bool = False):
    # data
    features, labels = load_COBRE()
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.33, random_state=42, stratify=labels
    )
    n_quantiles = 10
    transform = TSQuantileTransformer(n_quantiles=n_quantiles, random_state=42).fit(X_train)
    X_train2 = transform.transform(X_train)
    X_test2 = transform.transform(X_test)

    # loaders
    train_dataset = TemporalDataset(X_train2, y_train)
    labels = train_dataset.get_labels()
    sampler = BatchBalanceClassSampler(labels=labels, num_classes=2, num_samples=16)
    bs = sampler.batch_size
    loaders = {
        "train": DataLoader(train_dataset, batch_size=bs, num_workers=1),
        "valid": DataLoader(
            TemporalDataset(X_test2, y_test), batch_size=32, num_workers=1, shuffle=False
        ),
    }

    # model
    model = TemporalModel(
        in_features=n_quantiles + 1,
        emb_features=16,
        n_channels=53,
        hid_features=512,
        out_features=2,
    )
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0005)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.3)

    criterion_ce = nn.CrossEntropyLoss()
    sampler_inbatch = HardTripletsSampler()
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
        dl.AccuracyCallback(input_key="logits", target_key="targets", topk=(1,)),
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
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        num_epochs=20,
        callbacks=callbacks,
        logdir=f"{LOGS_ROOT}/ts-ml{ml_flag}-{strtime}",
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
            dl.AccuracyCallback(input_key="logits", target_key="targets", topk=(1,)),
            dl.PrecisionRecallF1SupportCallback(
                input_key="logits", target_key="targets", num_classes=2
            ),
        ],
    )
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    utils.boolean_flag(parser, "use-ml", default=False)
    args = parser.parse_args()
    main(args.use_ml)
