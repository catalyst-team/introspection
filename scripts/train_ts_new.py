import argparse
from datetime import datetime
from functools import reduce
import math

from catalyst import dl, utils
from catalyst.contrib.data import HardTripletsSampler
from catalyst.contrib.losses import TripletMarginLossWithSampler
from catalyst.data import BatchBalanceClassSampler
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from x_transformers import Encoder, TransformerWrapper

from introspection.settings import LOGS_ROOT
from introspection.ts import load_ABIDE1

N_CHANNEL = 2  # up to 53 (prior)
assert N_CHANNEL >= 1 and N_CHANNEL <= 53


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = mask.cumsum(dim=-1) > (num_tokens * prob).ceil()
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


# https://github.com/lucidrains/mlm-pytorch/blob/master/mlm_pytorch/mlm_pytorch.py
class MLM(nn.Module):
    def __init__(
        self,
        mask_prob=0.15,
        replace_prob=0.9,
        num_tokens=None,
        random_token_prob=0.0,
        mask_token_id=2,
        pad_token_id=0,
        mask_ignore_token_ids=[],
    ):
        super().__init__()
        # mlm related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob
        # token ids
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

    def forward(self, seq):
        # seq: [bs; seq_len]
        # do not mask [pad] tokens,
        # or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens(seq, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)
        # mask input with mask tokens with probability of `replace_prob`
        # (keep tokens the same with probability 1 - replace_prob)
        masked_seq = seq.clone().detach()
        # derive labels to predict
        labels = seq.masked_fill(~mask, self.pad_token_id).type(torch.LongTensor)
        # if random token probability > 0 for mlm
        if self.random_token_prob > 0:
            assert self.num_tokens is not None, (
                "num_tokens keyword must be supplied when instantiating MLM"
                "if using random token replacement"
            )
            random_token_prob = prob_mask_like(seq, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, seq.shape, device=seq.device)
            random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_seq = torch.where(random_token_prob, random_tokens, masked_seq)
            # remove tokens that were substituted randomly from being [mask]ed later
            mask = mask & ~random_token_prob

        # [mask] input
        replace_prob = prob_mask_like(seq, self.replace_prob)
        masked_seq = masked_seq.masked_fill(mask * replace_prob, self.mask_token_id)

        return masked_seq, labels


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
        time_features = np.swapaxes(self.features[idx], 0, 1)  # [time, features]

        if self.num_segments is not None:
            time_features = np.array_split(time_features, self.num_segments)

        if self.segment_len is not None:
            sampled_time_features = []
            for time_segment in time_features:
                idxs = np.random.choice(
                    len(time_segment),
                    self.segment_len,
                    replace=len(time_segment) < self.segment_len,
                )
                sampled_time_features.extend(time_segment[idxs])
        else:
            sampled_time_features = time_features

        features = np.vstack([x[np.newaxis] for x in sampled_time_features])
        features = features[:, :N_CHANNEL]
        return features, target

    def __len__(self):
        return len(self.targets)

    def get_labels(self):
        return list(self.targets)


class TemporalModel(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        max_seq_len: int,
        emb_features: int,
        out_features: int,
    ):
        super().__init__()
        self.transformer = TransformerWrapper(
            num_tokens=num_tokens,
            max_seq_len=max_seq_len,
            attn_layers=Encoder(dim=emb_features, depth=4, heads=4),
        )
        self.classifier = nn.Linear(emb_features, out_features)

    def forward(self, seq):  # seq: [bs; seq_len]
        seq_emb = self.transformer(seq, return_embeddings=True)
        seq_logits = self.transformer.to_logits(seq_emb)
        logits = self.classifier(seq_emb).mean(1)
        return seq_emb, seq_logits, logits


class CustomRunner(dl.Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlm = MLM()

    def handle_batch(self, batch) -> None:
        features, targets = batch
        features_seq = features.flatten(1, -1)  # [bs; seq_len, ch] -> [bs; seq_len * ch]
        seq_masked, seq_targets = self.mlm(seq=features_seq)

        # seq_emb: [bs; seq_len; emb_size]
        # seq_logits: [bs; seq_len; num_tokens]
        # logits: [bs; num_classes]
        seq_emb, seq_logits, logits = self.model(seq_masked)

        seq_logits = seq_logits.transpose(1, 2)
        # mlm_loss = F.cross_entropy(
        #     seq_logits.transpose(1, 2),
        #     masked_lbl,
        #     ignore_index=self.mlm.pad_token_id
        # )

        # batch size, seq len, feature size
        bs, sl, fs = seq_emb.shape
        t_embeddings = seq_emb.view(bs * sl, fs)
        t_targets = targets.repeat_interleave(sl)

        self.batch = {
            "seq_logits": seq_logits,
            "seq_targets": seq_targets,
            "temporal_embeddings": t_embeddings,
            "temporal_targets": t_targets,
            "targets": targets,
            "logits": logits,
        }

    def get_loggers(self):
        return {
            "console": dl.ConsoleLogger(),
            "wandb": dl.WandbLogger(
                project="wandb_test", name="abide_experiment_new N_CANNELS = " + str(args.N)
            ),
        }


def main(use_ml: bool = False):

    print("N_CHANNEL = " + str(N_CHANNEL))
    # data
    features, labels = load_ABIDE1()
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )
    n_quantiles = 10
    n_offset = 3  # 0 - pad, 1 - cls, 2 - mask
    transform = TSQuantileTransformer(n_quantiles=n_quantiles, random_state=42)
    transform = transform.fit(X_train)
    X_train2 = transform.transform(X_train) + n_offset
    X_test2 = transform.transform(X_test) + n_offset

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
    assert N_CHANNEL >= 1 and N_CHANNEL <= 53
    model = TemporalModel(
        num_tokens=n_quantiles + 1 + n_offset,
        max_seq_len=N_CHANNEL * 140,  # prior: num_challens * seq_len
        emb_features=16,
        out_features=2,
    )
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.3)

    criterion_mlm = nn.CrossEntropyLoss(ignore_index=0)  # TODO: use from mlm
    criterion_ce = nn.CrossEntropyLoss()
    sampler_inbatch = HardTripletsSampler()
    criterion_ml = TripletMarginLossWithSampler(margin=0.5, sampler_inbatch=sampler_inbatch)
    criterion = {"ce": criterion_ce, "ml": criterion_ml, "mlm": criterion_mlm}

    # runner
    runner = CustomRunner()

    use_ml = False
    # callbacks
    callbacks = [
        dl.CriterionCallback(
            input_key="logits",
            target_key="targets",
            metric_key="loss_ce",
            criterion_key="ce",
        ),
        dl.CriterionCallback(
            input_key="seq_logits",
            target_key="seq_targets",
            metric_key="loss_mlm",
            criterion_key="mlm",
        ),
        dl.MetricAggregationCallback(
            metric_key="loss",
            metrics=["loss_ce", "loss_mlm"],
            mode="mean",
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
    parser.add_argument("N", action="store", type=int)
    utils.boolean_flag(parser, "use-ml", default=False)
    args = parser.parse_args()
    N_CHANNEL = args.N
    main(args.use_ml)
