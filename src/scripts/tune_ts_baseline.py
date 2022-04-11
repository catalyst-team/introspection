import argparse

from animus import IExperiment
from apto.utils.report import get_classification_report
from catalyst import utils
import numpy as np
import optuna
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split

from src.settings import LOGS_ROOT, UTCNOW
from src.ts import load_ABIDE1, TSQuantileTransformer

import wandb
import time
import pdb


class Experiment(IExperiment):
    def __init__(self, quantile: bool) -> None:
        super().__init__()
        self._quantile: bool = quantile
        self._trial: optuna.Trial = None

        # init wandb logger
        self.wandbLogger: wandb.run = wandb.init(
            project="tune_ts_baseline", name="tune_ts_baseline"
        )
        # for timer
        self.start: float = 0.0

    def on_tune_start(self):
        features, labels = load_ABIDE1()
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        if self._quantile:
            n_quantiles = 10
            n_offset = 3  # 0 - pad, 1 - cls, 2 - mask
            transform = TSQuantileTransformer(n_quantiles=n_quantiles, random_state=42)
            transform = transform.fit(X_train)
            X_train = transform.transform(X_train) + n_offset
            X_test = transform.transform(X_test) + n_offset

        X_train = np.reshape(X_train, (len(X_train), -1))
        X_test = np.reshape(X_test, (len(X_test), -1))
        self.datasets = {
            "ABIDE1": (X_train, X_test, y_train, y_test),
        }

    def on_experiment_start(self, exp: "IExperiment"):
        super().on_experiment_start(exp)
        # setup model
        clf_type = self._trial.suggest_categorical(
            "classifier",
            choices=[
                "LogisticRegression",
                "SGDClassifier",
                "AdaBoostClassifier",
                "RandomForestClassifier",
            ],
        )

        if clf_type == "LogisticRegression":
            solver = self._trial.suggest_categorical(
                "classifier.logistic.solver", ["liblinear", "lbfgs"]
            )
            decay = self._trial.suggest_loguniform("classifier.logistic.C", low=1e-3, high=1e3)
            if solver == "liblinear":
                penalty = self._trial.suggest_categorical(
                    "classifier.logistic.penalty", ["l1", "l2"]
                )
            else:
                penalty = "l2"

            self.classifier = LogisticRegression(
                solver=solver, C=decay, penalty=penalty, max_iter=1000
            )
        elif clf_type == "SGDClassifier":
            penalty = self._trial.suggest_categorical(
                "classifier.sgd.penalty", ["l1", "l2", "elasticnet"]
            )
            alpha = self._trial.suggest_loguniform("classifier.sgd.alpha", low=1e-4, high=1e-2)
            self.classifier = SGDClassifier(
                loss="modified_huber",
                penalty=penalty,
                alpha=alpha,
                max_iter=1000,
                tol=1e-3,
            )

        elif clf_type == "AdaBoostClassifier":
            n_estimators = self._trial.suggest_int(
                "classifier.adaboost.n_estimators", 2, 32, log=True
            )
            self.classifier = AdaBoostClassifier(n_estimators=n_estimators)
        elif clf_type == "RandomForestClassifier":
            max_depth = self._trial.suggest_int(
                "classifier.random_forest.max_depth", 2, 32, log=True
            )
            n_estimators = self._trial.suggest_int(
                "classifier.random_forest.n_estimators", 2, 32, log=True
            )
            self.classifier = RandomForestClassifier(
                max_depth=max_depth, n_estimators=n_estimators, max_features=1
            )

    def run_dataset(self) -> None:
        X_train, X_test, y_train, y_test = self.dataset
        self.classifier.fit(X_train, y_train)
        y_pred = self.classifier.predict(X_test)
        y_score = self.classifier.predict_proba(X_test)
        report = get_classification_report(
            y_true=y_test, y_pred=y_pred, y_score=y_score, beta=0.5
        )
        for stats_type in [0, 1, "macro", "weighted"]:
            stats = report.loc[stats_type]
            for key, value in stats.items():
                if "support" not in key:
                    self._trial.set_user_attr(f"{key}_{stats_type}", float(value))
        self.dataset_metrics = {"score": report["auc"].loc["weighted"]}

    def on_experiment_end(self, exp: "IExperiment") -> None:
        super().on_experiment_end(exp)
        # we have only 1 epoch for baselines, so...
        # pdb.set_trace()
        self._score = self.experiment_metrics[1]["ABIDE1"]["score"]

    def _objective(self, trial) -> float:
        # start timer
        self.start = time.process_time()

        self._trial = trial
        self.run()

        # log score and experiment time
        self.wandbLogger.log(
            {
                "overall score": self._score,
                type(self.classifier).__name__ + " score": self._score,
                type(self.classifier).__name__ + " time": time.process_time() - self.start,
            }
        )

        # pdb.set_trace()
        return self._score

    def tune(self, n_trials: int):
        # pdb.set_trace()
        self.on_tune_start()

        # pdb.set_trace()
        self.study = optuna.create_study(direction="maximize")

        # pdb.set_trace()
        self.study.optimize(self._objective, n_trials=n_trials, n_jobs=1)

        # pdb.set_trace()
        logfile = f"{LOGS_ROOT}/{UTCNOW}-ts-baseline-q{self._quantile}.optuna.csv"
        df = self.study.trials_dataframe()
        df.to_csv(logfile, index=False)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser()
    utils.boolean_flag(parser, "quantile", default=False)
    parser.add_argument("--num-trials", type=int, default=1)
    args = parser.parse_args()
    Experiment(quantile=args.quantile).tune(n_trials=args.num_trials)
