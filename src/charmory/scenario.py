"""
Primary class for scenario
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import Any, Optional

from tqdm import tqdm

import armory
from armory import metrics
from armory.instrument import MetricsLogger, del_globals, get_hub, get_probe
from armory.instrument.export import ExportMeter, PredictionMeter
from armory.logs import log
from armory.metrics import compute
from armory.paths import HostPaths
from armory.utils import config_loading
import armory.version
from charmory.evaluation import Evaluation


class Scenario(ABC):
    """
    Contains the configuration and helper classes needed to execute an Amory evaluation.
    This is the base class of specific tasks like ImageClassificationTask and
    provides significant common processing.
    """

    @dataclass
    class Batch:
        """Iteration batch being evaluated during scenario evaluation"""

        i: int
        x: Any
        y: Any
        y_pred: Optional[Any] = None
        y_target: Optional[Any] = None
        x_adv: Optional[Any] = None
        y_pred_adv: Optional[Any] = None
        misclassified: Optional[bool] = None

    # change name and type of config to evaluation
    def __init__(
        self,
        evaluation: Evaluation,
        check_run: bool = False,
        skip_benign: bool = False,
        skip_attack: bool = False,
        skip_misclassified: bool = False,
    ):
        # Set up instrumentation
        self.probe = get_probe("scenario")
        self.hub = get_hub()

        self.check_run = check_run
        self.evaluation = evaluation
        self.num_eval_batches = None
        self.skip_benign = skip_benign
        self.skip_attack = skip_attack
        self.skip_misclassified = skip_misclassified

        # Set export paths
        self.time_stamp = time.time()
        self.evaluation_id = str(self.time_stamp)
        self.export_dir = f"{HostPaths().output_dir}/{self.evaluation_id}/outputs"
        self.export_subdir = "saved_samples"
        self.results = None

        # Set up the model
        self.model = self.evaluation.model.model
        self.defense_type = self.apply_defense_to_model()

        # Set up the dataset(s)
        self.test_dataset = self.evaluation.dataset.test_dataset

        # Load the attack
        # NOTE: This is somtimes called in the subclass constructor(super().__init__),
        # and contains attributes that are used for exporting metrics. -CW
        self.load_attack()

        # Load the defense
        # TODO: Better interface for defense. -CW

        # Load Metrics
        self.load_metrics()

        # Load Export Meters
        self.num_export_batches = 0
        if self.evaluation.scenario.export_batches:
            self.num_export_batches = len(self.test_dataset)
            # Create the export directory
            Path(self.export_dir).mkdir(parents=True, exist_ok=True)
            self.sample_exporter = self._load_sample_exporter()
        else:
            self.sample_exporter = None

        self.load_export_meters(self.num_export_batches, self.sample_exporter)

    def evaluate(self):
        """
        Evaluate a config for robustness against attack and save results JSON
        """
        try:
            self.evaluate_all()
            self.finalize_results()
            log.debug("Clearing global instrumentation variables")
            del_globals()
        except Exception as e:
            if str(e) == "assignment destination is read-only":
                log.exception(
                    "Encountered error during scenario evaluation. Be sure "
                    + "that the classifier's predict() isn't directly modifying the "
                    + "input variable itself, as this can cause unexpected behavior in ART."
                )
            else:
                log.exception("Encountered error during scenario evaluation.")
            log.exception(str(e))
            sys.exit(1)

        if self.results is None:
            log.warning("self.results is not a dict")

        if not hasattr(self, "results"):
            raise AttributeError(
                "Results have not been finalized. Please call "
                "finalize_results() before saving output."
            )

        return {
            "armory_version": armory.version.__version__,
            "evaluation": self.evaluation,
            "results": self.results,
            "timestamp": int(self.time_stamp),
        }

    def evaluate_all(self):
        log.info("Running inference on benign and adversarial examples")
        batch = None
        for _ in tqdm(range(len(self.test_dataset)), desc="Evaluation"):
            batch = self.next(batch)
            self.evaluate_current(batch)
        self.hub.set_context(stage="finished")

    def apply_defense_to_model(self):
        if self.evaluation.defense is not None:
            defense_config = self.evaluation.defense or {}
            defense_type = defense_config.get("type")
            if defense_type in ["Preprocessor", "Postprocessor"]:
                log.info(f"Applying internal {defense_type} defense to model")
                self.model = config_loading.load_defense_internal(
                    defense_config, self.model
                )
            elif defense_type == "Trainer":
                self.trainer = config_loading.load_defense_wrapper(
                    defense_config, self.model
                )
            elif defense_type is not None:
                raise ValueError(f"{defense_type} not currently supported")
        else:
            log.info("Not loading any defenses for model")
            defense_type = None

        return defense_type

    def load_attack(self):
        attack_config = self.evaluation.attack
        attack_type = attack_config.type
        if attack_type == "preloaded" and self.skip_misclassified:
            raise ValueError("Cannot use skip_misclassified with preloaded dataset")

        kwargs_val = attack_config.kwargs

        if "summary_writer" in kwargs_val:
            summary_writer_kwarg = attack_config.kwargs.get("summary_writer")
            if isinstance(summary_writer_kwarg, str):
                log.warning(
                    f"Overriding 'summary_writer' attack kwarg {summary_writer_kwarg} with {self.scenario_output_dir}."
                )
            attack_config.kwargs[
                "summary_writer"
            ] = f"{self.scenario_output_dir}/tfevents_{self.time_stamp}"
        if attack_type == "preloaded":
            preloaded_split = kwargs_val.get("split", "adversarial")
            self.test_dataset = config_loading.load_adversarial_dataset(
                attack_config,
                epochs=1,
                split=preloaded_split,
                num_batches=self.num_eval_batches,
                shuffle_files=False,
            )

            targeted = False

        else:
            attack = config_loading.load_attack(attack_config, self.model)
            self.attack = attack

            targeted_val = attack_config.kwargs

            targeted = targeted_val.get("targeted", False)
            if targeted:
                label_targeter = config_loading.load_label_targeter(
                    attack_config.targeted_labels
                )

        use_label = bool(attack_config.use_label)
        if targeted and use_label:
            raise ValueError("Targeted attacks cannot have 'use_label'")

        generate_kwargs = {}

        self.attack_type = attack_type
        self.targeted = targeted
        if self.targeted:
            self.label_targeter = label_targeter
        self.use_label = use_label
        self.generate_kwargs = generate_kwargs

    def load_metrics(self):
        if not hasattr(self, "targeted"):
            log.warning(
                "Run 'load_attack' before 'load_metrics' if not just doing benign inference"
            )

        metrics_config = self.evaluation.metric
        metrics_logger = MetricsLogger.from_config(
            metrics_config,
            include_benign=not self.skip_benign,
            include_adversarial=not self.skip_attack,
            include_targeted=self.targeted,
        )
        self.profiler = compute.profiler_from_config(metrics_config)
        self.metrics_logger = metrics_logger

    def load_export_meters(self, num_export_batches: int, sample_exporter):
        # The export_samples field was deprecated in Armory 0.15.0. Please use scenario.export_batches instead.
        for probe_value in ["x", "x_adv"]:
            export_meter = ExportMeter(
                f"{probe_value}_exporter",
                sample_exporter,
                f"scenario.{probe_value}",
                max_batches=num_export_batches,
            )
            self.hub.connect_meter(export_meter, use_default_writers=False)
            if self.skip_attack:
                break

        pred_meter = PredictionMeter(
            "pred_dict_exporter",
            self.export_dir,
            y_probe="scenario.y",
            y_pred_clean_probe="scenario.y_pred" if not self.skip_benign else None,
            y_pred_adv_probe="scenario.y_pred_adv" if not self.skip_attack else None,
            max_batches=num_export_batches,
        )
        self.hub.connect_meter(pred_meter, use_default_writers=False)

    @abstractmethod
    def _load_sample_exporter(self):
        raise NotImplementedError(
            f"_load_sample_exporter() method is not implemented for scenario {self.__class__}"
        )

    def next(self, prev: Optional[Batch]):
        self.hub.set_context(stage="next")
        x, y = next(self.test_dataset)
        i = prev.i + 1 if prev else 0
        self.hub.set_context(batch=i)
        self.probe.update(i=i, x=x, y=y)
        return self.Batch(i=i, x=x, y=y)

    def run_benign(self, batch: Batch):
        self.hub.set_context(stage="benign")

        batch.x.flags.writeable = False
        with self.profiler.measure("Inference"):
            batch.y_pred = self.model.predict(
                batch.x, **self.evaluation.model.predict_kwargs
            )
        self.probe.update(y_pred=batch.y_pred)

        if self.skip_misclassified:
            batch.misclassified = not any(
                metrics.task.batch.categorical_accuracy(batch.y, batch.y_pred)
            )

    def run_attack(self, batch: Batch):
        self.hub.set_context(stage="attack")
        x = batch.x
        y = batch.y
        y_pred = batch.y_pred

        with self.profiler.measure("Attack"):
            if self.skip_misclassified and batch.misclassified:
                y_target = None

                x_adv = x
            elif self.attack_type == "preloaded":
                if self.targeted:
                    y, y_target = y
                else:
                    y_target = None

                if len(x) == 2:
                    x, x_adv = x
                else:
                    x_adv = x
            else:
                if self.use_label:
                    y_target = y
                elif self.targeted:
                    y_target = self.label_targeter.generate(y)
                else:
                    y_target = None

                x_adv = self.attack.generate(x=x, y=y_target, **self.generate_kwargs)

        self.hub.set_context(stage="adversarial")
        if self.skip_misclassified and batch.misclassified:
            y_pred_adv = y_pred
        else:
            # Ensure that input sample isn't overwritten by model
            x_adv.flags.writeable = False
            y_pred_adv = self.model.predict(
                x_adv, **self.evaluation.model.predict_kwargs
            )

        self.probe.update(x_adv=x_adv, y_pred_adv=y_pred_adv)
        if self.targeted:
            self.probe.update(y_target=y_target)

        batch.x_adv = x_adv
        batch.y_target = y_target
        batch.y_pred_adv = y_pred_adv

    def evaluate_current(self, batch: Batch):
        if not self.skip_benign:
            self.run_benign(batch)
        if not self.skip_attack:
            self.run_attack(batch)

    def finalize_results(self):
        self.metric_results = self.metrics_logger.results()
        self.compute_results = self.profiler.results()
        self.results = {}
        self.results.update(self.metric_results)
        self.results["compute"] = self.compute_results
