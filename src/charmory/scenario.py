"""
Primary class for scenario
"""

import copy
import dataclasses
import sys
import time

from tqdm import tqdm

import armory
from armory import metrics
from armory.instrument import MetricsLogger, del_globals, get_hub, get_probe
from armory.instrument.export import ExportMeter, PredictionMeter
from armory.logs import log
from armory.metrics import compute
from armory.utils import config_loading
import armory.version
from charmory.evaluation import Evaluation

class Scenario:
    """
    Contains the configuration and helper classes needed to execute an Amory evaluation.
    This is the base class of specific tasks like ImageClassificationTask and
    provides significant common processing.
    """
    #change name and type of config to evaluation
    def __init__(
        self,
        evaluation: Evaluation,
        check_run: bool = False,
    ):
        self.check_run = check_run
        self.model = None
        self.dataset = {"test": None, "train": None}
        self.evaluation = evaluation
        self.i = -1
        self.num_eval_batches = None
        self.skip_benign = False
        self.skip_attack = False
        self.skip_misclassified = False

        self.time_stamp = time.time()
        self.export_dir = "/tmp"
        self.export_subdir = "armory"
        self.results = None

        # Load the model
        self._loaded_model = self.load_model()
        self.model = self._loaded_model["model"]
        #self.model_name = f"{self.config['model']['function'].__module__}.{self.config['model']['function'].__name__}"
        self.model_name = f"{self.evaluation.model.function.__module__}.{self.evaluation.model.function.__name__}"
        self.use_fit = self._loaded_model["use_fit"]
        self.fit_kwargs = self._loaded_model["fit_kwargs"]
        self.predict_kwargs = self._loaded_model["predict_kwargs"]
        self.defense_type = self._loaded_model["defense_type"]

        # Load the dataset(s)
        self.dataset = self.load_dataset_config()
        if bool(self.evaluation.model.fit):
            self.train_dataset = self.dataset["train"]
            self.fit()
        self.test_dataset = self.dataset["test"]

        # Load the attack
        self.load_attack()

        # Load the defense
        # TODO: Better interface for defense. -CW

        # Load Metrics
        self.probe = get_probe("scenario")
        self.hub = get_hub()
        self.load_metrics()

        # Load Export Meters
        self.load_export_meters()


    def load_dataset_config(self):
        _dataset = {"test": None, "train": None}

        config = self.evaluation
        config_dataset = config.dataset
        use_fit = config.model.fit

        if hasattr(config_dataset, "test"):
            # Dataset is a custom Armory generator
            _dataset["test"] = config_dataset["test"]
            if hasattr(config_dataset, "train"):
                _dataset["train"] = config_dataset["train"]
        else:
            _dataset["test"] = self.load_dataset(config_dataset)
            if bool(use_fit):
                _dataset["train"] = self.load_train_dataset()

        return _dataset

    def evaluate(self):
        """
        Evaluate a config for robustness against attack and save results JSON
        """
        try:
            self.run_inference()
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

        output = {
            "armory_version": armory.version.__version__,
            "config": self.evaluation,
            "results": self.results,
            "timestamp": int(self.time_stamp),
        }
        return output

    def run_inference(self):
        log.info("Running inference on benign and adversarial examples")
        for _ in tqdm(range(len(self.test_dataset)), desc="Evaluation"):
            self.next()
            self.evaluate_current()
        self.hub.set_context(stage="finished")

    def load_model(self, defended=True):
        model_config = self.evaluation.model
        model, _ = config_loading.load_model(model_config)

        if defended:
            defense_config = self.evaluation.defense or {}
            defense_type = defense_config.get("type")
            if defense_type in ["Preprocessor", "Postprocessor"]:
                log.info(f"Applying internal {defense_type} defense to model")
                model = config_loading.load_defense_internal(defense_config, model)
            elif defense_type == "Trainer":
                self.trainer = config_loading.load_defense_wrapper(
                    defense_config, model
                )
            elif defense_type is not None:
                raise ValueError(f"{defense_type} not currently supported")
        else:
            log.info("Not loading any defenses for model")
            defense_type = None
        fit_kwargs_val = {}
        try:
            fit_kwargs_val = model_config.fit_kwargs
        except:
            pass
        
        predict_kwargs_val = {}
        try:
            predict_kwargs_val = model_config.predict_kwargs
        except:
            pass

        return {
            "model": model,
            "defense_type": defense_type,
            "use_fit": bool(model_config.fit),
            "fit_kwargs": fit_kwargs_val,
            "predict_kwargs": predict_kwargs_val,
        }

    def load_train_dataset(self, train_split_default="train"):
        dataset_config = self.evaluation.dataset
        log.info("Loading train dataset...")

        split_val = train_split_default
        try:
            split_val = dataset_config.train_split
        except:
            pass
        return config_loading.load_dataset(
            dataset_config,
            epochs=self.fit_kwargs["nb_epochs"],
            split=split_val,
            check_run=self.check_run,
            shuffle_files=True,
        )

    def fit(self):
        if self.defense_type == "Trainer":
            log.info(f"Training with {type(self.trainer)} Trainer defense...")
            self.trainer.fit_generator(self.train_dataset, **self.fit_kwargs)
        else:
            log.info(f"Fitting model {self.model_name}...")
            self.model.fit_generator(self.train_dataset, **self.fit_kwargs)

    def load_attack(self):
        attack_config = self.evaluation.attack
        attack_type = attack_config.type
        if attack_type == "preloaded" and self.skip_misclassified:
            raise ValueError("Cannot use skip_misclassified with preloaded dataset")

        kwargs_val = {}
        try:
            kwargs_val = attack.config.kwargs
        except:
            pass
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
            preloaded_split = kwargs_val.get(
                "split", "adversarial"
            )
            self.test_dataset = config_loading.load_adversarial_dataset(
                attack_config,
                epochs=1,
                split=preloaded_split,
                num_batches=self.num_eval_batches,
                shuffle_files=False,
            )
            targeted = False
            try:
                targeted = attack_config.targeted
            except:
                pass
        else:
            attack = config_loading.load_attack(attack_config, self.model)
            self.attack = attack
            targeted_val = {}
            try:
                targeted_val = attack_config.kwargs
            except:
                pass
            targeted = targeted_val.get("targeted", False)
            if targeted:
                label_targeter = config_loading.load_label_targeter(
                    attack_config.targeted_labels
                )

        use_label = bool(attack_config.use_label)
        if targeted and use_label:
            raise ValueError("Targeted attacks cannot have 'use_label'")
        
        generate_kwargs = {}
        try:
            generate_kwargs = copy.deepcopy(attack_config.generate_kwargs)
        except:
            pass

        self.attack_type = attack_type
        self.targeted = targeted
        if self.targeted:
            self.label_targeter = label_targeter
        self.use_label = use_label
        self.generate_kwargs = generate_kwargs

    def load_dataset(self, dataset_config=None, eval_split_default="test"):
        dataset_config = (
            self.evaluation.dataset if dataset_config is None else dataset_config
        )
        #eval_split = dataset_config.get("eval_split", eval_split_default)

        eval_split = eval_split_default
        try:
            eval_split = dataset_config.eval_split
        except:
            pass
        # Evaluate the ART model on benign test examples
        log.info("Loading test dataset...")
        return config_loading.load_dataset(
            dataset_config,
            epochs=1,
            split=eval_split,
            num_batches=self.num_eval_batches,
            check_run=self.check_run,
            shuffle_files=False,
        )

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

    def load_export_meters(self):
        #Removed warning logged regarding deprecated field from Armory 0.15.0
        try:
            num_export_batches = self.evaluation.scenario.export_batches
        except:
            num_export_batches = 0
        
        if num_export_batches is True:
            num_export_batches = len(self.test_dataset)
        self.num_export_batches = int(num_export_batches)
        self.sample_exporter = self._load_sample_exporter()

        for probe_value in ["x", "x_adv"]:
            export_meter = ExportMeter(
                f"{probe_value}_exporter",
                self.sample_exporter,
                f"scenario.{probe_value}",
                max_batches=self.num_export_batches,
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
            max_batches=self.num_export_batches,
        )
        self.hub.connect_meter(pred_meter, use_default_writers=False)

    def _load_sample_exporter(self):
        raise NotImplementedError(
            f"_load_sample_exporter() method is not implemented for scenario {self.__class__}"
        )

    def next(self):
        self.hub.set_context(stage="next")
        x, y = next(self.test_dataset)
        i = self.i + 1
        self.hub.set_context(batch=i)
        self.i, self.x, self.y = i, x, y
        self.probe.update(i=i, x=x, y=y)
        self.y_pred, self.y_target, self.x_adv, self.y_pred_adv = None, None, None, None

    def _check_x(self, function_name):
        if not hasattr(self, "x"):
            raise ValueError(f"Run `next()` before `{function_name}()`")

    def run_benign(self):
        self._check_x("run_benign")
        self.hub.set_context(stage="benign")

        x, y = self.x, self.y
        x.flags.writeable = False
        with self.profiler.measure("Inference"):
            y_pred = self.model.predict(x, **self.predict_kwargs)
        self.y_pred = y_pred
        self.probe.update(y_pred=y_pred)

        if self.skip_misclassified:
            self.misclassified = not any(
                metrics.task.batch.categorical_accuracy(y, y_pred)
            )

    def run_attack(self):
        self._check_x("run_attack")
        self.hub.set_context(stage="attack")
        x, y, y_pred = self.x, self.y, self.y_pred

        with self.profiler.measure("Attack"):
            if self.skip_misclassified and self.misclassified:
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
        if self.skip_misclassified and self.misclassified:
            y_pred_adv = y_pred
        else:
            # Ensure that input sample isn't overwritten by model
            x_adv.flags.writeable = False
            y_pred_adv = self.model.predict(x_adv, **self.predict_kwargs)

        self.probe.update(x_adv=x_adv, y_pred_adv=y_pred_adv)
        if self.targeted:
            self.probe.update(y_target=y_target)

        self.x_adv, self.y_target, self.y_pred_adv = x_adv, y_target, y_pred_adv

    def evaluate_current(self):
        self._check_x("evaluate_current")
        if not self.skip_benign:
            self.run_benign()
        if not self.skip_attack:
            self.run_attack()

    def finalize_results(self):
        self.metric_results = self.metrics_logger.results()
        self.compute_results = self.profiler.results()
        self.results = {}
        self.results.update(self.metric_results)
        self.results["compute"] = self.compute_results
