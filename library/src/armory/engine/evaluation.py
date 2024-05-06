"""Armory engine to perform model robustness evaluations"""

from contextlib import nullcontext
from typing import Any, Dict, Mapping, Optional, TypedDict

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.utilities import rank_zero_only
from torch import Tensor

from armory.engine.evaluation_module import EvaluationModule
from armory.evaluation import Chain, Evaluation, SysConfig
from armory.metrics.compute import NullProfiler, Profiler
from armory.track import get_current_params, init_tracking_uri, track_system_metrics
import armory.version


class EvaluationResults(TypedDict):
    """Robustness evaluation results"""

    run_id: str
    """MLFlow run ID"""
    compute: Mapping[str, float]
    """Computational metrics"""
    metrics: Mapping[str, Tensor]
    """Task-specific evaluation metrics"""


class EvaluationEngine:
    """
    Armory engine to perform model robustness evaluations.

    Example::

        from charmory.engine import EvaluationEngine

        # assuming `evaluation` has been defined using `charmory.evaluation.Evaluation`
        engine = EvaluationEngine(evaluation)
        results = engine.run()
    """

    def __init__(
        self,
        evaluation: Evaluation,
        profiler: Optional[Profiler] = None,
        sysconfig: Optional[SysConfig] = None,
        **kwargs,
    ):
        """
        Initializes the engine.

        Args:
            evaluation: Configuration for the evaluation
            profiler: Optional, profiler to collect computational metrics. By
                default, no computational metrics will be collected.
            sysconfig: Optional, custom system configuration
            run_id: Optional, MLflow run ID to which to record evaluation results
            **kwargs: All other keyword arguments will be forwarded to the
                `lightning.pytorch.Trainer` class.
        """
        self.evaluation = evaluation
        self.profiler = profiler or NullProfiler()
        self.sysconfig = sysconfig or SysConfig()
        self.trainer_kwargs = kwargs
        self._was_run = False

    @rank_zero_only
    def _log_params(self, logger: pl_loggers.MLFlowLogger, params: Dict[str, Any]):
        """Log tracked params with MLFlow"""
        params["Armory.version"] = armory.version.__version__
        logger.log_hyperparams(params)

    def _track_system_metrics(self, run_id: Optional[str]):
        """Track system metrics with MLFlow"""
        # run_id will only be valid if we are running on the rank zero node
        if run_id is None:
            return nullcontext()
        return track_system_metrics(run_id)

    def run(self, verbose: bool = False) -> Dict[str, EvaluationResults]:
        """Perform the evaluation"""
        if self._was_run:
            raise RuntimeError(
                "Evaluation engine has already been run. Create a new EvaluationEngine "
                "instance to perform a subsequent run."
            )
        self._was_run = True

        logger = pl_loggers.MLFlowLogger(
            experiment_name=self.evaluation.name,
            tags={"mlflow.note.content": self.evaluation.description},
            tracking_uri=init_tracking_uri(self.sysconfig.armory_home),
        )
        self.run_id = logger.run_id
        self._log_params(logger, get_current_params())

        try:
            results: Dict[str, EvaluationResults] = {}
            with self._track_system_metrics(self.run_id):
                for chain_name, chain in self.evaluation.chains.items():
                    results[chain_name] = self._evaluate_chain(
                        logger.run_id, chain_name, chain, verbose
                    )
            logger.finalize()

        except BaseException as err:
            logger.finalize("failed")
            raise RuntimeError(
                f"Error performing evaluation of chain {chain_name}"
            ) from err

        return results

    def _evaluate_chain(
        self,
        parent_run_id: Optional[str],
        chain_name: str,
        chain: Chain,
        verbose: bool = False,
    ) -> EvaluationResults:
        assert chain.dataset

        logger = pl_loggers.MLFlowLogger(
            experiment_name=self.evaluation.name,
            run_name=chain_name,
            tracking_uri=init_tracking_uri(self.sysconfig.armory_home),
            tags={"mlflow.parentRunId": parent_run_id},
        )
        self._log_params(logger, chain.get_tracked_params())

        module = EvaluationModule(chain, self.profiler)
        trainer = pl.Trainer(
            inference_mode=False,
            logger=logger,
            **self.trainer_kwargs,
        )
        self.profiler.reset()
        with self._track_system_metrics(logger.run_id):
            trainer.test(module, dataloaders=chain.dataset.dataloader, verbose=verbose)

        profiler_results = self.profiler.results()
        module.sink.log_dict(dict(profiler_results), "profiler_results.txt")

        return EvaluationResults(
            run_id=logger.run_id or "",
            compute=profiler_results,
            metrics=trainer.callback_metrics,
        )
