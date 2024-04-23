"""Armory engine to perform model robustness evaluations"""

from contextlib import nullcontext
from typing import Any, Dict, Mapping, Optional, TypedDict

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.utilities import rank_zero_only
import mlflow
from torch import Tensor

from armory.engine.evaluation_module import EvaluationModule
from armory.evaluation import Chain, Evaluation, SysConfig
from armory.metrics.compute import NullProfiler, Profiler
from armory.track import get_current_params, init_tracking_uri, track_system_metrics
import armory.version


class EvaluationResults(TypedDict):
    """Robustness evaluation results"""

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
        # run_id: Optional[str] = None,
        **kwargs,
    ):
        """
        Initializes the engine.

        Args:
            evaluation: Configuration for the evaluation
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

    @rank_zero_only
    def _create_nested_run_ids(
        self, logger: pl_loggers.MLFlowLogger
    ) -> Optional[Dict[str, str]]:
        """Create nested MLFlow run IDs for each perturbation chain"""
        run_ids = {}
        with mlflow.start_run(run_id=logger.run_id):
            for chain_name in self.evaluation.chains:
                with mlflow.start_run(
                    experiment_id=logger.experiment_id, run_name=chain_name, nested=True
                ) as run:
                    run_ids[chain_name] = run.info.run_id
        return run_ids

    def _track_system_metrics(self, run_id: Optional[str]):
        """Track system metrics with MLFlow"""
        # run_id will only be valid if we are running on the rank zero node
        if run_id is None:
            return nullcontext()
        return track_system_metrics(run_id)

    def run(self) -> Dict[str, EvaluationResults]:
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

        results: Dict[str, EvaluationResults] = {}
        chain_run_ids = self._create_nested_run_ids(logger) or {}
        for chain_name, chain in self.evaluation.chains.items():
            chain_run_id = chain_run_ids.get(chain_name, None)
            results[chain_name] = self._evaluate_chain(chain_run_id, chain_name, chain)
        return results

    def _evaluate_chain(
        self, chain_run_id: Optional[str], chain_name: str, chain: Chain
    ) -> EvaluationResults:
        assert chain.dataset

        logger = pl_loggers.MLFlowLogger(
            run_id=chain_run_id,
            tracking_uri=init_tracking_uri(self.sysconfig.armory_home),
        )
        self._log_params(logger, chain.get_tracked_params())

        with self._track_system_metrics(logger.run_id):
            module = EvaluationModule(chain, self.profiler)
            trainer = pl.Trainer(
                inference_mode=False,
                logger=logger,
                **self.trainer_kwargs,
            )
            trainer.test(module, dataloaders=chain.dataset.dataloader)

        return EvaluationResults(
            compute=self.profiler.results(),
            metrics=trainer.callback_metrics,
        )
