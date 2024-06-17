"""Armory engine to perform adversarial attack optimization"""

from typing import Optional

import lightning.pytorch as pl
import lightning.pytorch.loggers as pl_loggers
from lightning.pytorch.utilities import rank_zero_only

from armory.engine.optimization_module import OptimizationModule
from armory.evaluation import Optimization, SysConfig
from armory.metrics.compute import NullProfiler, Profiler
from armory.track import get_current_params, init_tracking_uri, track_system_metrics
import armory.version


class OptimizationEngine:
    """
    Armory engine to perform adversarial attack optimization
    """

    def __init__(
        self,
        optimization: Optimization,
        profiler: Optional[Profiler] = None,
        sysconfig: Optional[SysConfig] = None,
        **kwargs,
    ):
        """
        Initializes the optimization engine.

        Args:
            optimization: Configuration for the optimization
            sysconfig: Optional, custom system configuration
            **kwargs: All other keyword arguments will be forwarded to the
                `lightning.pytorch.Trainer` class.
        """
        self.optimization = optimization
        self.profiler = profiler or NullProfiler()
        self.sysconfig = sysconfig or SysConfig()
        self.trainer_kwargs = kwargs
        self._was_run = False

    @rank_zero_only
    def _log_params(self, logger: pl_loggers.MLFlowLogger):
        """Log tracked params with MLFlow"""
        params = get_current_params()
        params.update(self.optimization.get_tracked_params())
        params["Armory.version"] = armory.version.__version__
        logger.log_hyperparams(params)

    def run(self) -> None:
        """Perform the optimization"""
        if self._was_run:
            raise RuntimeError(
                "Optimization engine has already been run. Create a new OptimizationEngine "
                "instance to perform a subsequent run."
            )
        self._was_run = True

        logger = pl_loggers.MLFlowLogger(
            experiment_name=self.optimization.name,
            tags={"mlflow.note.content": self.optimization.description},
            tracking_uri=init_tracking_uri(self.sysconfig.armory_home),
        )
        self.run_id = logger.run_id
        self._log_params(logger)

        module = OptimizationModule(self.optimization)
        trainer = pl.Trainer(
            inference_mode=False,
            logger=logger,
            **self.trainer_kwargs,
        )
        self.profiler.reset()
        with track_system_metrics(logger.run_id):
            trainer.fit(module, train_dataloaders=self.optimization.dataset.dataloader)

        profiler_results = self.profiler.results()
        module.sink.log_dict(dict(profiler_results), "profiler_results.txt")
