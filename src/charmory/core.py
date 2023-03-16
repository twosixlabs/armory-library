import datetime
import os
import shutil

import armory
from armory import environment, paths
from armory.configuration import load_global_config
from armory.logs import log
from armory.scenarios.main import main as scenario_main
from armory.utils.printing import bold, red


class ArmoryInstance:
    def __init__(self, envs: dict = None):
        self.env = os.environ.copy()
        for k, v in envs.items():
            self.env[str(k)] = str(v)


class Evaluator:
    """
    Evaluators control launching of ARMORY evaluations.
    """

    def __init__(
        self,
        config: dict,
        no_docker: bool = True,
        root: bool = False,
    ):
        log.info("Constructing Evaluator Object")
        if not isinstance(config, dict):
            raise ValueError(f"config {config} must be a dict")
        self.config = config

        self.host_paths = paths.HostPaths()
        if os.path.exists(self.host_paths.armory_config):
            self.armory_global_config = load_global_config(
                self.host_paths.armory_config
            )
        else:
            self.armory_global_config = {"verify_ssl": True}

        # Output directory configuration
        date_time = datetime.datetime.utcnow().isoformat().replace(":", "")
        output_dir = self.config["sysconfig"].get("output_dir", None)
        eval_id = f"{output_dir}_{date_time}" if output_dir else date_time

        self.config["eval_id"] = eval_id
        self.output_dir = os.path.join(self.host_paths.output_dir, eval_id)
        self.tmp_dir = os.path.join(self.host_paths.tmp_dir, eval_id)

        # Retrieve environment variables that should be used in evaluation
        log.info("Retrieving Environment Variables")
        self.extra_env_vars = {
            "ARMORY_GITHUB_TOKEN": os.getenv("ARMORY_GITHUB_TOKEN", default=""),
            "ARMORY_PRIVATE_S3_ID": os.getenv("ARMORY_PRIVATE_S3_ID", default=""),
            "ARMORY_PRIVATE_S3_KEY": os.getenv("ARMORY_PRIVATE_S3_KEY", default=""),
            "ARMORY_INCLUDE_SUBMISSION_BUCKETS": os.getenv(
                "ARMORY_INCLUDE_SUBMISSION_BUCKETS", default=""
            ),
            "VERIFY_SSL": self.armory_global_config["verify_ssl"] or False,
            "NVIDIA_VISIBLE_DEVICES": self.config["sysconfig"].get("gpus", None),
            "PYTHONHASHSEED": self.config["sysconfig"].get("set_pythonhashseed", "0"),
            "TORCH_HOME": paths.HostPaths().pytorch_dir,
            environment.ARMORY_VERSION: armory.__version__,
            # "HOME": "/tmp",
        }

        self.manager = ArmoryInstance(envs=self.extra_env_vars)
        self.no_docker = True
        self.root = False

    def run(
        self,
        command=None,
        check_run=False,
        num_eval_batches=None,
        skip_benign=None,
        skip_attack=None,
        skip_misclassified=None,
        validate_config=None,
    ) -> int:
        exit_code = 0
        try:
            log.info(bold(red("Running Evaluation")))
            scenario_main(self.config)
        except KeyboardInterrupt:
            log.warning("Keyboard interrupt caught")
        finally:
            log.info("cleaning up...")
        self._cleanup()
        return exit_code

    def _cleanup(self):
        log.info(f"deleting tmp_dir {self.tmp_dir}")
        try:
            shutil.rmtree(self.tmp_dir)
        except OSError as e:
            if not isinstance(e, FileNotFoundError):
                log.exception(f"Error removing tmp_dir {self.tmp_dir}")
        try:
            os.rmdir(self.output_dir)
            log.warning(f"removed output_dir {self.output_dir} because it was empty")
        except FileNotFoundError:
            log.warning(f"output_dir {self.output_dir} was deleted or never created")
        except OSError:
            jsons = [x for x in os.listdir(self.output_dir) if x.endswith(".json")]
            json_output = jsons[0] if len(jsons) == 1 else ""
            output_path = os.path.join(self.output_dir, json_output)
            log.info(f"results output written to:\n{output_path}")