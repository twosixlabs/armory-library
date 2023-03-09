import base64
import datetime
import json
import os
import shutil
import subprocess
import sys

import armory
from armory import environment, paths
from armory.configuration import load_global_config
from armory.logs import added_filters, is_debug, log
from armory.utils.printing import bold, red


# TODO: Refactor -CW
class ArmoryInstance:
    def __init__(self, envs: dict = None):
        self.env = os.environ
        for k, v in envs.items():
            self.env[str(k)] = str(v)

    def exec_cmd(self, cmd: str):
        completion = subprocess.run(cmd, env=self.env, shell=True)
        if completion.returncode:
            log.error(f"command {cmd} did not finish cleanly")
        else:
            log.success("command exited cleanly")
        return completion.returncode


class Evaluator(object):
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

        date_time = datetime.datetime.utcnow().isoformat().replace(":", "")
        output_dir = self.config["sysconfig"].get("output_dir", None)
        eval_id = f"{output_dir}_{date_time}" if output_dir else date_time

        self.manager = ArmoryInstance
        self.config["eval_id"] = eval_id
        self.output_dir = os.path.join(self.host_paths.output_dir, eval_id)
        self.tmp_dir = os.path.join(self.host_paths.tmp_dir, eval_id)
        self.no_docker = True
        self.root = False

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
            # "HOME": "/tmp",
            "TORCH_HOME": paths.HostPaths().pytorch_dir,
            environment.ARMORY_VERSION: armory.__version__,
        }

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
        runner = self.manager(envs=self.extra_env_vars)
        try:
            log.info(bold(red("Running evaluation script")))

            bytes_config = json.dumps(self.config).encode("utf-8")
            base64_bytes = base64.b64encode(bytes_config)
            base64_config = base64_bytes.decode("utf-8")

            options = self._build_options(
                check_run=check_run,
                num_eval_batches=num_eval_batches,
                skip_benign=skip_benign,
                skip_attack=skip_attack,
                skip_misclassified=skip_misclassified,
                validate_config=validate_config,
            )

            cmd = f"{sys.executable} -m armory.scenarios.main {base64_config}{options} --base64"
            exit_code = runner.exec_cmd(cmd)

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
            if len(jsons) == 1:
                json = jsons[0]
            else:
                json = ""
            output_path = os.path.join(self.output_dir, json)
            log.info(f"results output written to:\n{output_path}")

    def _build_options(
        self,
        check_run,
        num_eval_batches,
        skip_benign,
        skip_attack,
        skip_misclassified,
        validate_config,
    ):
        options = ""
        options += " --no-docker"

        if check_run:
            options += " --check"
        if is_debug():
            options += " --debug"
        if num_eval_batches:
            options += f" --num-eval-batches {num_eval_batches}"
        if skip_benign:
            options += " --skip-benign"
        if skip_attack:
            options += " --skip-attack"
        if skip_misclassified:
            options += " --skip-misclassified"
        if validate_config:
            options += " --validate-config"
        for module, level in added_filters.items():
            options += f" --log-level {module}:{level}"
        return options
