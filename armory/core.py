import os
import subprocess

import armory
from armory import paths
from armory.logs import log


class ArmoryInstance(object):
    """
    This object will control a specific docker container.
    """

    def __init__(
        self,
        image_name,
        runtime: str = None,
        envs: dict = None,
        ports: dict = None,
        command: str = "tail -f /dev/null",
        user: str = "",
    ):
        # self.docker_client = docker.from_env(version="auto")

        host_paths = paths.HostPaths()
        docker_paths = paths.DockerPaths()

        # mounts = [
        #     docker.types.Mount(
        #         source=getattr(host_paths, dir),
        #         target=getattr(docker_paths, dir),
        #         type="bind",
        #         read_only=False,
        #     )
        #     for dir in "cwd dataset_dir local_git_dir output_dir saved_model_dir tmp_dir".split()
        # ]

        # container_args = {
        #     "runtime": runtime,
        #     "remove": True,
        #     "detach": True,
        #     "mounts": mounts,
        #     "shm_size": "16G",
        # }

        # if ports is not None:
        #     container_args["ports"] = ports
        # if command is not None:
        #     container_args["command"] = command
        # if user:
        #     container_args["user"] = user
        # if envs:
        #     container_args["environment"] = envs
        # self.docker_container = self.docker_client.containers.run(
        #     image_name, **container_args
        # )

        # log.info(f"ARMORY Instance {self.docker_container.short_id} created.")

    def exec_cmd(self, cmd: str, user="", expect_sentinel=True) -> int:
        # We would like to check the return code to see if the command ran cleanly,
        #  but `exec_run()` cannot both return the code and stream logs
        # https://docker-py.readthedocs.io/en/stable/containers.html#docker.models.containers.Container.exec_run
        result = self.docker_container.exec_run(
            cmd,
            stdout=True,
            stderr=True,
            stream=True,
            tty=True,
            user=user,
        )

        # the sentinel should be the last output from the container
        # but threading may cause certain warning messages to be printed during container shutdown
        #  ie after the sentinel
        sentinel_found = False
        for out in result.output:
            output = out.decode(encoding="utf-8", errors="replace").strip()
            if not output:  # skip empty lines
                continue
            # this looks absurd, but in some circumstances result.output will combine
            #  outputs from the container into a single string
            # eg, print(a); print(b) is delivered as 'a\r\nb'
            for inner_line in output.splitlines():
                inner_output = inner_line.strip()
                if not inner_output:
                    continue
                print(inner_output)
                if inner_output == armory.END_SENTINEL:
                    sentinel_found = True

        # if we're not running a config (eg armory exec or launch)
        #  we don't expect the sentinel to be printed and we have no way of
        #  knowing if the command ran cleanly so we return unconditionally
        if not expect_sentinel:
            return 0
        if sentinel_found:
            log.success("command exited cleanly")
            return 0
        else:
            log.error(f"command {cmd} did not finish cleanly")
            return 1

    def __del__(self):
        ...
        # # Needed if there is an error in __init__
        # if hasattr(self, "docker_container"):
        #     self.docker_container.stop()


class HostArmoryInstance:
    def __init__(self, envs: dict = None):
        self.env = os.environ
        for k, v in envs.items():
            self.env[k] = v

    # TODO: Refactor -CW
    def exec_cmd(self, cmd: str, user=""):
        if user:
            raise ValueError("HostArmoryInstance does not support the user input")
        completion = subprocess.run(cmd, env=self.env, shell=True)
        if completion.returncode:
            log.error(f"command {cmd} did not finish cleanly")
        else:
            log.success("command exited cleanly")
        return completion.returncode


"""
Evaluators control launching of ARMORY evaluations.
"""
import base64
import datetime
import json
# import os
import shutil
import sys
import time

import requests

# import armory
from armory import environment, paths
from armory.configuration import load_global_config
from armory.core import HostArmoryInstance, ArmoryInstance
from armory.logs import added_filters, is_debug, log
from armory.utils.printing import bold, red


class Evaluator(object):
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

        self.config["eval_id"] = eval_id
        self.output_dir = os.path.join(self.host_paths.output_dir, eval_id)
        self.tmp_dir = os.path.join(self.host_paths.tmp_dir, eval_id)

        kwargs = dict(image_name=None)
        self.no_docker = True
        self.root = False

        # Retrieve environment variables that should be used in evaluation
        log.info("Retrieving Environment Variables")
        self.extra_env_vars = dict()
        self._gather_env_variables()

        self.manager = HostArmoryInstance
        # self.manager = HostManagementInstance()


    def _gather_env_variables(self):
        """
        Update the extra env variable dictionary to pass into container or run on host
        """
        self.extra_env_vars["ARMORY_GITHUB_TOKEN"] = os.getenv(
            "ARMORY_GITHUB_TOKEN", default=""
        )
        self.extra_env_vars["ARMORY_PRIVATE_S3_ID"] = os.getenv(
            "ARMORY_PRIVATE_S3_ID", default=""
        )
        self.extra_env_vars["ARMORY_PRIVATE_S3_KEY"] = os.getenv(
            "ARMORY_PRIVATE_S3_KEY", default=""
        )
        self.extra_env_vars["ARMORY_INCLUDE_SUBMISSION_BUCKETS"] = os.getenv(
            "ARMORY_INCLUDE_SUBMISSION_BUCKETS", default=""
        )

        if not self.armory_global_config["verify_ssl"]:
            self.extra_env_vars["VERIFY_SSL"] = "false"

        if self.config["sysconfig"].get("use_gpu", None):
            gpus = self.config["sysconfig"].get("gpus")
            if gpus is not None:
                self.extra_env_vars["NVIDIA_VISIBLE_DEVICES"] = gpus
        if self.config["sysconfig"].get("set_pythonhashseed"):
            self.extra_env_vars["PYTHONHASHSEED"] = "0"

        if not self.no_docker:
            self.extra_env_vars["HOME"] = "/tmp"

        # Because we may want to allow specification of ARMORY_TORCH_HOME
        # this constant path is placed here among the other imports
        if self.no_docker:
            torch_home = paths.HostPaths().pytorch_dir
        else:
            torch_home = paths.DockerPaths().pytorch_dir
        self.extra_env_vars["TORCH_HOME"] = torch_home

        self.extra_env_vars[environment.ARMORY_VERSION] = armory.__version__

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

    def run(
        self,
        interactive=False, # TODO: Remove -CW
        jupyter=False,     # TODO: Remove -CW
        host_port=None,    # TODO: Remove -CW
        command=None,      # TODO: Remove -CW
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
            exit_code = self._run_config(
                runner,
                check_run=check_run,
                num_eval_batches=num_eval_batches,
                skip_benign=skip_benign,
                skip_attack=skip_attack,
                skip_misclassified=skip_misclassified,
                validate_config=validate_config,
            )
        except KeyboardInterrupt:
            log.warning("Keyboard interrupt caught")
        finally:
            log.info("cleaning up...")
        self._cleanup()
        return exit_code

    def _b64_encode_config(self):
        bytes_config = json.dumps(self.config).encode("utf-8")
        base64_bytes = base64.b64encode(bytes_config)
        return base64_bytes.decode("utf-8")

    def _run_config(
        self,
        runner: ArmoryInstance,
        check_run=False,
        num_eval_batches=None,
        skip_benign=None,
        skip_attack=None,
        skip_misclassified=None,
        validate_config=None,
    ) -> int:
        log.info(bold(red("Running evaluation script")))

        b64_config = self._b64_encode_config()
        options = self._build_options(
            check_run=check_run,
            num_eval_batches=num_eval_batches,
            skip_benign=skip_benign,
            skip_attack=skip_attack,
            skip_misclassified=skip_misclassified,
            validate_config=validate_config,
        )
        # if self.no_docker:
        kwargs = {}
        python = sys.executable
        # else:
        #     kwargs = {"user": self.get_id()}
        #     python = "python"

        cmd = f"{python} -m armory.scenarios.main {b64_config}{options} --base64"
        return runner.exec_cmd(cmd, **kwargs)

    def _run_command(self, runner: ArmoryInstance, command: str) -> int:
        log.info(bold(red(f"Running bash command: {command}")))
        return runner.exec_cmd(command, user=self.get_id(), expect_sentinel=False)

    def get_id(self):
        """
        Return uid, gid
        """
        # Windows docker does not require synchronizing file and
        # directory permissions via uid and gid.
        if os.name == "nt" or self.root:
            user_id = 0
            group_id = 0
        else:
            user_id = os.getuid()
            group_id = os.getgid()
        return f"{user_id}:{group_id}"

    def _run_interactive_bash(
        self,
        runner: ArmoryInstance,
        check_run=False,
        num_eval_batches=None,
        skip_benign=None,
        skip_attack=None,
        skip_misclassified=None,
        validate_config=None,
    ) -> None:
        user_group_id = self.get_id()
        lines = [
            "Container ready for interactive use.",
            bold("# In a new terminal, run the following to attach to the container:"),
            bold(
                red(
                    f"docker exec -it -u {user_group_id} {runner.docker_container.short_id} bash"
                )
            ),
            "",
        ]
        if self.config.get("scenario"):
            options = self._build_options(
                check_run=check_run,
                num_eval_batches=num_eval_batches,
                skip_benign=skip_benign,
                skip_attack=skip_attack,
                skip_misclassified=skip_misclassified,
                validate_config=validate_config,
            )
            init_options = self._constructor_options(
                check_run=check_run,
                num_eval_batches=num_eval_batches,
                skip_benign=skip_benign,
                skip_attack=skip_attack,
                skip_misclassified=skip_misclassified,
            )

            tmp_dir = os.path.join(self.host_paths.tmp_dir, self.config["eval_id"])
            os.makedirs(tmp_dir)
            self.tmp_config = os.path.join(tmp_dir, "interactive-config.json")
            docker_config_path = os.path.join(
                paths.runtime_paths().tmp_dir,
                self.config["eval_id"],
                "interactive-config.json",
            )
            with open(self.tmp_config, "w") as f:
                f.write(json.dumps(self.config, sort_keys=True, indent=4) + "\n")

            lines.extend(
                [
                    bold("# To run your scenario in the container:"),
                    bold(
                        red(
                            f"python -m armory.scenarios.main {docker_config_path}{options}"
                        )
                    ),
                    "",
                    bold("# To run your scenario interactively:"),
                    bold(
                        red(
                            "python\n"
                            "from armory.scenarios.main import get as get_scenario\n"
                            f's = get_scenario("{docker_config_path}"{init_options}).load()\n'
                            "s.evaluate()"
                        )
                    ),
                    "",
                    bold("# To gracefully shut down container, press: Ctrl-C"),
                    "",
                ]
            )
        log.info("\n".join(lines))
        while True:
            time.sleep(1)

    # def _run_jupyter(
    #     self,
    #     runner: ArmoryInstance,
    #     ports: dict,
    #     check_run=False,
    #     num_eval_batches=None,
    #     skip_benign=None,
    #     skip_attack=None,
    #     skip_misclassified=None,
    # ) -> None:
    #     if not self.root:
    #         log.warning("Running Jupyter Lab as root inside the container.")

    #     user_group_id = self.get_id()
    #     port = list(ports.keys())[0]
    #     tmp_dir = os.path.join(self.host_paths.tmp_dir, self.config["eval_id"])
    #     os.makedirs(tmp_dir)
    #     self.tmp_config = os.path.join(tmp_dir, "interactive-config.json")
    #     docker_config_path = os.path.join(
    #         paths.runtime_paths().tmp_dir,
    #         self.config["eval_id"],
    #         "interactive-config.json",
    #     )
    #     with open(self.tmp_config, "w") as f:
    #         f.write(json.dumps(self.config, sort_keys=True, indent=4) + "\n")
    #     init_options = self._constructor_options(
    #         check_run=check_run,
    #         num_eval_batches=num_eval_batches,
    #         skip_benign=skip_benign,
    #         skip_attack=skip_attack,
    #         skip_misclassified=skip_misclassified,
    #     )
    #     lines = [
    #         "About to launch jupyter.",
    #         bold("# To connect on the command line as well, in a new terminal, run:"),
    #         bold(
    #             red(
    #                 f"docker exec -it -u {user_group_id} {runner.docker_container.short_id} bash"
    #             )
    #         ),
    #         "",
    #     ]
    #     if "scenario" in self.config:
    #         # If not, config is not valid to load into scenario
    #         lines.extend(
    #             [
    #                 bold("# To run, inside of a notebook:"),
    #                 bold(
    #                     red(
    #                         "from armory.scenarios.main import get as get_scenario\n"
    #                         f's = get_scenario("{docker_config_path}"{init_options}).load()\n'
    #                         "s.evaluate()"
    #                     )
    #                 ),
    #                 "",
    #             ]
    #         )
    #     lines.extend(
    #         [
    #             bold("# To gracefully shut down container, press: Ctrl-C"),
    #             "",
    #             "Jupyter notebook log:",
    #         ]
    #     )
    #     log.info("\n".join(lines))
    #     runner.exec_cmd(
    #         f"jupyter lab --ip=0.0.0.0 --port {port} --no-browser",
    #         user=user_group_id,
    #         expect_sentinel=False,
    #     )

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
        # if self.no_docker:
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

    def _constructor_options(
        self,
        check_run=False,
        num_eval_batches=None,
        skip_benign=None,
        skip_attack=None,
        skip_misclassified=None,
    ):
        kwargs = dict(
            check_run=check_run,
            num_eval_batches=num_eval_batches,
            skip_benign=skip_benign,
            skip_attack=skip_attack,
            skip_misclassified=skip_misclassified,
        )
        options = "".join(f", {str(k)}={str(v)}" for k, v in kwargs.items() if v)
        return options
