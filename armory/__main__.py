"""
python -m armory run <json_config>
OR
armory run <json_config>

Try:
    <json_config> = 'examples/fgm_attack_binary_search.json'

This runs an arbitrary config file. Results are output to the `outputs/` directory.
"""


#########################################
# TODO: Remove this temp hack -CW
import torch

torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

if torch.device.type == 'cpu':
    print("WARNING: Running on CPU")
#########################################


import argparse
import json
import os
import re
import sys

from jsonschema import ValidationError

import armory
import armory.logs as logger
from armory import arguments, paths
from armory.configuration import load_global_config, save_config
from armory.eval import Evaluator
from armory.utils.configuration import load_config, load_config_stdin
from armory.utils.version import to_docker_tag


OLD_SCENARIOS = [
    "https://github.com/twosixlabs/armory-example/blob/master/scenario_download_configs/scenarios-set1.json"
]
DEFAULT_SCENARIO = "https://github.com/twosixlabs/armory-example/blob/master/scenario_download_configs/scenarios-set2.json"


def sorted_unique_nonnegative_numbers(values, warning_string):
    if not isinstance(values, str):
        raise ValueError(f"{values} invalid.\n Must be a string input.")

    if not re.match(r"^\s*\d+(\s*,\s*\d+)*\s*$", values):
        raise ValueError(
            f"{values} invalid. Must be ','-separated nonnegative integers"
        )

    numbers = [int(x) for x in values.split(",")]
    sorted_unique_numbers = sorted(set(numbers))
    if numbers != sorted_unique_numbers:
        logger.log.info(
            f"WARNING: {warning_string} sorted and made unique: {sorted_unique_numbers}"
        )
    return sorted_unique_numbers


class Index(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        sorted_unique_numbers = sorted_unique_nonnegative_numbers(values, "--index")
        setattr(namespace, self.dest, sorted_unique_numbers)


class Classes(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        sorted_unique_numbers = sorted_unique_nonnegative_numbers(values, "--classes")
        setattr(namespace, self.dest, sorted_unique_numbers)


class Command(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, values)
        except Exception as e:
            raise argparse.ArgumentError(self, f"Invalid command: {e}")


class DownloadConfig(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values.lower().endswith(".json") and os.path.isfile(values):
            setattr(namespace, self.dest, values)
        else:
            raise argparse.ArgumentError(
                self,
                f"Please provide a json config file. See the armory-example repo: "
                f"{DEFAULT_SCENARIO}",
            )


# Helper functions for parsers
def _debug(parser):
    parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="synonym for --log-level=armory:debug",
    )
    parser.add_argument(
        "--log-level",
        action="append",
        help="set log level per-module (ex. art:debug) can be used mulitple times",
    )


def _no_gpu(parser):
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Whether to not use GPU(s)",
    )


def _use_gpu(parser):
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Whether to use GPU(s)",
    )


def _gpus(parser):
    parser.add_argument(
        "--gpus",
        type=str,
        help="Which specific GPU(s) to use, such as '3', '1,5', or 'all'",
    )


def _root(parser):
    parser.add_argument(
        "--root",
        action="store_true",
        help="Whether to run docker as root",
    )


def _index(parser):
    parser.add_argument(
        "--index",
        type=str,
        help="Comma-separated nonnegative index for evaluation data point filtering"
        "e.g.: `2` or ``1,3,7`",
        action=Index,
    )


def _classes(parser):
    parser.add_argument(
        "--classes",
        type=str,
        help="Comma-separated nonnegative class ids for filtering"
        "e.g.: `2` or ``1,3,7`",
        action=Classes,
    )


# Config
def _set_gpus(config, use_gpu, no_gpu, gpus):
    """
    Set gpu values from parser in config
    """
    if (use_gpu or gpus) and no_gpu:
        raise ValueError("no_gpu cannot be set with use_gpu or gpus!")

    if gpus:
        if not use_gpu:
            logger.log.info("--gpus field specified. Setting --use-gpu to True")
            use_gpu = True
        config["sysconfig"]["gpus"] = gpus

    if use_gpu or "use_gpu" not in config["sysconfig"]:
        # Override if use_gpu, otherwise if config exists, leave config setting in place
        config["sysconfig"]["use_gpu"] = use_gpu
    elif no_gpu:
        config["sysconfig"]["use_gpu"] = False


def _set_outputs(config, output_dir, output_filename):
    if output_dir:
        config["sysconfig"]["output_dir"] = output_dir
    if output_filename:
        config["sysconfig"]["output_filename"] = output_filename


# Commands
def run(command_args, prog, description) -> int:
    parser = argparse.ArgumentParser(prog=prog, description=description)
    parser.add_argument(
        "filepath",
        metavar="<json_config>",
        type=str,
        help="json config file. Use '-' to accept standard input or pipe.",
    )
    _debug(parser)
    _use_gpu(parser)
    _no_gpu(parser)
    _gpus(parser)
    _root(parser)
    _index(parser)
    _classes(parser)
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override of default output directory prefix",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        help="Override of default output filename prefix",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Whether to quickly check to see if scenario code runs",
    )
    parser.add_argument(
        "--num-eval-batches",
        type=int,
        help="Number of batches to use for evaluation of benign and adversarial examples",
    )
    parser.add_argument(
        "--skip-benign",
        action="store_true",
        help="Skip benign inference and metric calculations",
    )
    parser.add_argument(
        "--skip-attack",
        action="store_true",
        help="Skip attack generation and metric calculations",
    )
    parser.add_argument(
        "--skip-misclassified",
        action="store_true",
        help="Skip attack of inputs that are already misclassified",
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate model configuration against several checks",
    )

    args = parser.parse_args(command_args)
    armory.logs.update_filters(args.log_level, args.debug)

    try:
        if args.filepath == "-":
            if sys.stdin.isatty():
                logger.log.error(
                    "Cannot read config from raw 'stdin'; must pipe or redirect a file"
                )
                return 1
            logger.log.info("Reading config from stdin...")
            config = load_config_stdin()
        else:
            config = load_config(args.filepath)
    except ValidationError as e:
        logger.log.error(
            f"Could not validate config: {e.message} @ {'.'.join(e.absolute_path)}"
        )
        return 1
    except json.decoder.JSONDecodeError:
        if args.filepath == "-":
            logger.log.error("'stdin' did not provide a json-parsable input")
        else:
            logger.log.error(f"Could not decode '{args.filepath}' as a json file.")
            if not args.filepath.lower().endswith(".json"):
                logger.log.warning(f"{args.filepath} is not a '*.json' file")
        return 1
    _set_gpus(config, args.use_gpu, args.no_gpu, args.gpus)
    _set_outputs(config, args.output_dir, args.output_filename)
    logger.log.debug(f"unifying sysconfig {config['sysconfig']} and args {args}")
    (config, args) = arguments.merge_config_and_args(config, args)

    if args.num_eval_batches and args.index:
        raise ValueError("Cannot have --num-eval-batches and --index")
    if args.index and config["dataset"].get("index"):
        logger.log.info("Overriding index in config with command line argument")
    if args.index:
        config["dataset"]["index"] = args.index
    if args.classes and config["dataset"].get("class_ids"):
        logger.log.info("Overriding class_ids in config with command line argument")
    if args.classes:
        config["dataset"]["class_ids"] = args.classes

    rig = Evaluator(config, no_docker=True, root=False)
    exit_code = rig.run(
        interactive=False,
        jupyter=False,
        host_port=None,
        check_run=args.check,
        num_eval_batches=args.num_eval_batches,
        skip_benign=args.skip_benign,
        skip_attack=args.skip_attack,
        skip_misclassified=args.skip_misclassified,
        validate_config=args.validate_config,
    )
    return exit_code


def download(command_args, prog, description):
    """
    Script to download all datasets and model weights for offline usage.
    """
    return  # TODO: Remove this line to enable download command -CW
    parser = argparse.ArgumentParser(prog=prog, description=description)
    _debug(parser)
    _docker_image_optional(parser)
    _skip_docker_images(parser)
    parser.add_argument(
        metavar="<download data config file>",
        dest="download_config",
        type=str,
        default="armory/configs/download_data.json",
        action=DownloadConfig,
        help=f"Configuration for download of data. See {DEFAULT_SCENARIO}. Note: file must be under current working directory.",
    )
    parser.add_argument(
        metavar="<scenario>",
        dest="scenario",
        type=str,
        default="all",
        help="scenario for which to download data, 'list' for available scenarios, or blank to download all scenarios",
        nargs="?",
    )
    from armory.data import datasets, model_weights

    paths.set_mode("host")

    args = parser.parse_args(command_args)

    logger.update_filters(args.log_level, args.debug)
    logger.log.info("Downloading requested datasets and model weights in host mode...")

    datasets.download_all(args.download_config, args.scenario)
    model_weights.download_all(args.download_config, args.scenario)

    return 1


def _get_path(name, default_path, absolute_required=True):
    answer = None
    while answer is None:
        try:
            answer = input(f'{name} [DEFAULT: "{default_path}"]: ')
        except EOFError:
            answer = ""
        if not answer:
            answer = default_path
        answer = os.path.expanduser(answer)
        if os.path.isabs(answer):
            answer = os.path.abspath(answer)
        elif absolute_required:
            print(f"Invalid answer: '{answer}' Absolute path required for {name}")
            answer = None
        else:
            answer = os.path.relpath(answer)
    return answer


def _get_verify_ssl():
    verify_ssl = None
    while verify_ssl is None:
        answer = input("Verify SSL during downloads? [Y/n] ")
        if answer in ("Y", "y", ""):
            verify_ssl = True
        elif answer in ("N", "n"):
            verify_ssl = False
        else:
            print(f"Invalid selection: {answer}")
        print()
        return verify_ssl


def configure(command_args, prog, description):
    parser = argparse.ArgumentParser(prog=prog, description=description)
    _debug(parser)

    parser.add_argument(
        "--use-defaults", default=False, action="store_true", help="Use Defaults"
    )

    args = parser.parse_args(command_args)
    logger.update_filters(args.log_level, args.debug)

    default_host_paths = paths.HostDefaultPaths()

    config = None

    if args.use_defaults:
        config = {
            "dataset_dir": default_host_paths.dataset_dir,
            "local_git_dir": default_host_paths.local_git_dir,
            "saved_model_dir": default_host_paths.saved_model_dir,
            "tmp_dir": default_host_paths.tmp_dir,
            "output_dir": default_host_paths.output_dir,
            "verify_ssl": True,
        }
        print("Saving configuration...")
        save_config(config, default_host_paths.armory_dir)
        print("Configure successful")
        print("Configure complete")
        return

    elif os.path.exists(default_host_paths.armory_config):
        response = None
        while response is None:
            prompt = f"Existing configuration found: {default_host_paths.armory_config}"
            print(prompt)

            response = input("Load existing configuration? [Y/n]")
            if response in ("Y", "y", ""):
                print("Loading configuration...")
                config = load_global_config(
                    default_host_paths.armory_config, validate=False
                )
                print("Load successful")
            elif response in ("N", "n"):
                print("Configuration not loaded")
            else:
                print(f"Invalid selection: {response}")
                response = None
            print()

    instructions = "\n".join(
        [
            "Configuring paths for armory usage",
            f'    This configuration will be stored at "{default_host_paths.armory_config}"',
            "",
            "Please enter desired target directory for the following paths.",
            "    If left empty, the default path will be used.",
            "    Absolute paths (which include '~' user paths) are required.",
            "",
        ]
    )
    print(instructions)

    default_dataset_dir = (
        config["dataset_dir"]
        if config is not None and "dataset_dir" in config.keys()
        else default_host_paths.dataset_dir
    )
    default_local_dir = (
        config["local_git_dir"]
        if config is not None and "local_git_dir" in config.keys()
        else default_host_paths.local_git_dir
    )
    default_saved_model_dir = (
        config["saved_model_dir"]
        if config is not None and "saved_model_dir" in config.keys()
        else default_host_paths.saved_model_dir
    )
    default_tmp_dir = (
        config["tmp_dir"]
        if config is not None and "tmp_dir" in config.keys()
        else default_host_paths.tmp_dir
    )
    default_output_dir = (
        config["output_dir"]
        if config is not None and "output_dir" in config.keys()
        else default_host_paths.output_dir
    )

    config = {
        "dataset_dir": _get_path("dataset_dir", default_dataset_dir),
        "local_git_dir": _get_path("local_git_dir", default_local_dir),
        "saved_model_dir": _get_path("saved_model_dir", default_saved_model_dir),
        "tmp_dir": _get_path("tmp_dir", default_tmp_dir),
        "output_dir": _get_path("output_dir", default_output_dir),
        "verify_ssl": _get_verify_ssl(),
    }
    resolved = "\n".join(
        [
            "Resolved paths:",
            f"    dataset_dir:     {config['dataset_dir']}",
            f"    local_git_dir:   {config['local_git_dir']}",
            f"    saved_model_dir: {config['saved_model_dir']}",
            f"    tmp_dir:         {config['tmp_dir']}",
            f"    output_dir:      {config['output_dir']}",
            "Download options:",
            f"    verify_ssl:      {config['verify_ssl']}",
            "",
        ]
    )
    print(resolved)
    save = None
    while save is None:

        if os.path.isfile(default_host_paths.armory_config):
            print("WARNING: this will overwrite existing configuration.")
            print("    Press Ctrl-C to abort.")
        answer = input("Save this configuration? [Y/n] ")
        if answer in ("Y", "y", ""):
            print("Saving configuration...")
            save_config(config, default_host_paths.armory_dir)
            print("Configure successful")
            save = True
        elif answer in ("N", "n"):
            print("Configuration not saved")
            save = False
        else:
            print(f"Invalid selection: {answer}")
        print()
    print("Configure complete")


def usage(program: str, commands: dict):
    lines = [
        f"{program} <command>",
        "",
        "ARMORY Adversarial Robustness Evaluation Test Bed",
        "https://github.com/twosixlabs/armory",
        "",
        "Commands:",
    ]
    for name, (func, description) in commands.items():
        lines.append(f"    {name} - {description}")
    lines.extend(
        [
            "    -v, --version - get current armory version",
            "",
            f"Run '{program} <command> --help' for more information on a command.",
            " ",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    # TODO the run method now returns a status code instead of sys.exit directly
    # the rest of the COMMANDS should conform

    # command, (function, description)
    program = "armory"
    commands = {
        "run": (
            run,
            "run armory from config file"
        ),
        "download": (
            download,
            "download datasets and model weights used for a given evaluation scenario",
        ),
        "configure": (
            configure,
            "set up armory and dataset paths"
        ),
    }

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print(usage(program, commands))
        sys.exit(1)
    elif sys.argv[1] in ("-v", "--version", "version"):
        print(f"{armory.__version__}")
        sys.exit(0)
    elif sys.argv[1] == "--show-docker-version-tag":
        print(to_docker_tag(armory.__version__))
        sys.exit(0)

    parser = argparse.ArgumentParser(prog=program, usage=usage(program, commands))
    parser.add_argument(
        "command",
        metavar="<command>",
        type=str,
        help="armory command",
        action=Command,
    )
    args = parser.parse_args(sys.argv[1:2])

    func, description = commands[args.command]
    prog = f"{program} {args.command}"
    return func(sys.argv[2:], prog, description)


if __name__ == "__main__":
    sys.exit(main())
