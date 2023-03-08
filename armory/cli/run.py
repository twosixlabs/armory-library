'''
[command]
name        = "run"
description = "Run armory from config file"
'''

from armory.cli import CLI


class RunCLI(CLI):
  name = "run"
  description = "Run armory from config file"
  flags = [
      (["-v", "--version"], {
          "action":  "version",
          "help":    "Show the version and exit.",
          "version": "1" #CLI.usage
      }),
      (["-p", "--prefix"], {
          "type": str,
          "help": "Override of default output filename prefix."
      })
      # (["-h", "--help"], {
      #     "help":    "Show help message.",
      #     "action":  "help",
      #     "default": argparse.SUPPRESS,
      #     "dest":    "help"
      # })
  ]

  def setup(self):
    print('run.init')
    self.config(
        flags = self.flags,
        positional=[
            {
                'title': "Scenario Config Path",
                'description': "Path to the scenario config file",
            }
        ],
        # actions=[]
        # func=main
    )


  def usage(self):
    print("run.usage")
    # self.setup()
    # self.parser.print_help(sys.stderr)
    return self.parser.format_help()


  def run(self):
    print('run.run')
    # args = self.parser.parse_known_args(self.args)
    # print(args)
    return 0


'''
{
  "runtime": "nvidia",
  "remove": true,
  "detach": true,
  "mounts": [
    {
      "Target": "/workspace",
      "Source": "/home/chris/work/armory/armory",
      "Type": "bind",
      "ReadOnly": false
    },
    {
      "Target": "/armory/datasets",
      "Source": "/home/chris/.armory/datasets",
      "Type": "bind",
      "ReadOnly": false
    },
    {
      "Target": "/armory/git",
      "Source": "/home/chris/.armory/git",
      "Type": "bind",
      "ReadOnly": false
    },
    {
      "Target": "/armory/outputs",
      "Source": "/home/chris/.armory/outputs",
      "Type": "bind",
      "ReadOnly": false
    },
    {
      "Target": "/armory/saved_models",
      "Source": "/home/chris/.armory/saved_models",
      "Type": "bind",
      "ReadOnly": false
    },
    {
      "Target": "/armory/tmp",
      "Source": "/home/chris/.armory/tmp",
      "Type": "bind",
      "ReadOnly": false
    }
  ],
  "shm_size": "16G",
  "command": "tail -f /dev/null",
  "user": "1000:1000",
  "environment": {
    "ARMORY_GITHUB_TOKEN": "",
    "ARMORY_PRIVATE_S3_ID": "",
    "ARMORY_PRIVATE_S3_KEY": "",
    "ARMORY_INCLUDE_SUBMISSION_BUCKETS": "",
    "NVIDIA_VISIBLE_DEVICES": "all",
    "HOME": "/tmp",
    "TORCH_HOME": "/armory/saved_models/pytorch",
    "ARMORY_VERSION": "0.15.4"
  }
}
'''

'''
python -m armory.scenarios.main eyJfZGVzY3JpcHRpb24iOiAiQmFzZWxpbmUgY2lmYXIxMCBpbWFnZSBjbGFzc2lmaWNhdGlvbiIsICJhZGhvYyI6IG51bGwsICJhdHRhY2siOiB7Imtub3dsZWRnZSI6ICJ3aGl0ZSIsICJrd2FyZ3MiOiB7ImJhdGNoX3NpemUiOiAxLCAiZXBzIjogMC4wMzEsICJlcHNfc3RlcCI6IDAuMDA3LCAibWF4X2l0ZXIiOiAyMCwgIm51bV9yYW5kb21faW5pdCI6IDEsICJyYW5kb21fZXBzIjogZmFsc2UsICJ0YXJnZXRlZCI6IGZhbHNlLCAidmVyYm9zZSI6IGZhbHNlfSwgIm1vZHVsZSI6ICJhcnQuYXR0YWNrcy5ldmFzaW9uIiwgIm5hbWUiOiAiUHJvamVjdGVkR3JhZGllbnREZXNjZW50IiwgInVzZV9sYWJlbCI6IHRydWV9LCAiZGF0YXNldCI6IHsiYmF0Y2hfc2l6ZSI6IDY0LCAiZnJhbWV3b3JrIjogIm51bXB5IiwgIm1vZHVsZSI6ICJhcm1vcnkuZGF0YS5kYXRhc2V0cyIsICJuYW1lIjogImNpZmFyMTAifSwgImRlZmVuc2UiOiBudWxsLCAibWV0cmljIjogeyJtZWFucyI6IHRydWUsICJwZXJ0dXJiYXRpb24iOiAibGluZiIsICJyZWNvcmRfbWV0cmljX3Blcl9zYW1wbGUiOiBmYWxzZSwgInRhc2siOiBbImNhdGVnb3JpY2FsX2FjY3VyYWN5Il19LCAibW9kZWwiOiB7ImZpdCI6IHRydWUsICJmaXRfa3dhcmdzIjogeyJuYl9lcG9jaHMiOiAyMH0sICJtb2RlbF9rd2FyZ3MiOiB7fSwgIm1vZHVsZSI6ICJhcm1vcnkuYmFzZWxpbmVfbW9kZWxzLnB5dG9yY2guY2lmYXIiLCAibmFtZSI6ICJnZXRfYXJ0X21vZGVsIiwgIndlaWdodHNfZmlsZSI6IG51bGwsICJ3cmFwcGVyX2t3YXJncyI6IHt9fSwgInNjZW5hcmlvIjogeyJrd2FyZ3MiOiB7fSwgIm1vZHVsZSI6ICJhcm1vcnkuc2NlbmFyaW9zLmltYWdlX2NsYXNzaWZpY2F0aW9uIiwgIm5hbWUiOiAiSW1hZ2VDbGFzc2lmaWNhdGlvblRhc2sifSwgInN5c2NvbmZpZyI6IHsiZG9ja2VyX2ltYWdlIjogInR3b3NpeGFybW9yeS9weXRvcmNoIiwgImV4dGVybmFsX2dpdGh1Yl9yZXBvIjogbnVsbCwgImdwdXMiOiAiYWxsIiwgIm91dHB1dF9kaXIiOiBudWxsLCAib3V0cHV0X2ZpbGVuYW1lIjogbnVsbCwgInVzZV9ncHUiOiB0cnVlLCAiZmlsZXBhdGgiOiAic2NlbmFyaW9fY29uZmlncy9jaWZhcjEwX2Jhc2VsaW5lLmpzb24iLCAiY2hlY2siOiB0cnVlfSwgImV2YWxfaWQiOiAiMjAyMi0xMC0wM1QxNDI4MjYuNjc0NjkyIn0= --check --base64
'''

###############################
# TODO:
# def run(command_args, prog, description) -> int:
#     parser.add_argument(
#         "filepath",
#         metavar="<json_config>",
#         type=str,
#         help="json config file. Use '-' to accept standard input or pipe.",
#     )
#     _debug(parser)
#     _interactive(parser)
#     _jupyter(parser)
#     _port(parser)
#     _use_gpu(parser)
#     _no_gpu(parser)
#     _gpus(parser)
#     _no_docker(parser)
#     _root(parser)
#     _index(parser)
#     _classes(parser)
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         help="Override of default output directory prefix",
#     )
#     parser.add_argument(
#         "--output-filename",
#         type=str,
#         help="Override of default output filename prefix",
#     )
#     parser.add_argument(
#         "--check",
#         action="store_true",
#         help="Whether to quickly check to see if scenario code runs",
#     )
#     parser.add_argument(
#         "--num-eval-batches",
#         type=int,
#         help="Number of batches to use for evaluation of benign and adversarial examples",
#     )
#     parser.add_argument(
#         "--skip-benign",
#         action="store_true",
#         help="Skip benign inference and metric calculations",
#     )
#     parser.add_argument(
#         "--skip-attack",
#         action="store_true",
#         help="Skip attack generation and metric calculations",
#     )
#     parser.add_argument(
#         "--skip-misclassified",
#         action="store_true",
#         help="Skip attack of inputs that are already misclassified",
#     )
#     parser.add_argument(
#         "--validate-config",
#         action="store_true",
#         help="Validate model configuration against several checks",
#     )

#     args = parser.parse_args(command_args)
#     armory.logs.update_filters(args.log_level, args.debug)

#     try:
#         if args.filepath == "-":
#             if sys.stdin.isatty():
#                 log.error(
#                     "Cannot read config from raw 'stdin'; must pipe or redirect a file"
#                 )
#                 return 1
#             log.info("Reading config from stdin...")
#             config = load_config_stdin()
#         else:
#             # TODO: do not assume we know where the command is being called from...
#             # todo = Path(os.getcwd()).parent
#             # filepath = Path(f"{todo}/{args.filepath}")
#             filepath = args.filepath
#             config = load_config(filepath)
#     except ValidationError as e:
#         log.error(
#             f"Could not validate config: {e.message} @ {'.'.join(e.absolute_path)}"
#         )
#         return 1
#     except json.decoder.JSONDecodeError:
#         if args.filepath == "-":
#             log.error("'stdin' did not provide a json-parsable input")
#         else:
#             log.error(f"Could not decode '{args.filepath}' as a json file.")
#             if not args.filepath.lower().endswith(".json"):
#                 log.warning(f"{args.filepath} is not a '*.json' file")
#         return 1
#     _set_gpus(config, args.use_gpu, args.no_gpu, args.gpus)
#     _set_outputs(config, args.output_dir, args.output_filename)
#     log.debug(f"unifying sysconfig {config['sysconfig']} and args {args}")
#     (config, args) = arguments.merge_config_and_args(config, args)

#     if args.num_eval_batches and args.index:
#         raise ValueError("Cannot have --num-eval-batches and --index")
#     if args.index and config["dataset"].get("index"):
#         log.info("Overriding index in config with command line argument")
#     if args.index:
#         config["dataset"]["index"] = args.index
#     if args.classes and config["dataset"].get("class_ids"):
#         log.info("Overriding class_ids in config with command line argument")
#     if args.classes:
#         config["dataset"]["class_ids"] = args.classes

#     rig = Evaluator(config, no_docker=args.no_docker, root=args.root)
#     exit_code = rig.run(
#         interactive=args.interactive,
#         jupyter=args.jupyter,
#         host_port=args.port,
#         check_run=args.check,
#         num_eval_batches=args.num_eval_batches,
#         skip_benign=args.skip_benign,
#         skip_attack=args.skip_attack,
#         skip_misclassified=args.skip_misclassified,
#         validate_config=args.validate_config,
#     )
#     return exit_code


# TODO: run & launch apperently rely on the same code
# def launch(command_args, prog, description):
#     parser = argparse.ArgumentParser(prog=prog, description=description)
#     _docker_image(parser)
#     _debug(parser)
#     _interactive(parser)
#     _jupyter(parser)
#     _port(parser)
#     _use_gpu(parser)
#     _no_gpu(parser)
#     _gpus(parser)
#     _root(parser)

#     args = parser.parse_args(command_args)
#     armory.logs.update_filters(args.log_level, args.debug)

#     config = {"sysconfig": {"docker_image": args.docker_image}}
#     _set_gpus(config, args.use_gpu, args.no_gpu, args.gpus)
#     (config, args) = arguments.merge_config_and_args(config, args)

#     rig = Evaluator(config, root=args.root)
#     exit_code = rig.run(
#         interactive=args.interactive,
#         jupyter=args.jupyter,
#         host_port=args.port,
#         command="true # No-op",
#     )
#     sys.exit(exit_code)



# positional arguments:
#   <json_config>         json config file. Use '-' to accept standard input or pipe.

# options:
#   -h, --help            #  show this help message and exit
#   -d, --debug           #  synonym for --log-level=armory:debug
#   --log-level LOG_LEVEL #  set log level per-module (ex. art:debug) can be used mulitple times
#   -i, --interactive     #  Whether to allow interactive access to container
#   -j, --jupyter         #  Whether to set up Jupyter notebook from container
#   -p , --port           #  Port number {0, ..., 65535} to expose from docker container. If
#                         #  --jupyter flag is set then this port will be used for the jupyter
#                         #  server.
#   --use-gpu             #  Whether to use GPU(s)
#   --no-gpu              #  Whether to not use GPU(s)
#   --gpus GPUS           #  Which specific GPU(s) to use, such as '3', '1,5', or 'all'
#   --no-docker           #  Whether to use Docker or the local host environment
#   --root                #  Whether to run docker as root
#   --index INDEX         #  Comma-separated nonnegative index for evaluation data point
#                         #  filteringe.g.: `2` or ``1,3,7`
#   --classes CLASSES     #  Comma-separated nonnegative class ids for filteringe.g.: `2` or
#                         #  ``1,3,7`
#   --output-dir OUTPUT_DIR #  Override of default output directory prefix
#   --output-filename OUTPUT_FILENAME #  Override of default output filename prefix
#   --check               #  Whether to quickly check to see if scenario code runs
#   --num-eval-batches NUM_EVAL_BATCHES #  Number of batches to use for evaluation of benign and adversarial examples
#   --skip-benign         #  Skip benign inference and metric calculations
#   --skip-attack         #  Skip attack generation and metric calculations
#   --skip-misclassified  #  Skip attack of inputs that are already misclassified
#   --validate-config     #  Validate model configuration against several checks



