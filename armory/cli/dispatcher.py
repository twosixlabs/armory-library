


import argparse
import sys



# if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
#     # print(usage())
#     # sys.exit(1)
#     print('version')
# elif sys.argv[1] in ("-v", "--version", "version"):
#     print(f"version")
#     # sys.exit(0)

# parser = argparse.ArgumentParser(prog="armory") #, usage=usage())
# parser.add_argument(
#     "command",
#     metavar="<command>",
#     type=str,
#     help="armory command",
#     # action=Command,
# )
# args = parser.parse_args(sys.argv[1:2])


from pathlib import Path



class Dispatcher:
  def __init__(self):
    self.commands = Path(__file__).parent / "commands"


print("dispatcher")


def command(a):
    print(a.pop(1))
    print(a)
    commands = Path(__file__).parent / "commands"

    print([f.stem for f in commands.iterdir() if f.is_file()])
    print(commands)




"""
{
    "command": {
        "help": "armory command",
        "action": "store",
        "choices": [
            "help",
            "version"
        ],
        "default": "help"
    },
    "args": {
        "help": "armory command arguments",
        "action": "store",
        "nargs": "*"
}
"""


'''
docker run --rm -it --entrypoint "bash" twosixarmory/pytorch:0.15.1

docker run --rm --interactive \
    --gpus all \
    -v ~/.armory:/root/.armory \
    -v ~/.armory:/workspace/.armory \
    twosixarmory/pytorch:0.15.1 \
    /bin/bash -s <<EOF
python -m armory.scenarios.main '`cat scenario_configs/cifar10_baseline.json | base64`' --check --base64
EOF
'''
