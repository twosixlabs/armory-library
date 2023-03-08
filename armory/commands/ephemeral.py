import re
import sys
import argparse

from pathlib import Path

try:
    import tomllib  # Python 3.11
except ImportError:
    import toml     # Python 3.10


from armory.cli import CLI


class EmphemeralCLI(CLI):
    name = "armory"
    description = "ARMORY Adversarial Robustness Evaluation Test Bed"

    def setup(self):
        self.commands = self.locate_commands()
        # if self.args[0] not in self.commands:
        #     print(f"{self.usage()}")
        #     # TODO: return (message, exit_code)
        #     return 1
        return 1


    def run(self):
        command = self.args[0]
        data = self.commands.get(command, False)
        if data:
            module  = self.module_loader(data['module'], data['path'])
            return module.init(args=self.args)
        sys.exit(f"{self.usage()}")


    def usage(self):
        line_index = 4
        lines = [
            f"armory <command>\n",
            f"ARMORY Adversarial Robustness Evaluation Test Bed\n",
            f"https://github.com/twosixlabs/armory\n",
            f"Commands:\n",
            # Insert Command Here(index==4)
            f"    -v, --version - get current armory version\n",
            f"Run 'armory <command> --help' for more information on a command.\n",
        ]
        for command, settings in self.locate_commands().items():
            lines.insert(line_index, f"    {command} - {settings['description']}")
            line_index += 1
        return "\n".join(lines)


    def locate_commands(self):
        skip_names = ("__init__.py", "__main__.py", "ephemeral.py")
        commands   = { f.stem: {
                "path": f,
                "name": f.stem,
                "module": f"{f.stem.capitalize()}CLI",
                "description": self.parse_docstring(f)['command']['description']
            }
            for f in Path(__file__).parent.iterdir() if f.is_file() and f.name not in skip_names
        }
        return commands


    def parse_docstring(self, filepath, strict=False):
        docstring_regex = r'^[\'\"]{3}(?P<docstring>.*?)[\"\']{3}'
        docmatch = re.search(docstring_regex,filepath.read_text(), re.DOTALL)
        if docmatch is not None:
            try:
                return toml.loads(docmatch.group('docstring'))
            except Exception as err:
                if strict:
                    raise err
        return None


def main(args=sys.argv):
    EmphemeralCLI.init(args)


if __name__ == "__main__":
    # # Ensure docker/podman is installed
    # if not shutil.which(container_platform):
    #     sys.exit(f"ERROR:\tCannot find compatible container on the system.\n" \
    #              f"\tAsk your system administrator to install either `docker` or `podman`.")

    main()
