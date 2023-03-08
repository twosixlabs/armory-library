# import os
import sys
import argparse
import importlib

from abc import ABC, abstractmethod
from inspect import getsourcefile, getmembers, isclass, isfunction
from pathlib import Path

try:
  # https://github.com/kislyuk/argcomplete
  import argcomplete
  HAS_ARGCOMPLETE = True
except ImportError:
  HAS_ARGCOMPLETE = False


try:
    import armory
except Exception as e:
    raise Exception(f"Error importing Armory module from {__file__}!")

# import armory.logs


global __modpath__
__modpath__ = {getsourcefile(lambda:0)}



# if sys.version_info < (3, 7):
#     raise SystemExit(
#         'ERROR: Armory requires Python 3.7 or newer. '
#         'Current version: %s' % ''.join(sys.version.splitlines())
#     )


class CLI(ABC):
  '''CLI Module Base Class

  The entry point for CLI modules is the class method `.init()`.

  Example:
    >>>  class CommandCLI(CLI):
    >>>    def setup():
    >>>      ...
    >>>    def run():
    >>>      ...
    >>>   def usage():
    >>>      ...
    >>>   CommandCLI.init()
  '''
  name        = None
  description = None
  version     = armory.__version__

  def __init__(self, args, callback=None):
    args = args or sys.argv
    self.args = args = args[1:] if len(args) else []
    self.cmd_path = Path.cwd()

    self.parser = None
    self.callback = callback

    self.exit_code = 1

    # try:
    print("CLI:__init__ => start") ## DEBUG
    # TODO: check function signatures
    setup = self.setup()
    if setup:
      self.exit_code = self.run()
    # except KeyboardInterrupt:
    #   # log.warn("Execution interrupted(KeyboardInterrupt)")
    #   exit_code = 1
    # except Exception as e:
    #   # log.error(e)
    #   # TODO: Show stacktrace(in debug mode), start `pdb`, and enter post mortem.
    #   exit_code = 1
    print("CLI:__init__ => exit") ## DEBUG
    sys.exit(self.exit_code)


  @classmethod
  def init(cls, args=None, exit_code=0):
    # TODO: loader; check that issubclass(cls, CLI)
    # TODO: Manager
    # self.plugins = {}
    # self.plugins['commands'] = {}
    args = args or sys.argv
    print(args)
    cli = cls(args)
    # issubclass(cls, CLI) or sys.exit(1)
    return cli


  # TODO: Should this be abstract?
  @abstractmethod
  def setup():
    raise NotImplementedError("Method not implemented!")


  @abstractmethod
  def run():
    raise NotImplementedError("Method not implemented!")


  @abstractmethod
  def usage():
    raise NotImplementedError("Method not implemented!")


  def config(self, positional=None, flags=None, func=None):
    print('config') ## DEBUG
    parser = self.parser = argparse.ArgumentParser(prog=self.name, description=self.description)
    if func:
      parser.set_defaults(func=init)
    if positional:
      for position in positional:
        subparsers = parser.add_subparsers(**position)
    if flags:
      for arg, kwargs in flags:
        parser.add_argument(*arg, **kwargs)
    if HAS_ARGCOMPLETE:
      argcomplete.autocomplete(self.parser)

    # TODO: Does not fail gracefully if given junk args
    print(self.args)

    # TODO: Move somwhere...
    # Help/Version check
    help_flags = ('help', '--help', '-h', '/h', '?', '/?', )
    # TODO: exit to return
    if len(self.args) == 0:
      print(f"{self.usage()}")
      sys.exit(1)
    if self.args[0] in help_flags:
      print(f"{self.usage()}")
      sys.exit(0)
    if self.args[0] in ('-v', '--version'):
      print(f"{self.version}")
      sys.exit(0)


    # TODO: Add usage prefix: `armory <command>`
    # TODO
    # args = argparse.Namespace()
    # args.func = func
    # print(getattr(args, 'func', None))
    # self.args = args
    # return self.args
    argvs, leftovers = self.parser.parse_known_args(self.args)
    args = self.args

    self.args = argvs
    # # self.args = self.parser.parse_args(self.args)



  def module_loader(self, module_name: str, filepath: Path):
    path   = str(filepath.absolute())
    loader = importlib._bootstrap_external.SourceFileLoader(filepath.stem, path)
    spec   = importlib.util.spec_from_file_location(filepath.stem, path, loader=loader)
    try:
      module = getattr(importlib._bootstrap._load(spec), module_name, False)
      if module and isclass(module):
        return module
      else:
        raise Exception(f"Error importing {module_name} module from {filepath}!")
    except:
      raise ImportError(path, sys.exc_info())



class PluginManager:
  def __init__(self, path: Path, module_name: str):
    ...


class Plugin:
  @classmethod
  def load(cls, path: Path, module_name: str):
    ...
