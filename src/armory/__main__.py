import sys
import pathlib
import argparse

try:
    from importlib.metadata import distribution
except ModuleNotFoundError:
    # Python <= 3.7
    from importlib_metadata import distribution  # type: ignore


# Backwards compat for people still calling it from this package
def main(app_name='armory'):
    dist       = distribution(app_name)
    entry_map  = {ep.name: ep for ep in dist.entry_points if ep.group == 'console_scripts'}
    entry_main = entry_map[app_name].load()
    entry_main(sys.argv)


if __name__ == '__main__':
    main()
