"""Generate Armory evaluation report"""

import argparse

import armory.report.console as console
import armory.report.html as html

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="python -m armory.report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    console.configure_args(
        subparsers.add_parser(
            "console",
            help="Print report to console",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    )
    html.configure_args(
        subparsers.add_parser(
            "html",
            help="Produce HTML output",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    )

    args = parser.parse_args()

    if args.cmd == "console":
        console.generate(**vars(args))
    elif args.cmd == "html":
        html.generate(**vars(args))
    else:
        print("no subcmd")
