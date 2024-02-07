"""Generate Armory evaluation report"""

import argparse

import armory.report.html as html

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        prog="python -m armory.report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    html.configure_args(
        subparsers.add_parser(
            "html",
            help="Produce HTML output",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    )

    args = parser.parse_args()

    if args.cmd == "html":
        html.generate(**vars(args))
    else:
        print("no subcmd")
