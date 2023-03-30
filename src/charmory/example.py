"""
Example programmatic entrypoint for scenario execution
"""
import json
import sys

from armory.logs import log
from armory.utils.printing import bold, red
import armory.version
from charmory.blocks import cifar10, mnist  # noqa: F401
from charmory.engine import Engine


def main(argv: list = sys.argv[1:]):
    if len(argv) > 0:
        if "--version" in argv:
            print(armory.version.__version__)
            sys.exit(0)

    log.info("Armory: Example Programmatic Entrypoint for Scenario Execution")

    baseline = cifar10.baseline

    log.info(bold(f"Starting Demo for {red(baseline.name)}"))

    result = Engine(baseline).run()

    log.info(("=" * 64))
    log.info(json.dumps(baseline.asdict(), indent=4, sort_keys=True))
    log.info("-" * 64)
    log.info(result)
    log.info("=" * 64)

    log.info(bold("CIFAR10 Experiment Complete!"))
    return result


if __name__ == "__main__":
    main()
    sys.exit(0)
