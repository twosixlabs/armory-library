"""
Example programmatic entrypoint for scenario execution
"""
import sys

import armory
from armory.logs import log
from armory.utils.printing import bold, red
from charmory.blocks import cifar10, mnist  # noqa: F401
from charmory.engine import Engine


def main():
    # if len(sys.argv) > 0:
    #     print(sys.argv)
    #     if sys.argv[1] == "--version":
    #         print(armory.__version__)
    #         sys.exit(0)

    log.info("Armory: Example Programmatic Entrypoint for Scenario Execution")
    log.info(bold(f"Starting Demo for {red(baseline.name)}"))

    baseline = cifar10.baseline
    result = Engine(baseline).run()

    log.info(("=" * 64))
    log.info(__import__("json").dumps(baseline.asdict(), indent=4, sort_keys=True))
    log.info("-" * 64)
    log.info(result)
    log.info("=" * 64)

    log.info(bold("CIFAR10 Experiment Complete!"))
    return result


if __name__ == "__main__":
    main()
    sys.exit(0)
