"""
Example programmatic entrypoint for scenario execution
"""
import sys

from armory.logs import log
from armory.utils.printing import bold, red
from charmory.blocks import cifar10, mnist  # noqa: F401
from charmory.engine import Engine


def main():
    log.info("Armory: Example Programmatic Entrypoint for Scenario Execution")

    baseline = cifar10.baseline

    log.info(bold(f"Starting Demo for {red(baseline.name)}"))

    result = Engine(baseline).run()
    # result["benign"] = id(baseline)

    # if self.evaluation.attack:
    #     result["attack"] = id(baseline)
    log.info(("=" * 64))

    log.info(__import__("json").dumps(baseline.asdict(), indent=4, sort_keys=True))
    log.info("-" * 64)

    log.info(result)
    log.info("=" * 64)

    log.info(bold("mnist experiment results tracked"))

    return result


if __name__ == "__main__":
    main()
    sys.exit(0)
