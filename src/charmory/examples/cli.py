"""
Based on `track.py`
"""
import os
import sys

from armory.logs import log
from armory.utils.printing import bold, red
from charmory.engine import Engine

# from charmory.examples.mnist_evaluation import mnist_baseline
from charmory.examples.cifar10_evaluation import cifar10_baseline


def main():
    log.info("Armory: Example Programmatic Entrypoint for Scenario Execution")
    # demo_evaluation = mnist_baseline()
    demo_evaluation = cifar10_baseline()

    log.info(bold(f"Starting Demo for {red(demo_evaluation._metadata.name)}"))

    result = Engine(demo_evaluation).run()
    # result["benign"] = id(demo_evaluation)

    # if self.evaluation.attack:
    #     result["attack"] = id(demo_evaluation)
    log.info(("=" * 64))

    log.info(
        __import__("json").dumps(demo_evaluation.asdict(), indent=4, sort_keys=True)
    )
    log.info("-" * 64)

    log.info(result)
    log.info("=" * 64)

    log.info(bold("mnist experiment results tracked"))

    return result


if __name__ == "__main__":
    main()
    sys.exit(os.EX_OK)
