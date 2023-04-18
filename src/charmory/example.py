"""
Example programmatic entrypoint for scenario execution
"""
import json
import sys

import armory.version
from charmory.blocks import cifar10, mnist  # noqa: F401
from charmory.engine import Engine


def main(argv: list = sys.argv[1:]):
    if len(argv) > 0:
        if "--version" in argv:
            print(armory.version.__version__)
            sys.exit(0)

    print("Armory: Example Programmatic Entrypoint for Scenario Execution")

    baseline = cifar10.baseline

    print(f"Starting Demo for {baseline.name}")

    result = Engine(baseline).run()

    print("=" * 64)
    print(json.dumps(baseline.asdict(), indent=4, sort_keys=True))
    print("-" * 64)
    print(json.dumps(result, indent=4, sort_keys=True))
    print("=" * 64)

    print("CIFAR10 Experiment Complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
