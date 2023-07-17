"""
This file is meant to serve as a tool to automate the inspection of the many json configs that exist both on the Github repository as well as in the docs/scenario_configs directory. This file allows the user to specify whether or not they want to look only at the kwargs parts of the config files and then with this specification in mind compiles all of the keys from the json files, finds the most common keys, finds the keys with the most and fewest values, and finds which keys' values have type discrepancies across different json files.

Example Usage:
  $ python tools/json_comprehension.py --directory ./docs/scenario_configs
"""

import argparse
import json
from pathlib import Path
from pprint import pprint
import sys
from typing import Any, Dict, List, Optional, Union


def merge_dictionaries(
    dictionary1: Dict[str, Any], dictionary2: Dict[str, Any]
) -> Dict[str, Any]:
    for key in dictionary2:
        if (
            key in dictionary1
            and isinstance(dictionary1[key], dict)
            and isinstance(dictionary2[key], dict)
        ):
            dictionary1[key] = merge_dictionaries(dictionary1[key], dictionary2[key])
        else:
            dictionary1[key] = dictionary2[key]
    return dictionary1


def extract_keys_from_object(
    json_object: Union[Dict[str, Any], List[Any]]
) -> Optional[Dict[str, Any]]:
    if isinstance(json_object, dict):
        keys_dictionary = {}
        for key, value in json_object.items():
            keys_dictionary[key] = extract_keys_from_object(value)
        return keys_dictionary


def process_json_files(json_files: List[Path]) -> Dict[str, Any]:
    keys_dictionary = {}
    for json_file in json_files:
        with open(json_file, "r") as file:
            data = json.load(file)
            file_keys = extract_keys_from_object(data)
            keys_dictionary = merge_dictionaries(keys_dictionary, file_keys)
    return keys_dictionary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process json files to get unique keys."
    )
    parser.add_argument(
        "--directory",
        type=str,
        required=True,
        help="Directory containing the json files",
    )
    parser.add_argument(
        "--output",
        type=argparse.FileType("w"),
        default=sys.stdout,
        required=False,
        help="Output file to write the result. If not provided, output will be printed to stdout",
    )

    args = parser.parse_args()

    json_files = list(Path(args.directory).rglob("*.json"))
    unique_keys = process_json_files(json_files)

    pprint(unique_keys, stream=args.output)


if __name__ == "__main__":
    main()
