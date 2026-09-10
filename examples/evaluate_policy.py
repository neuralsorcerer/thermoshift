# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run the one-step policy example."""

import argparse
import json

from thermoshift.evaluation import evaluate_policy


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--report")
    args = p.parse_args()
    result = evaluate_policy(args.data, args.split)
    if args.report:
        from thermoshift.filesystem import atomic_json

        atomic_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
