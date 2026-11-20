# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Run the one-step policy example."""

import argparse
import json

from thermoshift.evaluation import evaluate_policy
from thermoshift.filesystem import atomic_json
from thermoshift.provenance import experiment_report_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--report")
    args = p.parse_args()
    try:
        report = experiment_report_path(args.report, args.data) if args.report else None
    except ValueError as error:
        p.error(str(error))
    result = evaluate_policy(args.data, args.split)
    if report is not None:
        atomic_json(report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
