# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Stream a local release or Hub repository without materializing all rows."""

import argparse

from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dataset", help="Local dataset directory or OWNER/REPO")
    p.add_argument("--config", choices=["logged", "oracle"], default="logged")
    p.add_argument("--split", default="train")
    p.add_argument("--revision", default=None)
    p.add_argument("--limit", type=int, default=3)
    args = p.parse_args()
    if args.limit < 1:
        p.error("--limit must be positive")
    ds = load_dataset(
        args.dataset, name=args.config, split=args.split, revision=args.revision, streaming=True
    )
    for row in ds.take(args.limit):
        print(row)


if __name__ == "__main__":
    main()
