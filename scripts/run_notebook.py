# Copyright (c) Soumyadip Sarkar.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Execute the project walkthrough and save its outputs."""

import argparse

from thermoshift.notebook import execute


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", nargs="?", default="notebooks/quickstart.ipynb")
    parser.add_argument("--cwd", default=".")
    args = parser.parse_args()
    print(f"Executed {execute(args.notebook, args.cwd)} code cells")


if __name__ == "__main__":
    main()
