#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Average the parameters of a ComBAT model file (*.model.csv) across bundles.

Each parameter (row) is averaged over all bundles (columns), and the average
is written back for every bundle. The output file keeps the same header and
layout as the input model, so it can be used directly with combat_apply.

Use --exclude to leave some bundles out of the average computation. The
excluded bundles still receive the averaged values in the output.

Example:
    combat_average_model SH-QC.md.clinical.model.csv \\
        SH-QC.md.clinical.avg.model.csv --exclude FX_L FX_R
"""

import argparse
import os

import numpy as np


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("in_model", help="Input model file (*.model.csv).")
    p.add_argument("out_model", help="Output averaged model file.")
    p.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        metavar="BUNDLE",
        help="Bundles excluded from the average computation.\n"
        "They are still assigned the averaged values in the output.",
    )
    p.add_argument(
        "-f", dest="overwrite", action="store_true",
        help="Force overwriting of the output file.",
    )
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if os.path.exists(args.out_model) and not args.overwrite:
        parser.error("{} already exists. Use -f to overwrite.".format(
            args.out_model))

    # Keep the model_params header line as-is
    with open(args.in_model) as f:
        header = f.readline()
    if not header.startswith("#"):
        parser.error("Missing model header line in {}.".format(args.in_model))

    data = np.loadtxt(args.in_model, delimiter=",", dtype=str, comments="#")
    bundle_names = data[0, 1:]
    param_labels = data[1:, 0]
    params = data[1:, 1:].astype(float)

    unknown = sorted(set(args.exclude) - set(bundle_names))
    if unknown:
        parser.error("Unknown bundle(s) in --exclude: {}".format(
            ", ".join(unknown)))

    keep = ~np.isin(bundle_names, args.exclude)
    if not keep.any():
        parser.error("All bundles are excluded; nothing to average.")

    mean_params = params[:, keep].mean(axis=1)
    avg_params = np.repeat(mean_params[:, None], len(bundle_names), axis=1)

    out = np.vstack([
        np.hstack(["bundle_names", bundle_names]),
        np.hstack([param_labels.reshape(-1, 1), avg_params.astype(str)]),
    ])

    with open(args.out_model, "w") as f:
        f.write(header)
        np.savetxt(f, out, delimiter=",", fmt="%s")

    print("Averaged {} parameters over {}/{} bundles.".format(
        len(param_labels), keep.sum(), len(bundle_names)))
    if args.exclude:
        print("Excluded: {}".format(", ".join(args.exclude)))


if __name__ == "__main__":
    main()
