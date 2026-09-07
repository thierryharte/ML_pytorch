#!/usr/bin/env python
"""Compute the alpha factor (N_ttbar / N_data) per region from PocketCoffea
outputs, for the multiclass QCD-morphing weight in convert_to_onnx.py.

alpha_R is needed in the morphing weight
    w = relu(r_0 - alpha * r_2) / relu(r_1 - alpha_den * r_3)
because the classifier is trained with all classes balanced to equal total
weight, so each ttbar output node is normalised as if ttbar were 100% of the
data. alpha_R scales it back to its physical size:
    alpha_R = (ttbar MC yield in R) / (data yield in R).
Pass the numerator-region row (boosted_control_region_C) to convert_to_onnx.py
as --alpha and the denominator-region row (boosted_control_sideband_region_D)
as --alpha-den.

Usage:
    python compute_alpha.py <coffea_output_dir>
"""
import glob
import math
import os
import sys

from coffea.util import load

REGIONS = [
    "boosted_signal_region_A",
    "boosted_signal_sideband_region_B",
    "boosted_control_region_C",
    "boosted_control_sideband_region_D",
]


def walk(node):
    """Sum the innermost non-dict 'nominal' leaves of a region sub-tree."""
    if isinstance(node, dict):
        if "nominal" in node and not isinstance(node["nominal"], dict):
            return node["nominal"]
        return sum(walk(v) for v in node.values())
    return 0.0


def main(in_dir):
    data_files = sorted(glob.glob(os.path.join(in_dir, "output_DATA_*.coffea")))
    ttbar_files = sorted(glob.glob(os.path.join(in_dir, "output_TT*.coffea")))
    if not data_files or not ttbar_files:
        sys.exit(f"no DATA/TT coffea files found in {in_dir}")

    # data: event weight == 1, and the sumw accumulator is not filled for data,
    # so the raw cutflow count is the yield.
    data_n = {r: 0.0 for r in REGIONS}
    for f in data_files:
        o = load(f)
        for r in REGIONS:
            data_n[r] += walk(o["cutflow"][r])

    # ttbar: use the lumi-normalised sumw (and sumw2 for the MC-stat error).
    tt_sumw = {r: 0.0 for r in REGIONS}
    tt_sumw2 = {r: 0.0 for r in REGIONS}
    for f in ttbar_files:
        o = load(f)
        for r in REGIONS:
            tt_sumw[r] += walk(o["sumw"][r])
            tt_sumw2[r] += walk(o["sumw2"][r])

    print(f"{'region':38s} {'N_data':>10s} {'N_ttbar':>12s} {'alpha':>9s} {'MCstat':>9s}")
    for r in REGIONS:
        a = tt_sumw[r] / data_n[r]
        da = math.sqrt(tt_sumw2[r]) / data_n[r]
        print(f"{r:38s} {data_n[r]:10.0f} {tt_sumw[r]:12.1f} {a:9.4f} {da:9.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
