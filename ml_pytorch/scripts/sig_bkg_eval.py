import os
import matplotlib

matplotlib.use("Agg")
import argparse
import itertools
import numpy as np
import mplhep as hep
from scipy import stats
from sklearn.metrics import roc_curve, roc_auc_score, auc

from hist import Hist
from utils_configs.plot.HEPPlotter import HEPPlotter

LUMITEXT = "(13.6 TeV)"

# CMS colour palette of mplhep: the first colour is used for the background
# and the second one for the signal, consistently in all the plotting scripts
CMS_COLORS = [cycle["color"] for cycle in hep.style.CMS["axes.prop_cycle"]]
BKG_COLOR = CMS_COLORS[0]
SIG_COLOR = CMS_COLORS[1]


def get_layout(score_lbl_tensor):
    """Return ``(num_score_cols, label_col, weight_col, kl_col)``.

    The score_lbl_array always stores ``[<scores...>, label, weight, kl]`` so
    the trailing 3 columns are fixed and the number of leading score columns
    depends on whether the model is binary (1 column) or multi-class
    (``num_classes`` columns).
    """
    n_score = score_lbl_tensor.shape[1] - 3
    return n_score, n_score, n_score + 1, n_score + 2


def handle_arrays(score_lbl_tensor, column=0, sig_label=1, bkg_label=0):
    """Backward-compatible binary helper: split into the (signal, background)
    events using the label column derived from the array's layout."""
    _, label_col, _, _ = get_layout(score_lbl_tensor)
    sig = score_lbl_tensor[score_lbl_tensor[:, label_col] == sig_label]
    bkg = score_lbl_tensor[score_lbl_tensor[:, label_col] == bkg_label]

    sig_value = sig[:, column]
    bkg_value = bkg[:, column]

    return sig_value, bkg_value


def _class_label(class_info, class_idx, fallback=None):
    """Lookup a human-readable name for ``class_idx`` from the class metadata
    saved at training time. Falls back to a generic name when missing."""
    if class_info is not None:
        for c in class_info:
            ci = c["class_idx"] if isinstance(c, dict) else c.class_idx
            if int(ci) == int(class_idx):
                name = c["name"] if isinstance(c, dict) else c.name
                lbl = c["lbl"] if isinstance(c, dict) else c.lbl
                return f"{name} (lbl={lbl})"
    return fallback if fallback is not None else f"class {class_idx}"


def my_roc_auc(
    classes: np.ndarray, predictions: np.ndarray, sample_weight: np.ndarray = None
) -> float:
    """
    Calculating ROC AUC score as the probability of correct ordering
    """
    # based on https://github.com/SiLiKhon/my_roc_auc/blob/master/my_roc_auc.py

    if sample_weight is None:
        sample_weight = np.ones_like(predictions)

    assert len(classes) == len(predictions) == len(sample_weight)
    assert classes.ndim == predictions.ndim == sample_weight.ndim == 1
    class0, class1 = sorted(np.unique(classes))

    data = np.empty(
        shape=len(classes),
        dtype=[
            ("c", classes.dtype),
            ("p", predictions.dtype),
            ("w", sample_weight.dtype),
        ],
    )
    data["c"], data["p"], data["w"] = classes, predictions, sample_weight

    data = data[np.argsort(data["c"])]
    data = data[
        np.argsort(data["p"], kind="mergesort")
    ]  # here we're relying on stability as we need class orders preserved

    correction = 0.0
    # mask1 - bool mask to highlight collision areas
    # mask2 - bool mask with collision areas' start points
    mask1 = np.empty(len(data), dtype=bool)
    mask2 = np.empty(len(data), dtype=bool)
    mask1[0] = mask2[-1] = False
    mask1[1:] = data["p"][1:] == data["p"][:-1]
    if mask1.any():
        mask2[:-1] = ~mask1[:-1] & mask1[1:]
        mask1[:-1] |= mask1[1:]
        (ids,) = mask2.nonzero()
        correction = (
            sum(
                [
                    ((dsplit["c"] == class0) * dsplit["w"] * msplit).sum()
                    * ((dsplit["c"] == class1) * dsplit["w"] * msplit).sum()
                    for dsplit, msplit in zip(np.split(data, ids), np.split(mask1, ids))
                ]
            )
            * 0.5
        )

    weights_0 = data["w"] * (data["c"] == class0)
    weights_1 = data["w"] * (data["c"] == class1)
    cumsum_0 = weights_0.cumsum()

    return ((cumsum_0 * weights_1).sum() - correction) / (
        weights_1.sum() * cumsum_0[-1]
    )


def weighted_quantile(values, quantile, weights):
    """Weighted quantile: returns the value below which `quantile` fraction of total weight lies."""
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    cumulative_weights = np.cumsum(weights[sorted_idx])
    total_weight = cumulative_weights[-1]
    return float(np.interp(quantile * total_weight, cumulative_weights, sorted_values))


def find_threshold_and_bkg_rejection(
    signal_eff,
    sig_score_test,
    bkg_score_test,
    sig_weight_test,
    bkg_weight_test,
):
    """Find DNN score threshold for target signal efficiency using weighted quantile
    and compute background rejection as weighted fraction below threshold."""
    # (1 - signal_eff) quantile → signal_eff fraction of signal is above threshold
    threshold = weighted_quantile(sig_score_test, 1.0 - signal_eff, sig_weight_test)
    total_bkg_weight = np.sum(bkg_weight_test)
    bkg_rejection = (
        np.sum(bkg_weight_test[bkg_score_test < threshold]) / total_bkg_weight
        if total_bkg_weight > 0
        else 0.0
    )
    return threshold, bkg_rejection


def chi_square(hist_test, hist_train):
    """Chi2/ndof and p-value between the test and the training distribution.

    Both histograms are normalized to unit integral, as they are drawn, and the
    empty bins are skipped. The uncertainties of both are propagated.
    """
    integral_test = hist_test.values().sum()
    integral_train = hist_train.values().sum()
    if integral_test == 0 or integral_train == 0:
        return np.nan, np.nan

    h_test = hist_test.values() / integral_test
    h_train = hist_train.values() / integral_train
    err_test = np.sqrt(hist_test.variances()) / integral_test
    err_train = np.sqrt(hist_train.variances()) / integral_train

    # remove empty bins
    mask = (h_test != 0) & (h_train != 0)

    chi_squared = np.sum(
        (
            (h_test[mask] - h_train[mask])
            / np.sqrt(err_test[mask] ** 2 + err_train[mask] ** 2)
        )
        ** 2
    )
    ndof = len(h_test) - 1

    return chi_squared / ndof, 1 - stats.chi2.cdf(chi_squared, ndof)


def compute_significance(
    dnn_score_target,
    counts_test_list,
    bin_centers,
    bin_width,
    sig_weight_test,
    bkg_weight_test,
    test_fraction,
    rescale,
):
    """Compute significance from binned density histograms above the DNN score threshold.
    Integrates the normalized histograms above the threshold to obtain event fractions,
    then converts to absolute event counts using total weights and rescale factors."""
    bin_index = np.searchsorted(bin_centers, dnn_score_target)
    sig_fraction_above = np.sum(counts_test_list[0][bin_index:] * bin_width[bin_index:])
    bkg_fraction_above = np.sum(counts_test_list[1][bin_index:] * bin_width[bin_index:])
    sig_rescale = rescale[0] if rescale else 1
    bkg_rescale = rescale[1] if rescale else 1
    n_sig_above_target = (
        sig_fraction_above * np.sum(sig_weight_test) / test_fraction * sig_rescale
    )
    n_bkg_above_target = (
        bkg_fraction_above * np.sum(bkg_weight_test) / test_fraction * bkg_rescale
    )
    significance_above_target = np.sqrt(
        2
        * (
            (n_sig_above_target + n_bkg_above_target)
            * np.log(n_sig_above_target / n_bkg_above_target + 1)
            - n_sig_above_target
        )
    )
    return n_sig_above_target, n_bkg_above_target, significance_above_target


def plot_sig_bkg_distributions(
    score_lbl_tensor_train,
    score_lbl_tensor_test,
    dir,
    show,
    rescale,
    test_fraction,
    signal_eff=0.8,
    get_max_significance=False,
    comet_logger=None,
    class_info=None,
    kl_bkg_str=None,
):
    # Dispatch to the multi-class variant when the output has more than one
    # score column (i.e. C >= 3 class probabilities are stored).
    n_score, _, _, _ = get_layout(score_lbl_tensor_test)
    if n_score > 1:
        plot_multiclass_distributions(
            score_lbl_tensor_train,
            score_lbl_tensor_test,
            dir,
            show,
            class_info,
            comet_logger=comet_logger,
        )
        return

    # plot the signal and background distributions
    sig_score_train, bkg_score_train = handle_arrays(score_lbl_tensor_train, 0)
    sig_score_test, bkg_score_test = handle_arrays(score_lbl_tensor_test, 0)

    print("sig_score_train", sig_score_train, sig_score_train.shape)
    print("bkg_score_train", bkg_score_train, bkg_score_train.shape)
    print("sig_score_test", sig_score_test, sig_score_test.shape)
    print("bkg_score_test", bkg_score_test, bkg_score_test.shape)

    # get weights
    _, _, weight_col, kl_col = get_layout(score_lbl_tensor_test)
    try:
        sig_weight_train, bkg_weight_train = handle_arrays(
            score_lbl_tensor_train, weight_col
        )
        sig_weight_test, bkg_weight_test = handle_arrays(
            score_lbl_tensor_test, weight_col
        )
    except IndexError:
        print("WARNING: No weights found in the input file. Using equal weights.")
        sig_weight_train = np.ones_like(sig_score_train)
        bkg_weight_train = np.ones_like(bkg_score_train)
        sig_weight_test = np.ones_like(sig_score_test)
        bkg_weight_test = np.ones_like(bkg_score_test)

    print("sig_weight_train", sig_weight_train, sig_weight_train.shape)
    print("bkg_weight_train", bkg_weight_train, bkg_weight_train.shape)
    print("sig_weight_test", sig_weight_test, sig_weight_test.shape)
    print("bkg_weight_test", bkg_weight_test, bkg_weight_test.shape)

    # get the kl values
    try:
        sig_kl_train, bkg_kl_train = handle_arrays(score_lbl_tensor_train, kl_col)
        sig_kl_test, bkg_kl_test = handle_arrays(score_lbl_tensor_test, kl_col)
    except IndexError:
        print("WARNING: No kl values found in the input file. Using equal weights.")
        sig_kl_train = np.ones_like(sig_score_train) * 9999.0
        bkg_kl_train = np.ones_like(bkg_score_train) * 9999.0
        sig_kl_test = np.ones_like(sig_score_test) * 9999.0
        bkg_kl_test = np.ones_like(bkg_score_test) * 9999.0

    print("sig_kl_train", sig_kl_train, sig_kl_train.shape)
    print("bkg_kl_train", bkg_kl_train, bkg_kl_train.shape)
    print("sig_kl_test", sig_kl_test, sig_kl_test.shape)
    print("bkg_kl_test", bkg_kl_test, bkg_kl_test.shape)

    kl_unique_values = list(np.unique(sig_kl_train))
    print("kl_unique_values", kl_unique_values)

    # loop over the differetn kl for signal and take inclusively for bkg
    for kl in kl_unique_values + ["all"]:
        if kl != "all":
            sig_score_train_kl = sig_score_train[sig_kl_train == kl]
            sig_weight_train_kl = sig_weight_train[sig_kl_train == kl]
            sig_score_test_kl = sig_score_test[sig_kl_test == kl]
            sig_weight_test_kl = sig_weight_test[sig_kl_test == kl]
            kl_str = f"{kl:.2f}"
        else:
            sig_score_train_kl = sig_score_train
            sig_weight_train_kl = sig_weight_train
            sig_score_test_kl = sig_score_test
            sig_weight_test_kl = sig_weight_test
            kl_str = "all"

        # HEPPlotter.set_output strips whatever follows the last dot of the
        # output path, so the dots of the kl value are replaced with a "p"
        kl_tag = kl_str.replace("-", "m").replace(".", "p")

        ks_statistic_sig, p_value_sig = stats.ks_2samp(
            sig_score_train_kl, sig_score_test_kl
        )
        ks_statistic_bkg, p_value_bkg = stats.ks_2samp(bkg_score_train, bkg_score_test)
        print(f"\nKS: statistic (sig) = {ks_statistic_sig:.30f}")
        print(f"KS: p-value (sig) = {p_value_sig:.30f}")
        print(f"KS: statistic (bkg) = {ks_statistic_bkg:.30f}")
        print(f"KS: p-value (bkg) = {p_value_bkg:.30f}")

        # Compute significance

        counts_test_list = []
        for score, weight, rescale_factor in zip(
            [sig_score_test_kl, bkg_score_test],
            [sig_weight_test_kl, bkg_weight_test],
            rescale if rescale else [1, 1],
        ):
            counts, bins = np.histogram(
                score,
                weights=weight * rescale_factor,
                bins=1000,
                density=True,
                range=(0, 1),
            )
            counts_test_list.append(counts)
            bin_width = bins[1:] - bins[:-1]
            bin_centers = 0.5 * (bins[:-1] + bins[1:])

        n_sig = (
            np.sum(sig_weight_test_kl) / test_fraction * (rescale[0] if rescale else 1)
        )
        n_bkg = np.sum(bkg_weight_test) / test_fraction * (rescale[1] if rescale else 1)
        significance = n_sig / np.sqrt(n_bkg)
        print(f"\nNumber of signal events in the test dataset: {n_sig}")
        print(f"Number of background events in the test dataset: {n_bkg}")
        print(f"Significance: {significance:.2f}\n")

        lines = []

        if signal_eff != -1:
            if get_max_significance:
                max_significance = -1
                for sig_eff_target in np.linspace(0.0, 1.0, 30):
                    threshold, bkg_rej = find_threshold_and_bkg_rejection(
                        sig_eff_target,
                        sig_score_test_kl,
                        bkg_score_test,
                        sig_weight_test_kl,
                        bkg_weight_test,
                    )
                    n_sig, n_bkg, significance = compute_significance(
                        threshold,
                        counts_test_list,
                        bin_centers,
                        bin_width,
                        sig_weight_test_kl,
                        bkg_weight_test,
                        test_fraction,
                        rescale,
                    )
                    if significance > max_significance:
                        max_significance = significance
                        print("max_significance", max_significance)
                        dnn_score_target = threshold
                        bkg_rejection = bkg_rej
                        n_sig_above_target = n_sig
                        n_bkg_above_target = n_bkg
                        significance_above_target = significance
                        signal_eff = sig_eff_target
            else:
                dnn_score_target, bkg_rejection = find_threshold_and_bkg_rejection(
                    signal_eff,
                    sig_score_test_kl,
                    bkg_score_test,
                    sig_weight_test_kl,
                    bkg_weight_test,
                )
                n_sig_above_target, n_bkg_above_target, significance_above_target = (
                    compute_significance(
                        dnn_score_target,
                        counts_test_list,
                        bin_centers,
                        bin_width,
                        sig_weight_test_kl,
                        bkg_weight_test,
                        test_fraction,
                        rescale,
                    )
                )

            print(
                f"\n###########\nNumber of signal events above {signal_eff:.3f} signal efficiency threshold: {n_sig_above_target:.3f}"
            )
            print(
                f"Number of background events above {signal_eff:.3f} signal efficiency threshold: {n_bkg_above_target:.3f}"
            )
            print(
                f"Significance ({dnn_score_target:.3f} DNN cut): {significance_above_target:.3f}"
            )
            # plot the vertical line for the signal efficiency
            lines.append(
                {
                    "x": dnn_score_target,
                    "color": "grey",
                    "linestyle": "--",
                    "label": "Sig efficiency = {:.2f}\nBkg rejection = {:.2f}\nDNN score cut = {:.2f}".format(
                        signal_eff,
                        bkg_rejection,
                        dnn_score_target,
                    ),
                }
            )

        # Single overtraining plot with signal and background overlaid.
        # The histograms are filled with weights normalized to unit integral,
        # so that the four distributions can be compared with each other.
        def normalized_hist(scores, weights):
            h = Hist.new.Reg(50, 0, 1, name="score").Weight()
            total = np.sum(weights)
            h.fill(scores, weight=weights / total if total else weights)
            return h

        hist_sig_train = normalized_hist(sig_score_train_kl, sig_weight_train_kl)
        hist_sig_test = normalized_hist(sig_score_test_kl, sig_weight_test_kl)
        hist_bkg_train = normalized_hist(bkg_score_train, bkg_weight_train)
        hist_bkg_test = normalized_hist(bkg_score_test, bkg_weight_test)

        series_dict = {
            "Signal (training)": {
                "data": hist_sig_train,
                "style": {
                    "color": SIG_COLOR,
                    "histtype": "step",
                    # "edgecolor": SIG_COLOR,
                    # "facecolor": SIG_COLOR,
                    # "alpha": 0.5,
                },
            },
            "Signal (test)": {
                "data": hist_sig_test,
                "style": {"histtype": "errorbar", "color": SIG_COLOR},
            },
            "Background (training)": {
                "data": hist_bkg_train,
                "style": {"color": BKG_COLOR, "histtype": "step"},
            },
            "Background (test)": {
                "data": hist_bkg_test,
                "style": {"histtype": "errorbar", "color": BKG_COLOR},
            },
        }

        chi2_sig, pvalue_sig = chi_square(hist_sig_test, hist_sig_train)
        chi2_bkg, pvalue_bkg = chi_square(hist_bkg_test, hist_bkg_train)
        print(f"\nchi2/ndof (sig) = {chi2_sig:.3f}, p-value = {pvalue_sig:.3f}")
        print(f"chi2/ndof (bkg) = {chi2_bkg:.3f}, p-value = {pvalue_bkg:.3f}")

        base = f"{dir}/sig_bkg_distributions_kl_{kl_tag}"

        # the histograms are normalized to unit integral by HEPPlotter, so
        # the same log range can be used for every training
        for log in [False, True]:
            plotter = (
                HEPPlotter("CMS")
                .set_plot_config(figsize=[13, 13], lumitext=LUMITEXT, cmstext="Private")
                .set_output(f"{base}{'_log' if log else ''}")
                .set_labels(
                    xlabel="DNN Class Score",
                    ylabel="Normalized counts",
                )
                .set_data(series_dict, plot_type="1d")
                .set_options(
                    legend_loc="upper left",
                    legend_font_size=20,
                    split_legend=False,
                    grid=True,
                    y_log=log,
                    ylim_bottom_value=1e-4 if log else 0.0,
                    ylim_top_value=5 if log else 0.15,
                    # ylim_top_factor=2,
                )
                .add_annotation(
                    x=0.6,
                    y=0.95,
                    s=f"KS sig: p-value = {p_value_sig:.2f}",
                    fontsize=20,
                    ha="left",
                    va="center",
                    color=SIG_COLOR,
                )
                .add_annotation(
                    x=0.6,
                    y=0.9,
                    s=f"KS bkg: p-value = {p_value_bkg:.2f}",
                    fontsize=20,
                    ha="left",
                    va="center",
                    color=BKG_COLOR,
                )
                .add_annotation(
                    x=0.6,
                    y=0.85,
                    s=r"$\chi^2$/ndof= {:.1f},".format(chi2_sig)
                    + f"  p-value= {pvalue_sig:.2f}",
                    fontsize=20,
                    ha="left",
                    va="center",
                    color=SIG_COLOR,
                )
                .add_annotation(
                    x=0.6,
                    y=0.8,
                    s=r"$\chi^2$/ndof= {:.1f},".format(chi2_bkg)
                    + f"  p-value= {pvalue_bkg:.2f}",
                    fontsize=20,
                    ha="left",
                    va="center",
                    color=BKG_COLOR,
                )
                .add_annotation(
                    x=0.6,
                    y=0.75,
                    s=rf"sig $\kappa_\lambda$ = {kl_str}",
                    fontsize=20,
                    ha="left",
                    va="center",
                    color=SIG_COLOR,
                )
                .add_annotation(
                    x=0.6,
                    y=0.7,
                    s=rf"bkg $\kappa_\lambda$ = {kl_bkg_str if kl_bkg_str else 'all'}",
                    fontsize=20,
                    ha="left",
                    va="center",
                    color=BKG_COLOR,
                )
            )
            for line in lines:
                plotter.add_line("v", **line)
            if show:
                plotter.show()
            plotter.run()

            if comet_logger:
                comet_logger.log_image(
                    f"{base}{'_log' if log else ''}.png",
                    name=f"sig_bkg_distributions{'_log' if log else ''}",
                )


def plot_roc_curve(score_lbl_tensor_test, dir, show, comet_logger=None, class_info=None, kl_bkg_str=None):
    # Dispatch to multi-class ROCs when more than one score column is stored.
    n_score, label_col, weight_col, kl_col = get_layout(score_lbl_tensor_test)
    if n_score > 1:
        plot_multiclass_roc_curves(
            score_lbl_tensor_test, dir, show, class_info, comet_logger=comet_logger
        )
        return

    sig_score_test, bkg_score_test = handle_arrays(score_lbl_tensor_test, 0)
    sig_lbl_test, bkg_lbl_test = handle_arrays(score_lbl_tensor_test, label_col)

    # get the weight
    try:
        sig_weight_test, bkg_weight_test = handle_arrays(
            score_lbl_tensor_test, weight_col
        )
    except IndexError:
        print("WARNING: No weight values found in the input file. Using equal weight.")
        sig_weight_test = np.ones_like(sig_score_test)
        bkg_weight_test = np.ones_like(bkg_score_test)

    print("sig_weight_test", sig_weight_test, sig_weight_test.shape)
    print("bkg_weight_test", bkg_weight_test, bkg_weight_test.shape)

    # get the kl values
    try:
        sig_kl_test, bkg_kl_test = handle_arrays(score_lbl_tensor_test, kl_col)
    except IndexError:
        print("WARNING: No kl values found in the input file. Using equal weights.")
        sig_kl_test = np.ones_like(sig_score_test) * 9999.0
        bkg_kl_test = np.ones_like(bkg_score_test) * 9999.0

    print("sig_kl_test", sig_kl_test, sig_kl_test.shape)
    print("bkg_kl_test", bkg_kl_test, bkg_kl_test.shape)

    kl_unique_values = list(np.unique(sig_kl_test))
    print("kl_unique_values", kl_unique_values)
    roc_info_dict = {}

    # loop over the differetn kl for signal and take inclusively for bkg
    for kl in kl_unique_values + ["all"]:
        if kl != "all":
            sig_score_test_kl = sig_score_test[sig_kl_test == kl]
            sig_weight_test_kl = sig_weight_test[sig_kl_test == kl]
            sig_lbl_test_kl = sig_lbl_test[sig_kl_test == kl]
            kl_str = f"{kl:.2f}"
        else:
            sig_score_test_kl = sig_score_test
            sig_weight_test_kl = sig_weight_test
            sig_lbl_test_kl = sig_lbl_test
            kl_str = "all"

        kl_tag = kl_str.replace("-", "m").replace(".", "p")

        score = np.concatenate((sig_score_test_kl, bkg_score_test))
        weight = np.concatenate((sig_weight_test_kl, bkg_weight_test))
        lbl = np.concatenate((sig_lbl_test_kl, bkg_lbl_test))

        # plot the ROC curve
        fpr, tpr, _ = roc_curve(
            lbl,
            score,
            sample_weight=weight,
        )
        roc_auc = my_roc_auc(
            lbl,
            score,
            sample_weight=weight,
        )

        abs_weights_fpr, abs_weights_tpr, _ = roc_curve(
            lbl,
            score,
            sample_weight=abs(weight),
        )
        abs_weights_roc_auc = roc_auc_score(
            lbl,
            score,
            sample_weight=abs(weight),
        )

        # save tpr and fpr in a npz file

        roc_info_dict[f"tpr_kl_{kl_str}"] = tpr
        roc_info_dict[f"fpr_kl_{kl_str}"] = fpr
        roc_info_dict[f"abs_weights_tpr_kl_{kl_str}"] = abs_weights_tpr
        roc_info_dict[f"abs_weights_fpr_kl_{kl_str}"] = abs_weights_fpr

        # the ROC curves are drawn as graphs without error bars
        series_dict = {
            f"ROC curve - kl = {kl_str} (pos+neg weights AUC = {roc_auc:.3f})": {
                "data": {"x": [tpr, None], "y": [fpr, None]},
                "style": {"linestyle": "-", "markersize": 0},
            },
            f"ROC curve - kl = {kl_str} (abs weights AUC = {abs_weights_roc_auc:.3f})": {
                "data": {"x": [abs_weights_tpr, None], "y": [abs_weights_fpr, None]},
                "style": {"linestyle": "-", "markersize": 0},
            },
        }

        plotter = (
            HEPPlotter("CMS")
            .set_plot_config(lumitext=LUMITEXT)
            .set_output(f"{dir}/roc_curve_kl_{kl_tag}")
            .set_labels(xlabel="True positive rate", ylabel="False positive rate")
            .set_data(series_dict, plot_type="graph")
            .set_options(
                y_log=True,
                legend_loc="upper left",
                legend_font_size="small",
                split_legend=False,
                grid=False,
                set_ylim=False,
            )
            .add_annotation(
                x=0.98,
                y=0.05,
                s=rf"sig $\kappa_\lambda$ = {kl_str}"
                + "\n"
                + rf"bkg $\kappa_\lambda$ = {kl_bkg_str if kl_bkg_str else 'all'}",
                fontsize=16,
                ha="right",
                va="bottom",
            )
        )
        if show:
            plotter.show()
        plotter.run()

        if comet_logger:
            comet_logger.log_image(f"{dir}/roc_curve_kl_{kl_tag}.png", name="roc_curve")

    # save tpr and fpr in a npz file
    np.savez(f"{dir}/tpr_fpr.npz", **roc_info_dict)


def plot_kl_distributions(
    score_lbl_train,
    score_lbl_test,
    out_dir,
    kls_background_to_plot,
    train_test_fraction,
    show=False,
    rescale=None,
    signal_eff=0.8,
    get_max_significance=False,
    do_histos=True,
    do_roc=True,
    comet_logger=None,
):
    if rescale is None:
        rescale = []
    for kl_bkg in kls_background_to_plot:
        kl_bkg_str = "all" if kl_bkg == "all" else f"{kl_bkg:.2f}"

        if kl_bkg == "all":
            train_data = score_lbl_train
            test_data = score_lbl_test
        else:
            try:
                sig_train = score_lbl_train[:, 1] == 1
                bkg_train_kl = (score_lbl_train[:, 1] == 0) & (
                    score_lbl_train[:, 3] == float(kl_bkg)
                )
                train_data = score_lbl_train[sig_train | bkg_train_kl]

                sig_test = score_lbl_test[:, 1] == 1
                bkg_test_kl = (score_lbl_test[:, 1] == 0) & (
                    score_lbl_test[:, 3] == float(kl_bkg)
                )
                test_data = score_lbl_test[sig_test | bkg_test_kl]
            except IndexError:
                train_data = score_lbl_train
                test_data = score_lbl_test

        if do_histos:
            sig_bkg_out_dir = f"{out_dir}/sig_bkg_bkgkl_{kl_bkg_str}"
            os.makedirs(sig_bkg_out_dir, exist_ok=True)
            plot_sig_bkg_distributions(
                train_data,
                test_data,
                sig_bkg_out_dir,
                show,
                rescale,
                train_test_fraction,
                signal_eff=signal_eff,
                get_max_significance=get_max_significance,
                comet_logger=comet_logger,
                kl_bkg_str=kl_bkg_str,
            )

        if do_roc:
            roc_out_dir = f"{out_dir}/roc_bkgkl_{kl_bkg_str}"
            os.makedirs(roc_out_dir, exist_ok=True)
            plot_roc_curve(
                test_data,
                roc_out_dir,
                show,
                comet_logger=comet_logger,
                kl_bkg_str=kl_bkg_str,
            )


def plot_multiclass_distributions(
    score_lbl_tensor_train,
    score_lbl_tensor_test,
    dir,
    show,
    class_info=None,
    comet_logger=None,
):
    """Plot, for each output node, the distribution of its score split by
    true class. Produces ``score_distribution_class_<i>.png`` per output."""
    n_score, label_col, weight_col, _ = get_layout(score_lbl_tensor_test)
    train_labels = score_lbl_tensor_train[:, label_col].astype(int)
    test_labels = score_lbl_tensor_test[:, label_col].astype(int)
    train_weights = score_lbl_tensor_train[:, weight_col]
    test_weights = score_lbl_tensor_test[:, weight_col]

    true_classes = sorted(set(np.unique(train_labels)).union(np.unique(test_labels)))
    print(f"Multi-class evaluation: output nodes = {n_score}, true classes = {true_classes}")

    colors = plt.get_cmap("tab10").colors

    for out_idx in range(n_score):
        fig, ax = plt.subplots(figsize=[13, 9])
        node_name = _class_label(class_info, out_idx, fallback=f"class {out_idx}")

        train_scores = score_lbl_tensor_train[:, out_idx]
        test_scores = score_lbl_tensor_test[:, out_idx]

        for j, true_c in enumerate(true_classes):
            color = colors[j % len(colors)]
            tr_mask = train_labels == true_c
            te_mask = test_labels == true_c
            if tr_mask.sum() == 0 and te_mask.sum() == 0:
                continue

            label = _class_label(class_info, true_c, fallback=f"true class {true_c}")
            ax.hist(
                train_scores[tr_mask],
                weights=train_weights[tr_mask],
                bins=30,
                range=(0, 1),
                histtype="step",
                density=True,
                color=color,
                linestyle="--",
                label=f"{label} (train)",
            )
            ax.hist(
                test_scores[te_mask],
                weights=test_weights[te_mask],
                bins=30,
                range=(0, 1),
                histtype="step",
                density=True,
                color=color,
                linestyle="-",
                label=f"{label} (test)",
            )

        ax.set_xlabel(f"Score of output node: {node_name}")
        ax.set_ylabel("Normalized counts")
        ax.legend(loc="upper center", fontsize=14, frameon=False)
        ax.grid()
        hep.cms.lumitext("2022 (13.6 TeV)", ax=ax)
        hep.cms.text(text="Preliminary", ax=ax, loc=0)
        if comet_logger:
            comet_logger.log_figure(f"score_distribution_class_{out_idx}", plt)
        plt.savefig(
            f"{dir}/score_distribution_class_{out_idx}.png",
            bbox_inches="tight",
            dpi=300,
        )
        plt.savefig(
            f"{dir}/score_distribution_class_{out_idx}.pdf",
            bbox_inches="tight",
            dpi=300,
        )
        ax.set_yscale("log")
        plt.savefig(
            f"{dir}/score_distribution_class_{out_idx}_log.png",
            bbox_inches="tight",
            dpi=300,
        )
        if show:
            plt.show()
        plt.close(fig)


def plot_multiclass_roc_curves(
    score_lbl_tensor_test, dir, show, class_info=None, comet_logger=None
):
    """Plot ROC curves for the multi-class output:

    * One-vs-rest ROC for every output node (signal=node i, background=all
      events whose true class != i, using score of node i).
    * One-vs-one ROC for every ordered pair of classes (positive class A vs
      negative class B, considering only events whose true class is A or B
      and using score of node A).
    """
    n_score, label_col, weight_col, _ = get_layout(score_lbl_tensor_test)
    labels = score_lbl_tensor_test[:, label_col].astype(int)
    weights = score_lbl_tensor_test[:, weight_col]

    true_classes = sorted(np.unique(labels).tolist())
    print(f"Multi-class ROC: output nodes = {n_score}, true classes = {true_classes}")

    roc_info_dict = {}

    # --- One-vs-rest ROCs (overlaid on a single figure) ---
    fig, ax = plt.subplots(figsize=[10, 8])
    for out_idx in range(n_score):
        scores = score_lbl_tensor_test[:, out_idx]
        ovr_labels = (labels == out_idx).astype(int)
        if ovr_labels.sum() == 0 or (1 - ovr_labels).sum() == 0:
            continue

        try:
            fpr, tpr, _ = roc_curve(ovr_labels, scores, sample_weight=abs(weights))
            roc_auc = roc_auc_score(ovr_labels, scores, sample_weight=abs(weights))
        except ValueError as e:
            print(f"Skipping one-vs-rest ROC for class {out_idx}: {e}")
            continue

        name = _class_label(class_info, out_idx, fallback=f"class {out_idx}")
        ax.plot(tpr, fpr, label=f"{name} vs rest (AUC = {roc_auc:.3f})")
        roc_info_dict[f"ovr_tpr_class_{out_idx}"] = tpr
        roc_info_dict[f"ovr_fpr_class_{out_idx}"] = fpr
        roc_info_dict[f"ovr_auc_class_{out_idx}"] = roc_auc

    ax.set_xlabel("True positive rate")
    ax.set_ylabel("False positive rate")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize="small")
    ax.grid()
    hep.cms.lumitext("2022 (13.6 TeV)", ax=ax)
    hep.cms.text(text="Preliminary", ax=ax, loc=0)
    if comet_logger:
        comet_logger.log_figure("roc_curve_one_vs_rest", plt)
    plt.savefig(f"{dir}/roc_curve_one_vs_rest.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{dir}/roc_curve_one_vs_rest.pdf", bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    # --- One-vs-one pairwise ROCs (overlaid) ---
    fig, ax = plt.subplots(figsize=[10, 8])
    for a, b in itertools.combinations(range(n_score), 2):
        mask = (labels == a) | (labels == b)
        if mask.sum() == 0:
            continue
        scores_a = score_lbl_tensor_test[mask, a]
        sub_labels = (labels[mask] == a).astype(int)
        sub_weights = weights[mask]
        if sub_labels.sum() == 0 or (1 - sub_labels).sum() == 0:
            continue

        try:
            fpr, tpr, _ = roc_curve(
                sub_labels, scores_a, sample_weight=abs(sub_weights)
            )
            roc_auc = roc_auc_score(
                sub_labels, scores_a, sample_weight=abs(sub_weights)
            )
        except ValueError as e:
            print(f"Skipping pair {a} vs {b} ROC: {e}")
            continue

        a_name = _class_label(class_info, a, fallback=f"class {a}")
        b_name = _class_label(class_info, b, fallback=f"class {b}")
        ax.plot(
            tpr, fpr, label=f"{a_name} vs {b_name} (AUC = {roc_auc:.3f})"
        )
        roc_info_dict[f"ovo_tpr_{a}_vs_{b}"] = tpr
        roc_info_dict[f"ovo_fpr_{a}_vs_{b}"] = fpr
        roc_info_dict[f"ovo_auc_{a}_vs_{b}"] = roc_auc

    ax.set_xlabel("True positive rate")
    ax.set_ylabel("False positive rate")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize="small")
    ax.grid()
    hep.cms.lumitext("2022 (13.6 TeV)", ax=ax)
    hep.cms.text(text="Preliminary", ax=ax, loc=0)
    if comet_logger:
        comet_logger.log_figure("roc_curve_one_vs_one", plt)
    plt.savefig(f"{dir}/roc_curve_one_vs_one.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"{dir}/roc_curve_one_vs_one.pdf", bbox_inches="tight", dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    np.savez(f"{dir}/tpr_fpr_multiclass.npz", **roc_info_dict)


def main():
    # parse the arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", "--input-dir", default="score_lbls", help="Input directory", type=str
    )
    parser.add_argument(
        "-s", "--show", default=False, help="Show plots", action="store_true"
    )
    parser.add_argument(
        "-r",
        "--rescale",
        nargs="+",
        type=float,
        default=[
            # 0.3363,
            # 0.3937,  # this is the ratio of the (new xsec * BR) over the (old xsec)
        ],  # 2.889e-6 4.567e-5 (=1/sumgenweights*10) #9.71589e-7, 1.79814e-5] #  3.453609602837785e-05,0.00017658439204048897,
        help="Rescale the signal and background when computing the number of expected events",
    )
    parser.add_argument(
        "-e", "--signal-eff", default=-1, help="Signal efficiency to cut", type=float
    )
    parser.add_argument(
        "-klb",
        "--kl-background",
        nargs="+",
        default=["all", "1"],
        help="Background kl values to plot. Use 'all' for the inclusive plot, numbers for specific kl values, or 'full' to plot every available kl (default: all 1).",
    )

    parser.print_help()
    args = parser.parse_args()

    input_file = f"{args.input_dir}/score_lbl_array.npz"

    # load the labels and scores from the train and test datasets from a .npz file
    score_lbl_tensor_train = np.load(input_file, allow_pickle=True)[
        "score_lbl_array_train"
    ]
    score_lbl_tensor_test = np.load(input_file, allow_pickle=True)[
        "score_lbl_array_test"
    ]

    try:
        train_test_fractions = np.load(input_file, allow_pickle=True)[
            "train_test_fractions"
        ]
    except KeyError:
        train_test_fractions = [0.8, 0.1]

    try:
        class_info = list(np.load(input_file, allow_pickle=True)["class_info"])
    except KeyError:
        class_info = None
    # resolve background kl values to plot
    try:
        bkg_mask = score_lbl_tensor_train[:, 1] == 0
        kl_bkg_unique_values = list(np.unique(score_lbl_tensor_train[bkg_mask, 3]))
    except IndexError:
        kl_bkg_unique_values = [9999.0]

    if "full" in args.kl_background:
        kls_background_to_plot = ["all"] + kl_bkg_unique_values
    else:
        kl_bkg_requested = set()
        for v in args.kl_background:
            kl_bkg_requested.add(v if v == "all" else float(v))
        kls_background_to_plot = [
            kl
            for kl in ["all"] + kl_bkg_unique_values
            if (kl if kl == "all" else float(kl)) in kl_bkg_requested
        ]
    print(f"Background kl values to plot: {kls_background_to_plot}")

    plot_kl_distributions(
        score_lbl_tensor_train,
        score_lbl_tensor_test,
        args.input_dir,
        kls_background_to_plot,
        train_test_fractions[1],
        show=args.show,
        rescale=args.rescale,
        signal_eff=args.signal_eff,
        get_max_significance=False,
        class_info=class_info,
    )

    plot_roc_curve(
        score_lbl_tensor_test, args.input_dir, args.show, class_info=class_info
    )


if __name__ == "__main__":
    main()
