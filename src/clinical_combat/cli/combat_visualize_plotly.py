#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive (plotly) figure of the data before and after harmonization.

The figure contains 4 sub-figures:
    1. Scatter plot (age vs metric) with marginal histograms - BEFORE harmonization (raw)
    2. Scatter plot (age vs metric) with marginal histograms - AFTER harmonization
    3. Age curves with percentile bands (5-95 by default) - BEFORE harmonization
    4. Age curves with percentile bands (5-95 by default) - AFTER harmonization

Bundles are displayed with plotly animation frames (slider + play button at the top, no autoplay).

Percentiles are computed with a sliding age window (default: 4 years, centered,
step of 1 year). If --reference_site is given, the percentile bands are computed
on the reference site only (its raw data is used both before and after, since the
reference site is not modified by harmonization), shown in gray. Otherwise, the
gray bands are computed on all data pooled (raw for "before", harmonized for
"after"). Each moving site is shown with its own percentile bands (dotted
edges, site color) and its median curve, computed with the same sliding window
(use --hide_moving_percentiles to only keep the median curve).

Header: the harmonization method (Clinical-ComBAT / Pairwise-ComBAT, read
from the harmonized data or given with --method) and one info box per site
(N subjects, age range, mean age and, for moving sites, the mean
Bhattacharyya distance before -> after harmonization).

Tables (bottom, updated with the bundle slider):
    - Age info per site (N, age range, mean +/- SD) for the current bundle.
      Age is not modified by harmonization, so only the raw data is used.
    - QC: Bhattacharyya distance (moving vs reference) before and after
      harmonization for the current bundle, read from the files written by
      combat_QC (--bhattacharyya_raw / --bhattacharyya_harmonized, one file
      per moving site, same order). The mean distance over all bundles is
      shown in the site info boxes.

Usage examples:
combat_visualize_plotly --raw ref.raw.csv moving.raw.csv \\
                        --harmonized moving.method.harmonized.csv.gz \\
                        --reference_site REF --out_html figure.html

combat_visualize_plotly --raw ref.raw.csv moving.raw.csv \\
                        --harmonized moving.method.harmonized.csv.gz \\
                        --metric md --bundles AF_L AF_R --window_size 4

# With the QC (Bhattacharyya) files from combat_QC:
combat_visualize_plotly --raw ref.raw.csv moving.raw.csv \\
                        --harmonized moving.method.harmonized.csv.gz \\
                        --reference_site REF \\
                        --bhattacharyya_raw moving.raw.bhattacharrya.txt \\
                        --bhattacharyya_harmonized \\
                            moving.method.harmonized.bhattacharrya.txt

Output:
    By default, a standalone HTML file with plotly.js embedded (works offline,
    can be shared as a single file). Use --use_cdn for a lighter file, and
    --html_fragment to write only a <div> to insert in another HTML page.
    The camera icon of the toolbar saves the current bundle as a PNG.
"""

import argparse
import logging
import os
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from clinical_combat.utils.combatio import load_sites_data
from clinical_combat.utils.scilpy_utils import (
    add_overwrite_arg,
    add_verbose_arg,
    assert_outputs_exist,
)

SITE_COLORS = [
    "#207ef1", "#cf3d3d", "#1fa187", "#ce5873", "#8b5fbf",
    "#b38f00", "#5a8a3c", "#c0504d", "#4bacc6", "#7f7f7f",
]
BAND_COLOR = "120, 120, 120"


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--raw", nargs="+", required=True,
                   help="Raw data CSV file(s) (reference and moving sites).")
    p.add_argument("--harmonized", nargs="+", required=True,
                   help="Harmonized data CSV file(s).")
    p.add_argument("--reference_site",
                   help="Reference site name. Percentile bands are computed on "
                        "this site. If not given, bands use all sites pooled.")
    p.add_argument("--bhattacharyya_raw", nargs="+",
                   help="Bhattacharyya distance file(s) BEFORE harmonization "
                        "(*.bhattacharrya.txt from combat_QC).")
    p.add_argument("--bhattacharyya_harmonized", nargs="+",
                   help="Bhattacharyya distance file(s) AFTER harmonization, "
                        "same order as --bhattacharyya_raw.")
    p.add_argument("--method", choices=["clinical", "pairwise"],
                   help="Harmonization method displayed in the title. By "
                        "default, read from the 'harmonization' column of "
                        "the harmonized data.")
    p.add_argument("--metric",
                   help="Metric to display (default: first metric found).")
    add_figure_args(p)
    p.add_argument("--out_html", default="harmonization_plotly.html",
                   help="Output HTML file [%(default)s].")
    p.add_argument("--title", default=None, help="Figure title.")
    p.add_argument("--use_cdn", action="store_true",
                   help="Load plotly.js from a CDN instead of embedding it "
                        "in the HTML (lighter file, ~4 MB less, but needs "
                        "internet to be displayed).")
    p.add_argument("--html_fragment", action="store_true",
                   help="Write only an HTML <div> fragment (no <html>/<body>),"
                        " to be inserted in another HTML page/report.")
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


def add_figure_args(p):
    """Display options (shared with combat_report_plotly)."""
    p.add_argument("--bundles", nargs="+",
                   help="Bundles to display (default: all bundles).")
    p.add_argument("--window_size", type=float, default=4.0,
                   help="Age window (years) for the percentiles [%(default)s].")
    p.add_argument("--window_step", type=float, default=1.0,
                   help="Step (years) of the sliding window [%(default)s].")
    p.add_argument("--min_count", type=int, default=5,
                   help="Minimum number of subjects in a window [%(default)s].")
    p.add_argument("--percentiles", nargs="+", type=float,
                   default=[5, 25, 50, 75, 95],
                   help="Percentiles to display (symmetric, must contain 50) "
                        "[%(default)s].")
    p.add_argument("--nbins", type=int, default=30,
                   help="Number of bins of the marginal histograms [%(default)s].")
    p.add_argument("--hide_moving_percentiles", action="store_true",
                   help="Only display the median curve of the moving sites.")
    p.add_argument("--hide_disease", action="store_true",
                   help="Remove non-HC subjects.")


def sliding_percentiles(ages, values, percentiles, window, step, min_count,
                        age_min, age_max):
    """Percentiles of `values` in a centered sliding age window."""
    centers = np.arange(np.floor(age_min), np.ceil(age_max) + step, step)
    out = np.full((len(percentiles), len(centers)), np.nan)
    for i, c in enumerate(centers):
        mask = np.abs(ages - c) <= window / 2.0
        if mask.sum() >= min_count:
            out[:, i] = np.percentile(values[mask], percentiles)
    return centers, out


def padded_range(values, pad=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0, 1]
    vmin, vmax = values.min(), values.max()
    delta = (vmax - vmin) or abs(vmax) or 1.0
    return [vmin - pad * delta, vmax + pad * delta]


def build_traces(df_before, df_after, bundle, sites, colors, args, age_range):
    """Return (traces, y_range) for one bundle. Trace order/count is fixed."""
    traces = []
    percentiles = sorted(args.percentiles)
    n_pairs = len(percentiles) // 2
    y_all = []

    for state, df, col in (("before", df_before, 1), ("after", df_after, 3)):
        dfb = df[df["bundle"] == bundle]
        y_all.append(dfb["mean"].values)
        hist_bins = dict(start=age_range[0], end=age_range[1],
                         size=(age_range[1] - age_range[0]) / args.nbins)

        # --- Scatter + marginals (one trace per site, fixed order) ---
        for site in sites:
            d = dfb[dfb["site"] == site]
            show = state == "before"
            traces.append(go.Scatter(
                x=d["age"], y=d["mean"], mode="markers", name=site,
                legendgroup=site, showlegend=show,
                marker=dict(color=colors[site], size=6, opacity=0.65,
                            line=dict(width=0.5, color="white")),
                customdata=np.stack([d["sex"], d["disease"]], axis=-1)
                if len(d) else None,
                hovertemplate="site=" + site + "<br>age=%{x:.1f}<br>value=%{y:.4g}"
                              "<br>sex=%{customdata[0]}<br>disease=%{customdata[1]}"
                              "<extra></extra>",
                xaxis=f"x{_ax(2, col)}", yaxis=f"y{_ax(2, col)}"))
            traces.append(go.Histogram(
                x=d["age"], name=site, legendgroup=site, showlegend=False,
                marker_color=colors[site], opacity=0.6, xbins=hist_bins,
                hoverinfo="skip",
                xaxis=f"x{_ax(1, col)}", yaxis=f"y{_ax(1, col)}"))
            traces.append(go.Histogram(
                y=d["mean"], name=site, legendgroup=site, showlegend=False,
                marker_color=colors[site], opacity=0.6, nbinsy=args.nbins,
                hoverinfo="skip",
                xaxis=f"x{_ax(2, col + 1)}", yaxis=f"y{_ax(2, col + 1)}"))

        # --- Percentile bands (reference or pooled) ---
        if args.reference_site is not None:
            dref = dfb[dfb["site"] == args.reference_site]
        else:
            dref = dfb
        centers, pct = sliding_percentiles(
            dref["age"].values, dref["mean"].values, percentiles,
            args.window_size, args.window_step, args.min_count, *age_range)
        y_all.append(pct.ravel())
        xa, ya = f"x{_ax(3, col)}", f"y{_ax(3, col)}"
        band_label = (f"{args.reference_site} " if args.reference_site else "") \
            + "percentiles"
        for k in range(n_pairs):
            lo, hi = percentiles[k], percentiles[-1 - k]
            alpha = 0.12 + 0.12 * k
            traces.append(band_trace(
                centers, pct[k], pct[-1 - k],
                fillcolor=f"rgba({BAND_COLOR}, {alpha})",
                line=dict(width=0),
                name=f"{band_label} {lo:g}-{hi:g}", legendgroup=f"p{lo:g}",
                showlegend=state == "before",
                xaxis=xa, yaxis=ya))
        median = pct[n_pairs] if len(percentiles) % 2 else np.nanmean(
            pct[n_pairs - 1:n_pairs + 1], axis=0)
        traces.append(go.Scatter(
            x=centers, y=median, mode="lines",
            line=dict(color=f"rgb({BAND_COLOR})", width=2, dash="dash"),
            name=f"{band_label} median", legendgroup="median",
            showlegend=state == "before",
            hovertemplate="median<br>age=%{x}<br>value=%{y:.4g}<extra></extra>",
            xaxis=xa, yaxis=ya))

        # --- Moving sites: percentile bands (site color) + median curve ---
        # The number of traces is fixed for every bundle (animation frames):
        # the reference site gets empty traces.
        for site in sites:
            if site == args.reference_site:
                curve_x = []
                spct = np.full((len(percentiles), 0), np.nan)
            else:
                d = dfb[dfb["site"] == site]
                curve_x, spct = sliding_percentiles(
                    d["age"].values, d["mean"].values, percentiles,
                    args.window_size, args.window_step, args.min_count,
                    *age_range)
                y_all.append(spct.ravel())
            rgb = _hex_to_rgb(colors[site])
            for k in range(n_pairs):
                if args.hide_moving_percentiles:
                    break
                lo, hi = percentiles[k], percentiles[-1 - k]
                alpha = 0.10 + 0.10 * k
                traces.append(band_trace(
                    curve_x, spct[k], spct[-1 - k],
                    fillcolor=f"rgba({rgb}, {alpha})",
                    line=dict(color=f"rgba({rgb}, 0.7)", width=1, dash="dot"),
                    name=f"{site} percentiles {lo:g}-{hi:g}",
                    legendgroup=f"{site}_p{lo:g}",
                    showlegend=state == "before" and site != args.reference_site,
                    xaxis=xa, yaxis=ya))
            smedian = spct[n_pairs] if len(percentiles) % 2 else np.nanmean(
                spct[n_pairs - 1:n_pairs + 1], axis=0)
            traces.append(go.Scatter(
                x=curve_x, y=smedian, mode="lines+markers",
                name=f"{site} median", legendgroup=f"{site}_median",
                showlegend=state == "before" and site != args.reference_site,
                line=dict(color=colors[site], width=2.5),
                marker=dict(size=4),
                hovertemplate="site=" + site + " median<br>age=%{x}"
                              "<br>value=%{y:.4g}<extra></extra>",
                xaxis=xa, yaxis=ya))

    return traces, padded_range(np.concatenate(y_all))


def read_bhattacharyya(filename):
    """Read a combat_QC file: header 'HC b1 b2 ...', values 'N d1 d2 ...'."""
    with open(filename) as f:
        header = f.readline().split()
        values = f.readline().split()
    dists = {b: float(v) for b, v in zip(header[1:], values[1:])}
    return dists, int(float(values[0]))


def bhattacharyya_label(raw_file, harm_file):
    """Moving site label guessed from the file names (first token)."""
    tokens = [re.split(r"[._]", os.path.basename(f))[0]
              for f in (raw_file, harm_file)]
    return tokens[0] if tokens[0] == tokens[1] else " / ".join(tokens)


def _fmt(v, fmt="{:.3g}"):
    return "n/a" if v is None or not np.isfinite(v) else fmt.format(v)


TABLE_HEADER = dict(fill_color="#e8ecf2", align="center",
                    font=dict(size=12, color="#1f2a37"), height=26)
TABLE_CELLS = dict(align="center", font=dict(size=12, color="#1f2a37"),
                   height=24)


def build_age_table(df, bundle, sites, colors, reference_site):
    """Age info per site for one bundle (age is not changed by harmonization,
    so only the raw data is used)."""
    rows = []
    for site in sites:
        d = df[(df["bundle"] == bundle) & (df["site"] == site)]
        ages = d["age"].values
        rows.append([
            site + (" (ref)" if site == reference_site else ""),
            str(d["sid"].nunique()),
            f"{ages.min():.1f} – {ages.max():.1f}" if len(ages) else "n/a",
            f"{ages.mean():.1f} ± {ages.std():.1f}" if len(ages) else "n/a",
        ])
    cols = list(map(list, zip(*rows)))
    font_colors = [[colors[s] for s in sites]] + [["#1f2a37"] * len(rows)] * 3
    return go.Table(
        columnwidth=[1, 0.7, 1.4, 1.4],
        header=dict(values=["<b>Site</b>", "<b>N</b>",
                            "<b>Age range (y)</b>", "<b>Age mean ± SD (y)</b>"],
                    **TABLE_HEADER),
        cells=dict(values=cols,
                   fill_color=[[("white", "#f6f8fa")[i % 2]
                                for i in range(len(sites))]],
                   **{**TABLE_CELLS,
                      "font": dict(size=12, color=font_colors)}))


def rounded_box_path(x0, x1, y0, y1, rx, ry):
    """SVG path of a rounded rectangle (paper coordinates)."""
    return (f"M {x0 + rx},{y1} L {x1 - rx},{y1} Q {x1},{y1} {x1},{y1 - ry} "
            f"L {x1},{y0 + ry} Q {x1},{y0} {x1 - rx},{y0} "
            f"L {x0 + rx},{y0} Q {x0},{y0} {x0},{y0 + ry} "
            f"L {x0},{y1 - ry} Q {x0},{y1} {x0 + rx},{y1} Z")


def add_info_cards(fig, cards, y0, y1, px_per_x, px_per_y,
                   gap=0.015, max_width=0.32):
    """Add a row of rounded, semi-transparent info boxes above the plots.

    cards: list of (rgb string, html text). y0/y1: paper coordinates.
    """
    n = len(cards)
    width = min(max_width, (1 - gap * (n - 1)) / n)
    rx, ry = 10 / px_per_x, 10 / px_per_y
    for i, (rgb, text) in enumerate(cards):
        x0 = i * (width + gap)
        x1 = x0 + width
        fig.add_shape(type="path", xref="paper", yref="paper",
                      path=rounded_box_path(x0, x1, y0, y1, rx, ry),
                      fillcolor=f"rgba({rgb}, 0.10)",
                      line=dict(color=f"rgba({rgb}, 0.55)", width=1.5),
                      layer="below")
        fig.add_annotation(xref="paper", yref="paper",
                           x=x0 + 12 / px_per_x, y=y1 - 8 / px_per_y,
                           xanchor="left", yanchor="top", align="left",
                           valign="top", showarrow=False, text=text,
                           width=(x1 - x0) * px_per_x - 24,
                           height=(y1 - y0) * px_per_y - 16,
                           font=dict(size=12.5, color="#1f2a37"))


def build_qc_table(bhatt, bundle, bhatt_refs=None):
    """Bhattacharyya distance before/after harmonization for one bundle."""
    rows, fills = [], []
    for label, (raw, harm) in bhatt.items():
        pre, post = raw[0].get(bundle, np.nan), harm[0].get(bundle, np.nan)
        change = 100.0 * (post - pre) / pre if pre else np.nan
        vs = (bhatt_refs or {}).get(label)
        rows.append([label + (f" (vs {vs})" if vs else ""),
                     str(raw[1]), _fmt(pre), _fmt(post),
                     _fmt(change, "{:+.1f} %")])
        fills.append("#e3f4ea" if post < pre else
                     "#fbe5e5" if post > pre else "white")
    cols = list(map(list, zip(*rows)))
    return go.Table(
        columnwidth=[1, 0.6, 1.2, 1.2, 1],
        header=dict(values=["<b>Moving site</b>", "<b>N (HC)</b>",
                            "<b>Bhattacharyya pre</b>",
                            "<b>Bhattacharyya post</b>", "<b>Change</b>"],
                    **TABLE_HEADER),
        cells=dict(values=cols,
                   fill_color=[["white"] * len(rows)] * 4 + [fills],
                   **TABLE_CELLS))


def band_trace(x, lo, hi, name, **kwargs):
    """Closed polygon between `lo` and `hi` (windows without data dropped)."""
    x, lo, hi = (np.asarray(v, dtype=float) for v in (x, lo, hi))
    ok = np.isfinite(lo) & np.isfinite(hi)
    x, lo, hi = x[ok], lo[ok], hi[ok]
    return go.Scatter(
        x=np.concatenate([x, x[::-1]]), y=np.concatenate([hi, lo[::-1]]),
        mode="lines", fill="toself", name=name, hoveron="fills",
        hoverinfo="name", **kwargs)


def _bundle_count(d_raw, d_harm, is_reference):
    """'42 bundles' or, for a moving site with bundles that could not be
    harmonized (NaN), '41/42 bundles harmonized'."""
    n_raw, n_harm = d_raw["bundle"].nunique(), d_harm["bundle"].nunique()
    if is_reference or n_harm == n_raw:
        return f"{n_raw} bundles"
    return f"{n_harm}/{n_raw} bundles harmonized"


def _hex_to_rgb(color):
    color = color.lstrip("#")
    return ", ".join(str(int(color[i:i + 2], 16)) for i in (0, 2, 4))


def _ax(row, col):
    """Axis suffix of a 3x4 make_subplots grid (None cells skipped)."""
    idx = {(1, 1): 1, (1, 3): 2, (2, 1): 3, (2, 2): 4, (2, 3): 5,
           (2, 4): 6, (3, 1): 7, (3, 3): 8}[(row, col)]
    return "" if idx == 1 else str(idx)


METHOD_NAMES = {"clinical": "Clinical-ComBAT", "pairwise": "Pairwise-ComBAT"}

PLOTLY_CONFIG = dict(displaylogo=False,
                     toImageButtonOptions=dict(format="png", scale=2,
                                               filename="harmonization"))


def method_display_name(methods):
    """'clinical' -> 'Clinical-ComBAT' (several methods joined with ' / ')."""
    return " / ".join(
        METHOD_NAMES.get(str(m).lower(), f"{str(m).capitalize()}-ComBAT")
        for m in methods) or "ComBAT"


def prepare_data(df_raw, df_harm, metric, reference_site=None,
                 hide_disease=False):
    """Select one metric, add the reference site (unchanged by the
    harmonization) to the harmonized data and keep the common sites."""
    df_raw = df_raw[df_raw["metric"] == metric]
    df_harm = df_harm[df_harm["metric"] == metric]
    if hide_disease:
        df_raw = df_raw[df_raw["disease"] == "HC"]
        df_harm = df_harm[df_harm["disease"] == "HC"]
    if reference_site is not None:
        if reference_site not in df_raw["site"].unique():
            raise ValueError(f"Reference site '{reference_site}' not found "
                             f"in the raw data ({metric}).")
        if reference_site not in df_harm["site"].unique():
            df_harm = pd.concat(
                [df_harm, df_raw[df_raw["site"] == reference_site]])
    df_raw = df_raw[df_raw["site"].isin(set(df_harm["site"].unique()))]
    return df_raw, df_harm


def build_figure(df_raw, df_harm, metric, args, bhatt=None,
                 method_title="ComBAT", title=None, bhatt_refs=None):
    """Build the plotly figure (one animation frame per bundle).

    df_raw, df_harm: data returned by prepare_data().
    args: namespace with the add_figure_args() options + reference_site.
    bhatt: {moving site: ((dists, N), (dists, N))}, raw / harmonized
           Bhattacharyya distances (see read_bhattacharyya).
    """
    bhatt = bhatt or {}
    # Sites whose Bhattacharyya distance was computed against another site
    # than the displayed reference (e.g. chained harmonization: MTL -> QCSH).
    bhatt_refs = {k: v for k, v in (bhatt_refs or {}).items()
                  if v and v != args.reference_site}
    metric_label = metric.upper()
    harm_sites = set(df_harm["site"].unique())

    bundles = args.bundles or sorted(
        set(df_raw["bundle"]).intersection(df_harm["bundle"]))
    if not bundles:
        raise ValueError("No common bundle between raw and harmonized data.")

    sites = sorted(harm_sites, key=lambda s: (s != args.reference_site, s))
    colors = {s: SITE_COLORS[i % len(SITE_COLORS)] for i, s in enumerate(sites)}
    age_range = padded_range(
        np.concatenate([df_raw["age"].values, df_harm["age"].values]), 0.02)

    # --- Info cards: one per site (N, age) + mean QC for moving sites ---
    def qc_line(label):
        raw, harm = bhatt[label]
        pre = np.nanmean(list(raw[0].values()))
        post = np.nanmean(list(harm[0].values()))
        color = "#1a7f45" if post < pre else "#b42318"
        vs = f" vs {bhatt_refs[label]}" if label in bhatt_refs else ""
        return (f"<span style='color:#5b6573'>Bhattacharyya{vs} (mean "
                f"{len(raw[0])} bundles)</span><br>{pre:.3g} → {post:.3g} "
                f"<b><span style='color:{color}'>"
                f"({100 * (post - pre) / pre:+.1f} %)</span></b>")

    cards = []
    for site in sites:
        d = df_raw[df_raw["site"] == site]
        role = "reference" if site == args.reference_site else "moving"
        text = (f"<b><span style='font-size:16px;color:{colors[site]}'>"
                f"{site}</span></b>  <span style='color:#5b6573'>{role}</span>"
                f"<br><b>{d['sid'].nunique()}</b> subjects · "
                + _bundle_count(d, df_harm[df_harm["site"] == site],
                                site == args.reference_site)
                + f"<br>Age {d['age'].min():.1f}–{d['age'].max():.1f} y "
                f"({d['age'].mean():.1f} ± {d['age'].std():.1f})")
        if site in bhatt:
            text += "<br>" + qc_line(site)
        cards.append((_hex_to_rgb(colors[site]), text))
    for label in bhatt:
        if label not in sites:
            cards.append(("90, 101, 115", f"<b>QC {label}</b><br>"
                          + qc_line(label)))

    # --- Figure size (px): title / cards / slider above the plots; the
    #     plots keep a fixed height, the table row grows with the sites ---
    card_top, card_px = 100, 120
    margin_t = card_top + card_px + 140
    margin_b = 30 * (len(sites) + 1)
    plots_px = 840
    table_px = 28 * (len(sites) + 1) + 110
    table_frac = table_px / (plots_px + table_px)
    fig_height = margin_t + plots_px + table_px + margin_b
    paper_h, paper_w = plots_px + table_px, 1300 - 160

    def px_to_paper_y(px_from_top):
        return 1 + (margin_t - px_from_top) / paper_h

    # --- Layout: rows 1-2 = scatter + marginals, row 3 = age curves,
    #     row 4 = tables (age info / QC) ---
    fig = make_subplots(
        rows=4, cols=4,
        column_widths=[0.40, 0.08, 0.40, 0.08],
        row_heights=[f * (1 - table_frac) / 0.77 for f in (0.09, 0.34, 0.34)]
        + [table_frac],
        horizontal_spacing=0.015, vertical_spacing=0.035,
        specs=[[{}, None, {}, None],
               [{}, {}, {}, {}],
               [{}, None, {}, None],
               [{"type": "table", "colspan": 2}, None,
                {"type": "table", "colspan": 2} if bhatt else None, None]],
        subplot_titles=["Before harmonization (raw)", "After harmonization",
                        "", "", "", "", "", "",
                        "Age per site (current bundle)",
                        "QC – Bhattacharyya distance (current bundle)"
                        if bhatt else ""],
    )

    frames, y_ranges = [], {}
    for b in bundles:
        traces, y_ranges[b] = build_traces(
            df_raw, df_harm, b, sites, colors, args, age_range)
        tables = [build_age_table(df_raw, b, sites, colors,
                                  args.reference_site)]
        if bhatt:
            tables.append(build_qc_table(bhatt, b, bhatt_refs))
        frames.append(go.Frame(
            name=b, data=traces + tables,
            layout={f"yaxis{_ax(r, c)}": dict(range=y_ranges[b])
                    for r, c in [(2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 3)]}))

    n_tables = 2 if bhatt else 1
    for t in frames[0].data[:-n_tables]:
        fig.add_trace(t)
    for t, col in zip(frames[0].data[-n_tables:], (1, 3)):
        fig.add_trace(t, row=4, col=col)
    # Leave room for the age axis titles of row 3 above the tables.
    shift = 55 / paper_h
    for t in fig.data[-n_tables:]:
        t.domain.y = (t.domain.y[0], t.domain.y[1] - shift)
    for ann in fig.layout.annotations[-n_tables:]:
        ann.y -= shift
    add_info_cards(fig, cards, px_to_paper_y(card_top + card_px),
                   px_to_paper_y(card_top), paper_w, paper_h)
    fig.frames = frames

    # Axes
    for c in (1, 3):
        fig.update_xaxes(matches="x7", showticklabels=False, row=1, col=c)
        fig.update_xaxes(matches="x7", showticklabels=False, row=2, col=c)
        fig.update_yaxes(title_text="count", row=1, col=c)
        fig.update_yaxes(range=y_ranges[bundles[0]], row=2, col=c)
        fig.update_yaxes(matches="y3", showticklabels=False, row=2, col=c + 1)
        fig.update_xaxes(title_text="count", row=2, col=c + 1)
        fig.update_xaxes(title_text="Age (years)", range=age_range,
                         matches="x7" if c == 3 else None, row=3, col=c)
        fig.update_yaxes(range=y_ranges[bundles[0]], matches="y3",
                         row=3, col=c)
    fig.update_yaxes(title_text=metric_label, row=2, col=1)
    fig.update_yaxes(title_text=f"{metric_label} (window {args.window_size:g} y)",
                     row=3, col=1)
    fig.update_yaxes(showticklabels=False, row=2, col=3)
    fig.update_yaxes(showticklabels=False, row=3, col=3)

    slider_steps = [dict(
        method="animate", label=b,
        args=[[b], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                        transition=dict(duration=0))]) for b in bundles]

    fig.update_layout(
        title=dict(
            text=f"<b>{title or method_title}</b>",
            subtitle=dict(
                text=f"{metric_label} harmonization · reference: "
                f"{args.reference_site or 'pooled'} · moving: "
                f"{', '.join(s for s in sites if s != args.reference_site)}",
                font=dict(size=15, color="#5b6573")),
            font=dict(size=30, color="#1f2a37"),
            x=0.5, xanchor="center", yref="container",
            y=1 - 18 / fig_height, yanchor="top"),
        template="plotly_white", barmode="overlay",
        height=fig_height, width=1300,
        legend=dict(orientation="h", yanchor="top", y=0.0, x=0),
        margin=dict(t=margin_t, b=margin_b, l=80, r=80),
        updatemenus=[dict(
            type="buttons", direction="left", x=0, y=1.035,
            xanchor="left", yanchor="bottom", showactive=False,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=1200, redraw=True),
                                      fromcurrent=True,
                                      transition=dict(duration=0))]),
                dict(label="❚❚ Pause", method="animate",
                     args=[[None], dict(mode="immediate",
                                        frame=dict(duration=0, redraw=False),
                                        transition=dict(duration=0))]),
            ])],
        sliders=[dict(
            active=0, steps=slider_steps, x=0.16, y=1.03, len=0.84,
            xanchor="left", yanchor="bottom", pad=dict(t=0, b=10),
            currentvalue=dict(prefix="Bundle: ", font=dict(size=16)),
        )],
    )
    return fig


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_outputs_exist(parser, args, args.out_html)
    if 50 not in args.percentiles:
        parser.error("--percentiles must contain 50.")
    if bool(args.bhattacharyya_raw) != bool(args.bhattacharyya_harmonized) or (
            args.bhattacharyya_raw and len(args.bhattacharyya_raw)
            != len(args.bhattacharyya_harmonized)):
        parser.error("--bhattacharyya_raw and --bhattacharyya_harmonized "
                     "must be given together, with the same number of files.")

    df_raw = load_sites_data(args.raw)
    df_harm = load_sites_data(args.harmonized)

    methods = [args.method] if args.method else [
        m for m in df_harm["harmonization"].unique() if m != "raw"]
    metric = args.metric or df_raw["metric"].unique()[0]

    bhatt = {}
    for f_raw, f_harm in zip(args.bhattacharyya_raw or [],
                             args.bhattacharyya_harmonized or []):
        bhatt[bhattacharyya_label(f_raw, f_harm)] = (
            read_bhattacharyya(f_raw), read_bhattacharyya(f_harm))

    try:
        df_raw, df_harm = prepare_data(df_raw, df_harm, metric,
                                       args.reference_site, args.hide_disease)
        fig = build_figure(df_raw, df_harm, metric, args, bhatt,
                           method_display_name(methods), args.title)
    except ValueError as e:
        parser.error(str(e))

    fig.write_html(
        args.out_html,
        include_plotlyjs="cdn" if args.use_cdn else True,
        full_html=not args.html_fragment,
        auto_play=False,
        div_id="combat-harmonization",
        config=PLOTLY_CONFIG)
    logging.info("Saved %s", args.out_html)


if __name__ == "__main__":
    main()
