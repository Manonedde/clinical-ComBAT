#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file interactive HTML report of all the harmonizations.
All raw file should be in the same folder (only the files used to harmonize).
Resulting harmonized folders should be in the same folder (e.g. QC-SH, QC-MTL, ...).

Example : 
    - Raw files: raw/SH_md_mean_raw.csv, raw/MTL_md_mean_raw.csv, raw/QC_md_mean_raw.csv
    - Harmonized files: Clinical_QC-SH/*, Clinical_QC-MTL/*

The report contains:
    - an "Overview" tab: one row per harmonization (metric, method, reference,
      moving site) with N subjects, N bundles and the mean Bhattacharyya
      distance before / after harmonization (click on a row to open the
      corresponding figure), and a per-bundle table for each metric: mean
      value of the reference, mean value of each moving site before / after
      harmonization (with the difference to the reference) and the
      Bhattacharyya distance before -> after (click on a bundle to open the
      figure at this bundle), and a "Bundle completeness" table: bundles
      that are not present for all the subjects of a site (n/N per site)
      with the list of the missing subjects (absent / NaN in the raw data,
      or lost during the harmonization);
    - one tab per metric (MD, FA, ...), with one sub-tab per harmonization
      method / reference site (Clinical-ComBAT, Pairwise-ComBAT, ...). Each
      sub-tab holds the interactive figure of combat_visualize_plotly
      (scatter + marginals, age curves with percentiles, tables, one
      animation frame per bundle).

The harmonized files are grouped automatically using their content:
    metric           -> 'metric' column
    method           -> 'harmonization' column (clinical, pairwise, ...)
    reference site   -> 'model' column ('{moving}-{reference}.{metric}.{method}')
                        or --reference_site.
All the moving sites harmonized on the same reference with the same method
are displayed in the same figure.

Bhattacharyya files (combat_QC) are searched automatically in the directory
of each harmonized file:
    after  : {harmonized file without .csv(.gz)}.bhattacharrya.txt
    before : *.bhattacharrya.txt whose name contains the moving site and the
             metric and not 'harmonized' (e.g. SH_md_mean_raw.bhattacharrya.txt)
Additional candidate files can be given with --bhattacharyya.

The plotly figures are rendered when their tab is opened (fast loading) and
plotly.js is embedded only once (use --use_cdn for a lighter file).

Usage example:
combat_report_plotly --raw raw/*.csv \\
                     --harmonized QC-SH/*.harmonized.csv.gz \\
                                  QC-MTL/*.harmonized.csv.gz \\
                     --out_html combat_report.html
"""

import argparse
import datetime
import glob
import html
import json
import logging
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly
import plotly.io as pio
from plotly.offline import get_plotlyjs

try:
    from clinical_combat.cli.combat_visualize_plotly import (
        PLOTLY_CONFIG,
        SITE_COLORS,
        add_figure_args,
        build_figure,
        method_display_name,
        prepare_data,
        read_bhattacharyya,
    )
except ModuleNotFoundError:
    # The installed clinical_combat package is older than this script (e.g.
    # installed from another clone): use the module next to this file.
    # Because my pip install are not update (sorry)
    # NEED TO BE FIX AFTER !!!
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from combat_visualize_plotly import (
        PLOTLY_CONFIG,
        SITE_COLORS,
        add_figure_args,
        build_figure,
        method_display_name,
        prepare_data,
        read_bhattacharyya,
    )
from clinical_combat.utils.combatio import load_results_csv, load_sites_data
from clinical_combat.utils.scilpy_utils import (
    add_overwrite_arg,
    add_verbose_arg,
    assert_outputs_exist,
)


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--raw", nargs="+", required=True,
                   help="Raw data CSV files (all sites, all metrics).")
    p.add_argument("--harmonized", nargs="+", required=True,
                   help="Harmonized data CSV files (all methods, metrics).")
    p.add_argument("--bhattacharyya", nargs="+", default=[],
                   help="Additional Bhattacharyya files (combat_QC) to "
                        "search, if they are not next to the harmonized "
                        "files.")
    p.add_argument("--reference_site",
                   help="Reference site (default: read from the 'model' "
                        "column of the harmonized data).")
    p.add_argument("--title", default="ComBAT harmonization report",
                   help="Report title [%(default)s].")
    p.add_argument("--out_html", default="combat_report.html",
                   help="Output HTML file [%(default)s].")
    p.add_argument("--use_cdn", action="store_true",
                   help="Load plotly.js from a CDN instead of embedding it "
                        "(~4.6 MB lighter, needs internet).")
    add_figure_args(p)
    add_verbose_arg(p)
    add_overwrite_arg(p)
    return p


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------
def _strip_csv(path):
    return re.sub(r"\.csv(\.gz)?$", "", path)


def _tokens(path):
    return set(re.split(r"[._\-]", os.path.basename(path)))


def guess_reference(df, moving_site):
    """'SH-QC.md.clinical' -> 'QC' (None if it cannot be read)."""
    for model in df["model"].unique():
        prefix = str(model).split(".")[0]
        if prefix.startswith(f"{moving_site}-"):
            return prefix[len(moving_site) + 1:]
    return None


def find_bhattacharyya(harm_file, site, metric, extra_files):
    """Return (raw file, harmonized file) or None."""
    harm_bhatt = _strip_csv(harm_file) + ".bhattacharrya.txt"
    candidates = set(glob.glob(os.path.join(
        os.path.dirname(os.path.abspath(harm_file)), "*.bhattacharrya.txt")))
    candidates.update(os.path.abspath(f) for f in extra_files)
    if not os.path.exists(harm_bhatt):
        same_name = [f for f in candidates
                     if os.path.basename(f) == os.path.basename(harm_bhatt)]
        if not same_name:
            return None
        harm_bhatt = same_name[0]
    raw = [f for f in candidates
           if "harmonized" not in os.path.basename(f)
           and {site, metric} <= _tokens(f)]
    if len(raw) != 1:
        logging.warning("Bhattacharyya file before harmonization not found "
                        "(or ambiguous) for %s / %s: %s", site, metric, raw)
        return None
    return raw[0], harm_bhatt


def _new_group():
    return dict(dfs=[], bhatt={}, files=[], refs={})


def collect_harmonizations(args):
    """Group the harmonized data by (metric, method, reference site), the
    reference being the one used for the harmonization (from 'model')."""
    groups = defaultdict(_new_group)
    for f in args.harmonized:
        df = load_results_csv(f)  # rows with NaN are removed
        if df.empty:
            logging.warning("Skipping %s: no valid data (all values are "
                            "NaN?).", f)
            continue
        for (metric, method, site), d in df.groupby(
                ["metric", "harmonization", "site"]):
            ref = guess_reference(d, site) or args.reference_site
            g = groups[(metric, method, ref)]
            g["dfs"].append(d)
            g["files"].append(f)
            g["refs"][site] = ref
            pair = find_bhattacharyya(f, site, metric, args.bhattacharyya)
            if pair:
                g["bhatt"][site] = tuple(read_bhattacharyya(x) for x in pair)
            logging.info("%s: %s, %s, %s -> %s (QC: %s)", os.path.basename(f),
                         metric, method, site, ref, "yes" if pair else "no")
    return groups


def merge_chained_references(groups, raw_sites, reference_site=None):
    """Chained harmonizations (e.g. SH -> QC, then MTL -> QCSH where QCSH =
    QC + harmonized SH): a reference that is not a raw site is an
    intermediate merged site, so its harmonizations are displayed with the
    final reference (QC), each site keeping its own raw / harmonized data.
    With --reference_site, all harmonizations use this reference."""
    merged = defaultdict(_new_group)
    for (metric, method, ref), g in sorted(groups.items(), key=str):
        root = ref
        if reference_site is not None:
            root = reference_site
        elif ref not in raw_sites:
            roots = {k[2] for k in groups
                     if k[:2] == (metric, method) and k[2] in raw_sites}
            if len(roots) == 1:
                root = roots.pop()
                logging.info("%s / %s: reference %s is not a raw site, "
                             "displayed with reference %s.",
                             metric, method, ref, root)
            else:
                logging.warning("%s / %s: reference %s not found in the raw "
                                "data; use --reference_site.",
                                metric, method, ref)
        m = merged[(metric, method, root)]
        for key in ("dfs", "files"):
            m[key] += g[key]
        for key in ("bhatt", "refs"):
            m[key].update(g[key])
    return merged


# --------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------
CSS = """
:root {
  --bg: #f4f6f9; --surface: #ffffff; --text: #1f2a37; --muted: #5b6573;
  --border: #dde3ea; --accent: #207ef1; --accent-soft: rgba(32,126,241,.10);
  --good: #1a7f45; --good-soft: #e3f4ea; --bad: #b42318; --bad-soft: #fbe5e5;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.45 -apple-system, "Segoe UI", Roboto, "Helvetica Neue",
  Arial, sans-serif; }
header { background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 22px 32px 0; position: sticky; top: 0; z-index: 10; }
h1 { margin: 0; font-size: 26px; letter-spacing: -.01em; }
.subtitle { color: var(--muted); margin: 4px 0 14px; }
.tabs { display: flex; gap: 4px; flex-wrap: wrap; }
.tab { border: 0; background: none; padding: 10px 18px; font: inherit;
  font-weight: 600; color: var(--muted); cursor: pointer;
  border-bottom: 3px solid transparent; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }
main { padding: 24px 32px 48px; }
.panel { display: none; }
.panel.active { display: block; }
.pills { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 16px; }
.pill { border: 1px solid var(--border); background: var(--surface);
  border-radius: 999px; padding: 6px 16px; font: inherit; cursor: pointer;
  color: var(--muted); }
.pill:hover { border-color: var(--accent); color: var(--text); }
.pill.active { background: var(--accent-soft); border-color: var(--accent);
  color: var(--accent); font-weight: 600; }
.card { background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 16px; overflow-x: auto; }
.figure { width: 1300px; margin: 0 auto; min-height: 400px; }
.loading { color: var(--muted); padding: 40px; text-align: center; }
.kpis { display: grid; gap: 12px; margin-bottom: 20px;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); }
.kpi { background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; padding: 14px 16px; }
.kpi .label { color: var(--muted); font-size: 12px; text-transform: uppercase;
  letter-spacing: .04em; }
.kpi .value { font-size: 24px; font-weight: 700; margin-top: 2px; }
table { border-collapse: collapse; width: 100%; }
th, td { padding: 9px 12px; text-align: left;
  border-bottom: 1px solid var(--border); white-space: nowrap; }
th { color: var(--muted); font-size: 12px; text-transform: uppercase;
  letter-spacing: .04em; background: #f8fafc; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr { cursor: pointer; }
tbody tr:hover { background: var(--accent-soft); }
.chip { display: inline-block; border-radius: 999px; padding: 2px 10px;
  font-weight: 600; font-size: 12px; }
.chip.good { background: var(--good-soft); color: var(--good); }
.chip.bad { background: var(--bad-soft); color: var(--bad); }
.site-dot { display: inline-block; width: 9px; height: 9px;
  border-radius: 50%; margin-right: 6px; background: var(--muted); }
.muted { color: var(--muted); font-size: 12px; }
h2 { font-size: 18px; margin: 28px 0 12px; }
.toolbar { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  margin-bottom: 12px; }
.toolbar .pills { margin: 0; }
.filter { margin-left: auto; border: 1px solid var(--border);
  border-radius: 8px; padding: 6px 10px; font: inherit; min-width: 200px; }
th.group { text-align: center; border-bottom: 2px solid var(--site);
  color: var(--site); }
td.sep, th.sep { border-left: 1px solid var(--border); }
.delta { font-size: 12px; margin-left: 4px; }
.delta.good { color: var(--good); } .delta.bad { color: var(--bad); }
.table-wrap { max-height: 640px; overflow: auto; }
.table-wrap.tall { max-height: calc(100vh - 230px); min-height: 480px; }
.table-wrap.compact { max-height: 260px; }
.table-wrap.compact th, .table-wrap.compact td { padding: 5px 10px;
  font-size: 13px; }
.nwarn { color: #b54708; font-size: 11px; margin-left: 6px;
  font-weight: 600; white-space: nowrap; }
.warn-dot { display: inline-block; width: 7px; height: 7px;
  border-radius: 50%; background: #f79009; margin-left: 6px;
  vertical-align: middle; }
td.subjects { white-space: normal; max-width: 640px; font-size: 13px; }
.stage { display: inline-block; border-radius: 6px; padding: 1px 8px;
  font-size: 12px; font-weight: 600; }
.stage.raw { background: #fef0c7; color: #93370d; }
.stage.harm { background: var(--bad-soft); color: var(--bad); }
.table-wrap thead th { position: sticky; top: 0; z-index: 1; }
.table-wrap thead tr:nth-child(2) th { top: 37px; }
.note { color: var(--muted); font-size: 12px; margin-top: 10px; }
footer { color: var(--muted); font-size: 12px; text-align: center;
  padding: 0 0 24px; }
"""

JS = """
const rendered = new Map();
function renderFigure(id) {
  if (rendered.has(id)) return rendered.get(id);
  const el = document.getElementById(id);
  const data = document.getElementById(id + '-data');
  if (!el || !data) return Promise.resolve();
  const fig = JSON.parse(data.textContent);
  el.innerHTML = '';
  const p = Plotly.newPlot(el, {data: fig.data, layout: fig.layout,
                                frames: fig.frames || [], config: CONFIG});
  rendered.set(id, p);
  return p;
}
function goToBundle(id, bundle) {
  renderFigure(id).then(() => Plotly.animate(id, [bundle], {
    mode: 'immediate', frame: {duration: 0, redraw: true},
    transition: {duration: 0}}));
}
function showSub(metric, sub) {
  document.querySelectorAll(`[data-metric="${metric}"] .pill`).forEach(
    b => b.classList.toggle('active', b.dataset.sub === sub));
  document.querySelectorAll(`[data-metric="${metric}"] .sub`).forEach(
    p => p.classList.toggle('active', p.id === sub));
  renderFigure(sub + '-fig');
  if (sub) history.replaceState(null, '', '#' + sub);
}
function showTab(tab, sub) {
  document.querySelectorAll('.tab').forEach(
    b => b.classList.toggle('active', b.dataset.tab === tab));
  document.querySelectorAll('main > .panel').forEach(
    p => p.classList.toggle('active', p.id === tab));
  const panel = document.getElementById(tab);
  const subEl = sub && document.getElementById(sub);
  if (subEl) {
    showSub(subEl.closest('[data-metric]').dataset.metric, sub);
  } else if (panel && panel.dataset.metric) {
    const first = panel.querySelector('.pill');
    showSub(panel.dataset.metric, first && first.dataset.sub);
  } else {
    history.replaceState(null, '', '#' + tab);
  }
  if (!subEl || panel.dataset.metric) window.scrollTo({top: 0});
}
// Deep links: report.html#metric-md or report.html#fig-md-clinical-QC
function showFromHash() {
  const el = document.getElementById(location.hash.slice(1));
  if (!el) return;
  if (el.classList.contains('sub')) showTab(el.closest('section').id, el.id);
  else if (el.matches('main > .panel')) showTab(el.id);
}
window.addEventListener('DOMContentLoaded', showFromHash);
document.addEventListener('click', e => {
  const t = e.target.closest('[data-tab]');
  if (t && t.classList.contains('tab')) showTab(t.dataset.tab);
  const p = e.target.closest('.pill');
  if (p) showSub(p.closest('[data-metric]').dataset.metric, p.dataset.sub);
  const r = e.target.closest('tr[data-goto]');
  if (r) {
    showTab(r.dataset.goto, r.dataset.sub);
    if (r.dataset.bundle) goToBundle(r.dataset.sub + '-fig', r.dataset.bundle);
  }
});
document.addEventListener('input', e => {
  if (!e.target.classList.contains('filter')) return;
  const q = e.target.value.toLowerCase();
  e.target.closest('.card').querySelectorAll('tbody tr').forEach(
    tr => tr.style.display =
      tr.textContent.toLowerCase().includes(q) ? '' : 'none');
});
"""


def _slug(*parts):
    return re.sub(r"[^a-zA-Z0-9]+", "-", "-".join(map(str, parts))).strip("-")


def _json_for_script(fig):
    """Figure JSON safe to embed in a <script> tag."""
    return pio.to_json(fig, validate=False).replace("</", "<\\/")


def missing_subjects(df_raw, df_harm, ref, metric, method,
                     expected_bundles=()):
    """Subjects without value for a bundle (rows with NaN are removed when
    the data is loaded, so a missing value = an absent row).

    raw:        subject present for this site / metric but not for the bundle;
    harmonized: subject with a raw value but no harmonized value (moving
                sites only, the reference is not harmonized).
    """
    rows = []
    bundles = sorted(set(df_raw["bundle"]) | set(df_harm["bundle"])
                     | set(expected_bundles))
    for site in sorted(df_raw["site"].unique(), key=lambda s: (s != ref, s)):
        d_raw = df_raw[df_raw["site"] == site]
        d_harm = df_harm[df_harm["site"] == site]
        all_sids = set(d_raw["sid"])
        raw_sets = d_raw.groupby("bundle")["sid"].apply(set)
        harm_sets = d_harm.groupby("bundle")["sid"].apply(set)
        for b in bundles:
            raw_b = raw_sets.get(b, set())
            stages = [("raw", all_sids - raw_b)]
            if site != ref:
                stages.append(("harmonized", raw_b - harm_sets.get(b, set())))
            for stage, missing in stages:
                if missing:
                    rows.append(dict(
                        metric=metric, method=method, ref=ref, site=site,
                        bundle=b, stage=stage, n_total=len(all_sids),
                        n_avail=len(raw_b), missing=sorted(map(str, missing)),
                        all_lost=(stage == "harmonized"
                                  and not harm_sets.get(b, set()))))
    return rows


def bundle_completeness(missing_rows, df_raw, df_harm, ref):
    """Per incomplete bundle: {site: info} for every site of the figure.

    info: n_total (subjects of the site), n_raw (subjects with a raw value),
    n_harm (with a harmonized value, None for the reference), missing (raw),
    lost (raw value but no harmonized value), all_lost.
    """
    sites = sorted(df_raw["site"].unique(), key=lambda s: (s != ref, s))
    n_total = df_raw.groupby("site")["sid"].nunique()
    raw_n = df_raw.groupby(["site", "bundle"])["sid"].nunique()
    harm_n = df_harm.groupby(["site", "bundle"])["sid"].nunique()
    comp = {}
    for b in sorted({r["bundle"] for r in missing_rows}):
        comp[b] = {}
        for site in sites:
            miss = [r for r in missing_rows
                    if r["bundle"] == b and r["site"] == site]
            raw = [r for r in miss if r["stage"] == "raw"]
            lost = [r for r in miss if r["stage"] == "harmonized"]
            comp[b][site] = dict(
                n_total=int(n_total.get(site, 0)),
                n_raw=int(raw_n.get((site, b), 0)),
                n_harm=None if site == ref else int(harm_n.get((site, b), 0)),
                missing=raw[0]["missing"] if raw else [],
                lost=lost[0]["missing"] if lost else [],
                all_lost=bool(lost and lost[0]["all_lost"]))
    return sites, comp


def render_missing_section(completeness):
    """completeness: list of (metric, method, ref, sites, comp)."""
    # Same bundle with exactly the same missing subjects in several metrics
    # (e.g. FA and MD) -> one row.
    merged = {}
    all_sites, refs = [], []
    for metric, method, ref, sites, comp in completeness:
        refs.append(ref)
        for site in sites:
            if site not in all_sites:
                all_sites.append(site)
        for b, infos in comp.items():
            sig = tuple(sorted(
                (site, i["n_total"], i["n_raw"], i["n_harm"],
                 tuple(i["missing"]), tuple(i["lost"]))
                for site, i in infos.items()))
            key = (b, method, ref, sig)
            merged.setdefault(key, dict(metrics=[], infos=infos))
            merged[key]["metrics"].append(metric)
    all_sites.sort(key=lambda s: (s not in refs, s))

    if not merged:
        return """
<h2>Bundle completeness</h2>
<div class='card'><div class='note' style='margin:0'>All the bundles are
present for all the subjects of every site, before and after
harmonization.</div></div>"""

    colors = _site_colors(all_sites, refs[0] if refs else None)
    body = []
    for (b, method, ref, _), m in sorted(merged.items(),
                                         key=lambda kv: (kv[0][0],
                                                         kv[1]["metrics"])):
        metric = m["metrics"][0]
        tab = _slug("metric", metric)
        sub = _slug("fig", metric, method, ref)
        cells, subjects = [], []
        for site in all_sites:
            i = m["infos"].get(site)
            if i is None:
                cells.append("<td class='num muted'>–</td>")
                continue
            if i["n_raw"] == 0:
                txt = "<span class='nwarn' style='margin:0'>absent</span>"
            elif i["n_raw"] < i["n_total"]:
                txt = (f"<span class='nwarn' style='margin:0'>"
                       f"{i['n_raw']}/{i['n_total']}</span>")
            else:
                txt = f"<span class='muted'>✓ {i['n_raw']}/{i['n_total']}</span>"
            if i["n_harm"] is not None and i["n_harm"] < i["n_raw"]:
                txt += (f"<br><span class='stage harm'>post "
                        f"{i['n_harm']}/{i['n_raw']}</span>")
            cells.append(f"<td class='num'>{txt}</td>")

            name = (f"<b style='color:{colors[site]}'>"
                    f"{html.escape(str(site))}</b>")
            if i["n_raw"] == 0:
                subjects.append(f"{name}: <i>bundle absent for all "
                                f"subjects</i>")
            elif i["missing"]:
                subjects.append(f"{name}: "
                                + html.escape(", ".join(i["missing"])))
            if i["lost"]:
                subjects.append(
                    f"{name} <span class='stage harm'>harmonization</span> "
                    + ("<i>all subjects – bundle not harmonized</i>"
                       if i["all_lost"] else
                       html.escape(", ".join(i["lost"]))))
        body.append(
            f"<tr data-goto='{tab}' data-sub='{sub}' "
            f"data-bundle='{html.escape(str(b))}'>"
            f"<td><b>{html.escape(str(b))}</b></td>"
            f"<td>{html.escape(', '.join(x.upper() for x in m['metrics']))}"
            f"</td>{''.join(cells)}"
            f"<td class='subjects'>{'<br>'.join(subjects)}</td></tr>")

    site_heads = "".join(
        f"<th style='text-align:right;color:{colors[site]}'>"
        f"{html.escape(str(site))}"
        + (" (ref)" if site in refs else "") + "</th>" for site in all_sites)
    return f"""
<h2>Bundle completeness</h2>
<div class='card'>
<div class='toolbar'><span class='muted'>{len(merged)} incomplete bundle
row(s)</span><input class='filter' type='search'
placeholder='Filter (bundle, site, subject)…'></div>
<div class='table-wrap compact'><table>
<thead><tr><th>Bundle</th><th>Metric</th>{site_heads}
<th>Missing subjects</th></tr></thead>
<tbody>{''.join(body)}</tbody></table></div>
<div class='note'>Bundles that are not present for all the subjects of a
site. n/N: subjects with a value for this bundle / subjects of the site.
<span class='stage harm'>post</span>: subjects that still have a value after
the harmonization (e.g. NaN propagated by ComBAT). Rows are merged when the
metrics have exactly the same missing subjects. Click on a row to open the
figure at this bundle.</div>
</div>"""


def overview_rows(groups):
    rows = []
    for (metric, method, ref), g in sorted(groups.items(), key=str):
        df = pd.concat(g["dfs"])
        for site in sorted(df["site"].unique()):
            pre = post = np.nan
            if site in g["bhatt"]:
                (raw, _), (harm, _) = g["bhatt"][site]
                pre = np.nanmean(list(raw.values()))
                post = np.nanmean(list(harm.values()))
            incomplete = sorted({
                (m["bundle"], len(m["missing"]), m["n_total"])
                for m in g.get("missing", [])
                if m["site"] == site and m["stage"] == "raw"})
            rows.append(dict(metric=metric, method=method, ref=ref,
                             qc_ref=g["refs"].get(site),
                             incomplete=incomplete,
                             site=site, n=df.loc[df.site == site,
                                                 "sid"].nunique(),
                             nb=df.loc[df.site == site, "bundle"].nunique(),
                             pre=pre, post=post))
    return rows


def render_overview(rows, groups, bundle_tables=(), raw_bundles=(),
                    missing_rows=(), completeness=()):
    def fmt(v):
        return "–" if not np.isfinite(v) else f"{v:.3g}"

    body = []
    for r in rows:
        change = 100 * (r["post"] - r["pre"]) / r["pre"] \
            if np.isfinite(r["pre"]) and r["pre"] else np.nan
        chip = ("–" if not np.isfinite(change) else
                f"<span class='chip {'good' if change < 0 else 'bad'}'>"
                f"{change:+.1f} %</span>")
        tab = _slug("metric", r["metric"])
        sub = _slug("fig", r["metric"], r["method"], r["ref"])
        body.append(
            f"<tr data-goto='{tab}' data-sub='{sub}'>"
            f"<td><b>{html.escape(r['metric'].upper())}</b></td>"
            f"<td>{html.escape(method_display_name([r['method']]))}</td>"
            f"<td>{html.escape(str(r['ref'] or 'pooled'))}"
            + (f" <span class='muted'>(harmonized on {html.escape(r['qc_ref'])})"
               f"</span>" if r["qc_ref"] and r["qc_ref"] != r["ref"] else "")
            + "</td>"
            f"<td>{html.escape(str(r['site']))}</td>"
            f"<td class='num'>{r['n']}</td>"
            f"<td class='num'>{r['nb']}</td>"
            + ("<td class='muted'>–</td>" if not r["incomplete"] else
               "<td>" + ", ".join(
                   f"{html.escape(str(b))}<span class='nwarn'>"
                   f"−{n}/{t}</span>" for b, n, t in r["incomplete"])
               + "</td>")
            + f"<td class='num'>{fmt(r['pre'])}</td>"
            f"<td class='num'>{fmt(r['post'])}</td>"
            f"<td class='num'>{chip}</td></tr>")

    metrics = sorted({k[0] for k in groups})
    methods = sorted({k[1] for k in groups})
    sites = sorted({r["site"] for r in rows})
    refs = sorted({str(k[2] or "pooled") for k in groups})
    bundles = set()
    for g in groups.values():
        for d in g["dfs"]:
            bundles.update(d["bundle"].unique())
    kpis = [("Metrics", ", ".join(m.upper() for m in metrics)),
            ("Methods", ", ".join(method_display_name([m]) for m in methods)),
            ("Reference", ", ".join(refs)),
            ("Moving sites", ", ".join(map(str, sites))),
            ("Bundles", str(len(bundles)) if not set(raw_bundles) - bundles
             else f"{len(bundles)} / {len(set(raw_bundles))}"),
            ("Incomplete bundles", str(len(
                {m["bundle"] for m in missing_rows if m["stage"] == "raw"}))
             or "0"),
            ("Figures", str(len(groups)))]
    kpi_html = "".join(
        f"<div class='kpi'><div class='label'>{html.escape(k)}</div>"
        f"<div class='value'>{html.escape(v)}</div></div>" for k, v in kpis)
    return (f"""
<div class='kpis'>{kpi_html}</div>
<div class='card'>
<table>
<thead><tr><th>Metric</th><th>Method</th><th>Reference</th>
<th>Moving site</th><th style='text-align:right'>N subjects</th>
<th style='text-align:right'>N bundles</th>
<th>Bundles with missing subjects</th>
<th style='text-align:right'>Bhattacharyya pre</th>
<th style='text-align:right'>Bhattacharyya post</th>
<th style='text-align:right'>Change</th></tr></thead>
<tbody>{''.join(body)}</tbody>
</table>
<div class='note'>Mean Bhattacharyya distance over all bundles (combat_QC),
between the moving site and the site it was harmonized on. N bundles:
bundles with harmonized values. Click on a row to open its figure.</div>"""
    + (f"<div class='note'><b>Not harmonized</b> (NaN in the harmonized "
       f"files): {html.escape(', '.join(sorted(set(raw_bundles) - bundles)))}"
       f"</div>" if set(raw_bundles) - bundles else "") + """
</div>""" + render_bundle_section(bundle_tables)
        + render_missing_section(completeness))


def _site_colors(sites, reference_site):
    order = sorted(sites, key=lambda s: (s != reference_site, s))
    return {s: SITE_COLORS[i % len(SITE_COLORS)] for i, s in enumerate(order)}


def bundle_table(df_raw, df_harm, ref, bhatt, bhatt_refs, tab, sub):
    """Per bundle: reference mean, moving sites mean before / after
    harmonization (difference to the reference) and Bhattacharyya distance."""
    sites = sorted(df_harm["site"].unique(), key=lambda s: (s != ref, s))
    moving = [s for s in sites if s != ref]
    colors = _site_colors(sites, ref)
    bundles = sorted(set(df_raw["bundle"]) | set(df_harm["bundle"]))
    raw_mean = df_raw.groupby(["site", "bundle"])["mean"].mean()
    raw_n = df_raw.groupby(["site", "bundle"])["sid"].nunique()
    harm_n = df_harm.groupby(["site", "bundle"])["sid"].nunique()
    n_total = df_raw.groupby("site")["sid"].nunique()

    def nbadge(counts, site, bundle):
        """'n=15/20' when subjects are missing for this site / bundle."""
        n, tot = counts.get((site, bundle), 0), n_total.get(site, 0)
        return "" if n >= tot else f"<span class='nwarn'>n={n}/{tot}</span>"

    harm_mean = df_harm.groupby(["site", "bundle"])["mean"].mean()

    def get(series, site, bundle):
        return series.get((site, bundle), np.nan)

    def fmt(v):
        return "–" if not np.isfinite(v) else f"{v:.4g}"

    def with_delta(v, ref_v, other=None):
        if not (np.isfinite(v) and np.isfinite(ref_v) and ref_v):
            return fmt(v)
        d = 100 * (v - ref_v) / ref_v
        cls = ""
        if other is not None and np.isfinite(other):
            cls = "good" if abs(d) <= abs(other) else "bad"
        return f"{fmt(v)}<span class='delta {cls}'>({d:+.1f} %)</span>", d

    head1 = ["<th rowspan='2'>Bundle</th>"]
    head2 = []
    if ref in sites:
        head1.append(f"<th class='group sep' style='--site:{colors[ref]}'>"
                     f"{html.escape(str(ref))} (ref)</th>")
        head2.append("<th class='sep' style='text-align:right'>Mean</th>")
    for site in moving:
        vs = bhatt_refs.get(site)
        vs = f" vs {html.escape(vs)}" if vs and vs != ref else ""
        head1.append(f"<th class='group sep' colspan='3' "
                     f"style='--site:{colors[site]}'>"
                     f"{html.escape(str(site))}</th>")
        head2 += ["<th class='sep' style='text-align:right'>Mean pre</th>",
                  "<th style='text-align:right'>Mean post</th>",
                  f"<th style='text-align:right'>Bhattacharyya{vs} "
                  f"pre → post</th>"]

    body = []
    for b in bundles:
        ref_v = get(raw_mean, ref, b)
        incomplete = any(raw_n.get((s, b), 0) < n_total.get(s, 0)
                         or (s != ref and harm_n.get((s, b), 0)
                             < raw_n.get((s, b), 0)) for s in sites)
        cells = [f"<td><b>{html.escape(str(b))}</b>"
                 + ("<span class='warn-dot' title='missing subjects'></span>"
                    if incomplete else "") + "</td>"]
        if ref in sites:
            cells.append(f"<td class='num sep'>{fmt(ref_v)}"
                         f"{nbadge(raw_n, ref, b)}</td>")
        for site in moving:
            pre_v, post_v = get(raw_mean, site, b), get(harm_mean, site, b)
            pre = with_delta(pre_v, ref_v)
            pre_txt, pre_d = pre if isinstance(pre, tuple) else (pre, np.nan)
            post = with_delta(post_v, ref_v, pre_d)
            post_txt = post[0] if isinstance(post, tuple) else post
            q = "–"
            if site in bhatt:
                (raw_d, _), (harm_d, _) = bhatt[site]
                bp, bq = raw_d.get(b, np.nan), harm_d.get(b, np.nan)
                if np.isfinite(bp) and np.isfinite(bq):
                    cls = "good" if bq <= bp else "bad"
                    q = (f"{bp:.3g} → <span class='delta {cls}' "
                         f"style='margin:0;font-size:inherit'>{bq:.3g}</span>")
            cells += [f"<td class='num sep'>{pre_txt}"
                      f"{nbadge(raw_n, site, b)}</td>",
                      f"<td class='num'>{post_txt}"
                      f"{nbadge(harm_n, site, b)}</td>",
                      f"<td class='num'>{q}</td>"]
        body.append(f"<tr data-goto='{tab}' data-sub='{sub}' "
                    f"data-bundle='{html.escape(str(b))}'>{''.join(cells)}</tr>")
    return (f"<div class='table-wrap tall'><table><thead><tr>{''.join(head1)}</tr>"
            f"<tr>{''.join(head2)}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>")


def render_bundle_section(bundle_tables):
    """bundle_tables: list of (id, label, table html)."""
    if not bundle_tables:
        return ""
    pills = "".join(
        f"<button class='pill{' active' if i == 0 else ''}' "
        f"data-sub='{tid}'>{html.escape(label)}</button>"
        for i, (tid, label, _) in enumerate(bundle_tables))
    subs = "".join(
        f"<div class='panel sub{' active' if i == 0 else ''}' id='{tid}'>"
        f"{table}</div>"
        for i, (tid, _, table) in enumerate(bundle_tables))
    return f"""
<h2>Metrics per bundle (pre vs post harmonization)</h2>
<div class='card' data-metric='bundles'>
<div class='toolbar'><div class='pills'>{pills}</div>
<input class='filter' type='search' placeholder='Filter bundles…'></div>
{subs}
<div class='note'>Mean value per bundle. In parentheses: difference to the
reference mean (green when the post-harmonization value is closer to the
reference than the raw value). <span class='nwarn' style='margin:0'>n=15/20
</span>: number of subjects with a value when some are missing (see Missing
subjects below). Click on a bundle to open the figure at this bundle.</div>
</div>"""


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))
    assert_outputs_exist(parser, args, args.out_html)
    if 50 not in args.percentiles:
        parser.error("--percentiles must contain 50.")

    df_raw_all = load_sites_data(args.raw)
    groups = merge_chained_references(
        collect_harmonizations(args), set(df_raw_all["site"].unique()),
        args.reference_site)
    if not groups:
        parser.error("No harmonized data found.")

    # --- One figure per (metric, method, reference) ---
    by_metric = defaultdict(list)
    bundle_tables = []
    missing_rows = []
    completeness = []
    for (metric, method, ref), g in sorted(groups.items(), key=str):
        fig_args = argparse.Namespace(**vars(args))
        fig_args.reference_site = ref
        try:
            df_raw, df_harm = prepare_data(df_raw_all, pd.concat(g["dfs"]),
                                           metric, ref, args.hide_disease)
            fig = build_figure(df_raw, df_harm, metric, fig_args, g["bhatt"],
                               method_display_name([method]),
                               bhatt_refs=g["refs"])
        except ValueError as e:
            logging.warning("Skipping %s / %s / %s: %s", metric, method, ref, e)
            continue
        label = method_display_name([method])
        if len({k[2] for k in groups if k[0] == metric and k[1] == method}) > 1:
            label += f" → {ref}"
        expected = set(df_raw_all.loc[df_raw_all["metric"] == metric,
                                      "bundle"])
        g["missing"] = missing_subjects(df_raw, df_harm, ref, metric, method,
                                        expected)
        missing_rows += g["missing"]
        completeness.append((metric, method, ref) + bundle_completeness(
            g["missing"], df_raw, df_harm, ref))
        sub_id = _slug("fig", metric, method, ref)
        by_metric[metric].append((sub_id, label, _json_for_script(fig)))
        bundle_tables.append((
            _slug("bundles", metric, method, ref),
            f"{metric.upper()} · {label}",
            bundle_table(df_raw, df_harm, ref, g["bhatt"], g["refs"],
                         _slug("metric", metric), sub_id)))
        logging.info("Figure: %s / %s / %s", metric, method, ref)

    metrics = {k[0] for k in groups}
    raw_bundles = set(df_raw_all.loc[df_raw_all["metric"].isin(metrics),
                                     "bundle"])

    # --- Tabs ---
    tabs = ["<button class='tab active' data-tab='overview'>Overview</button>"]
    overview = render_overview(overview_rows(groups), groups, bundle_tables,
                               raw_bundles, missing_rows, completeness)
    panels = [f"<section class='panel active' id='overview'>{overview}"
              f"</section>"]
    for metric in sorted(by_metric):
        tab_id = _slug("metric", metric)
        tabs.append(f"<button class='tab' data-tab='{tab_id}'>"
                    f"{html.escape(metric.upper())}</button>")
        pills, subs = [], []
        for sub_id, label, fig_json in by_metric[metric]:
            pills.append(f"<button class='pill' data-sub='{sub_id}'>"
                         f"{html.escape(label)}</button>")
            subs.append(
                f"<div class='panel sub' id='{sub_id}'><div class='card'>"
                f"<div class='figure' id='{sub_id}-fig'>"
                f"<div class='loading'>Loading figure…</div></div></div>"
                f"<script type='application/json' id='{sub_id}-fig-data'>"
                f"{fig_json}</script></div>")
        panels.append(
            f"<section class='panel' id='{tab_id}' "
            f"data-metric='{html.escape(metric)}'>"
            f"<div class='pills'>{''.join(pills)}</div>{''.join(subs)}"
            f"</section>")

    if args.use_cdn:
        plotlyjs = (f"<script src='https://cdn.plot.ly/plotly-"
                    f"{plotly.offline.get_plotlyjs_version()}.min.js' "
                    f"charset='utf-8'></script>")
    else:
        plotlyjs = f"<script type='text/javascript'>{get_plotlyjs()}</script>"

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    n_figs = sum(len(v) for v in by_metric.values())
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(args.title)}</title>
<style>{CSS}</style>
{plotlyjs}
</head>
<body>
<header>
  <h1>{html.escape(args.title)}</h1>
  <div class="subtitle">{len(by_metric)} metric(s) · {n_figs} harmonization
  figure(s) · generated {now}</div>
  <nav class="tabs">{''.join(tabs)}</nav>
</header>
<main>{''.join(panels)}</main>
<footer>clinical-ComBAT · combat_report_plotly</footer>
<script>const CONFIG = {json.dumps(PLOTLY_CONFIG)};{JS}</script>
</body>
</html>"""

    with open(args.out_html, "w", encoding="utf-8") as f:
        f.write(page)
    logging.info("Saved %s (%.1f MB)", args.out_html,
                 os.path.getsize(args.out_html) / 1e6)


if __name__ == "__main__":
    main()
