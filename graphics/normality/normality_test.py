#!/usr/bin/env python3
"""
Shapiro-Wilk Normality Tests for IaC Tool Benchmark Results.

Outputs (in normality/outputs/):
  shapiro_wilk_results.csv      — full table (W, p-value, verdict per tool/topology/metric)
  01_pvalue_heatmap.png         — heatmap: p-values by tool × metric (Fat-Tree | Leaf-Spine)
  02_qq_timing_fat_tree.png     — Q-Q plots: timing metrics, Fat-Tree
  03_qq_timing_leaf_spine.png   — Q-Q plots: timing metrics, Leaf-Spine
  04_qq_resource_fat_tree.png   — Q-Q plots: resource metrics, Fat-Tree
  05_qq_resource_leaf_spine.png — Q-Q plots: resource metrics, Leaf-Spine
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
from pathlib import Path
from scipy import stats

# =====================================================================
# CONFIGURATION
# =====================================================================

BASE_PATH  = Path(__file__).parent.parent.parent / "results"
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
plt.rcParams.update({
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.titleweight':  'bold',
    'axes.titlesize':    11,
    'axes.labelsize':    10,
    'xtick.labelsize':   8,
    'ytick.labelsize':   8,
    'figure.facecolor':  'white',
    'axes.facecolor':    '#fafafa',
})

TOOL_COLORS = {
    'Ansible':        '#4e79a7',
    'Chef':           '#f28e2b',
    'Cloudformation': '#e15759',
    'Opentofu':       '#59a14f',
    'Pulumi':         '#76b7b2',
    'Puppet':         '#edc948',
    'Terraform':      '#b07aa1',
}

def _tool_color(tool):
    return TOOL_COLORS.get(tool, '#aaaaaa')

TIMING_METRICS = [
    ('duration_total_sec',    'Total Duration (s)'),
    ('duration_install_sec',  'Install Duration (s)'),
    ('duration_topology_sec', 'Topology Duration (s)'),
    ('convergence_sec',       'Convergence (s)'),
]

RESOURCE_METRICS = [
    ('cpu_avg_pct',   'Avg CPU (%)'),
    ('mem_avg_pct',   'Avg RAM (%)'),
    ('net_rx_mb',     'Network RX (MB)'),
    ('disk_write_mb', 'Disk Write (MB)'),
]

ALL_METRICS = TIMING_METRICS + RESOURCE_METRICS

# =====================================================================
# DATA LOADING
# =====================================================================

def load_topology_data(topology_name):
    data = {}
    topology_path = BASE_PATH / topology_name
    if not topology_path.exists():
        print(f"  ✗ Path not found: {topology_path}")
        return data
    for tool_dir in sorted(topology_path.iterdir()):
        if tool_dir.is_dir():
            csv_file = tool_dir / "results.csv"
            if csv_file.exists():
                tool_name = tool_dir.name.capitalize()
                try:
                    df = pd.read_csv(csv_file)
                    data[tool_name] = df
                    print(f"  ✓ {topology_name}/{tool_name}: {len(df)} records")
                except Exception as e:
                    print(f"  ✗ {topology_name}/{tool_name}: {e}")
    return data

# =====================================================================
# SHAPIRO-WILK TEST
# =====================================================================

def run_shapiro_wilk(ft, ls):
    """Run Shapiro-Wilk for every tool × topology × metric combination."""
    rows = []
    for topo_name, data in [('Fat-Tree', ft), ('Leaf-Spine', ls)]:
        for tool in sorted(data.keys()):
            df = data[tool]
            for col, label in ALL_METRICS:
                if col not in df.columns:
                    continue
                vals = df[col].dropna().values
                if len(vals) < 3:
                    continue
                # Shapiro-Wilk is most reliable for n ≤ 5000; use it directly
                w_stat, p_val = stats.shapiro(vals)
                rows.append({
                    'Topology': topo_name,
                    'Tool':     tool,
                    'Metric':   label,
                    'Column':   col,
                    'N':        len(vals),
                    'W':        round(w_stat, 6),
                    'p_value':  round(p_val, 6),
                    'Normal_α0.05': 'Yes' if p_val > 0.05 else 'No',
                    'Normal_α0.01': 'Yes' if p_val > 0.01 else 'No',
                })
    return pd.DataFrame(rows)

def print_summary(df):
    print("\n" + "="*65)
    print("SHAPIRO-WILK NORMALITY TEST — SUMMARY")
    print("="*65)
    for topo in ['Fat-Tree', 'Leaf-Spine']:
        sub = df[df['Topology'] == topo]
        n_normal = (sub['Normal_α0.05'] == 'Yes').sum()
        n_total  = len(sub)
        print(f"\n  {topo}: {n_normal}/{n_total} metric-tool pairs are normal (α=0.05)")

    print("\n  Non-normal distributions (α=0.05):")
    non_normal = df[df['Normal_α0.05'] == 'No'][['Topology','Tool','Metric','W','p_value']]
    if non_normal.empty:
        print("  — None")
    else:
        for _, row in non_normal.iterrows():
            print(f"  [{row['Topology']}] {row['Tool']:14s} {row['Metric']:28s} "
                  f"W={row['W']:.4f}  p={row['p_value']:.4f}")
    print()

# =====================================================================
# GRAPH 01: P-Value Heatmap
# =====================================================================

def plot_pvalue_heatmap(df):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Shapiro-Wilk p-Values — Heatmap (red = non-normal at α=0.05)',
                 fontsize=14, fontweight='bold', y=0.98)

    for idx, topo in enumerate(['Fat-Tree', 'Leaf-Spine']):
        ax = axes[idx]
        sub = df[df['Topology'] == topo]
        pivot = sub.pivot(index='Tool', columns='Metric', values='p_value')

        # Order columns by ALL_METRICS list
        col_order = [lbl for _, lbl in ALL_METRICS if lbl in pivot.columns]
        pivot = pivot[col_order]

        # Custom colormap: red (p≤0.05, non-normal) → green (p>0.05, normal)
        cmap = sns.diverging_palette(10, 145, s=80, l=50, as_cmap=True)

        sns.heatmap(
            pivot, ax=ax, annot=True, fmt='.3f',
            cmap=cmap, vmin=0, vmax=0.20, center=0.05,
            linewidths=0.5, linecolor='#dddddd',
            annot_kws={'size': 8},
            cbar_kws={'label': 'p-value (red ≤ 0.05 = non-normal)', 'shrink': 0.8},
        )

        ax.set_title(f'{topo}', pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right', fontsize=8.5)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)

        # Mark cells where p ≤ 0.05 with a border
        for i, tool in enumerate(pivot.index):
            for j, metric in enumerate(pivot.columns):
                pval = pivot.loc[tool, metric]
                if pd.notna(pval) and pval <= 0.05:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                               edgecolor='#cc0000', lw=2.0))

    plt.tight_layout(pad=2.0, w_pad=3.5)
    plt.savefig(OUTPUT_DIR / '01_pvalue_heatmap.png', dpi=300, bbox_inches='tight')
    print("  ✓ 01_pvalue_heatmap.png")
    plt.close()

# =====================================================================
# GRAPH 02–05: Q-Q Plots (Timing + Resource × Fat-Tree + Leaf-Spine)
# =====================================================================

def _plot_qq_group(data, topo_name, metric_group, filename, group_title):
    """Q-Q plots for a group of metrics, one topology."""
    n_metrics = len(metric_group)
    tools     = sorted(data.keys())
    n_tools   = len(tools)

    fig, axes = plt.subplots(n_metrics, n_tools, figsize=(n_tools * 2.6, n_metrics * 2.6))
    fig.suptitle(f'Q-Q Plots — {group_title} | {topo_name}',
                 fontsize=14, fontweight='bold', y=0.995)

    # Ensure axes is always 2D
    if n_metrics == 1:
        axes = axes[np.newaxis, :]
    if n_tools == 1:
        axes = axes[:, np.newaxis]

    for row, (col, metric_label) in enumerate(metric_group):
        for c_idx, tool in enumerate(tools):
            ax = axes[row, c_idx]
            df = data[tool]

            if col not in df.columns or df[col].dropna().empty:
                ax.set_visible(False)
                continue

            vals = df[col].dropna().values
            color = _tool_color(tool)

            # Q-Q plot
            (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist='norm')
            ax.plot(osm, osr, 'o', markersize=3.5, color=color, alpha=0.65)
            ax.plot(osm, slope * np.array(osm) + intercept,
                    '-', color='#444444', linewidth=1.3, alpha=0.85)

            # Shapiro-Wilk result as annotation
            w_stat, p_val = stats.shapiro(vals)
            verdict_color = '#cc0000' if p_val <= 0.05 else '#1a7a1a'
            ax.text(0.97, 0.04, f'W={w_stat:.3f}\np={p_val:.3f}',
                    transform=ax.transAxes, fontsize=7, va='bottom', ha='right',
                    color=verdict_color, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=1.5))

            ax.yaxis.grid(True, alpha=0.3)
            ax.xaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)

            # Row labels (metric) on left
            if c_idx == 0:
                ax.set_ylabel(metric_label, fontsize=8.5, fontweight='bold')
            else:
                ax.set_ylabel('')

            # Column labels (tool) on top
            if row == 0:
                ax.set_title(tool, fontsize=9, fontweight='bold',
                             color=color, pad=5)
            else:
                ax.set_title('')

            ax.set_xlabel('')
            ax.tick_params(labelsize=7)

    plt.tight_layout(pad=1.5, h_pad=1.8, w_pad=1.5)
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches='tight')
    print(f"  ✓ {filename}")
    plt.close()

def plot_all_qq(ft, ls):
    _plot_qq_group(ft, 'Fat-Tree',   TIMING_METRICS,   '02_qq_timing_fat_tree.png',   'Timing Metrics')
    _plot_qq_group(ls, 'Leaf-Spine', TIMING_METRICS,   '03_qq_timing_leaf_spine.png', 'Timing Metrics')
    _plot_qq_group(ft, 'Fat-Tree',   RESOURCE_METRICS, '04_qq_resource_fat_tree.png', 'Resource Metrics')
    _plot_qq_group(ls, 'Leaf-Spine', RESOURCE_METRICS, '05_qq_resource_leaf_spine.png','Resource Metrics')

# =====================================================================
# MAIN
# =====================================================================

def main():
    print("\n" + "="*65)
    print("Shapiro-Wilk Normality Tests — IaC Tools Benchmark")
    print("="*65 + "\n")

    print("Loading data...")
    ft = load_topology_data('fat-tree')
    ls = load_topology_data('leaf-spine')

    if not ft or not ls:
        print("\n✗ Error: Could not load data from both topologies")
        return 1

    print(f"\n✓ Fat-Tree:   {len(ft)} tools")
    print(f"✓ Leaf-Spine: {len(ls)} tools")

    # Run tests
    print("\nRunning Shapiro-Wilk tests...")
    results = run_shapiro_wilk(ft, ls)

    # Save CSV
    csv_path = OUTPUT_DIR / 'shapiro_wilk_results.csv'
    results.to_csv(csv_path, index=False)
    print(f"  ✓ shapiro_wilk_results.csv  ({len(results)} rows)")

    # Print summary
    print_summary(results)

    # Generate figures
    print("Generating figures...\n")
    try:
        plot_pvalue_heatmap(results)
        plot_all_qq(ft, ls)

        print("\n" + "="*65)
        print("✓ Normality tests complete!")
        print(f"✓ Output: {OUTPUT_DIR}")
        print("="*65 + "\n")
        return 0

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
