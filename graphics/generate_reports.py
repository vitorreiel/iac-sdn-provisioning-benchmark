#!/usr/bin/env python3
"""
IaC Tools Comparative Analysis - Graphics Generator
Generates comparative analysis graphics for IaC tools across topologies.
"""

import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# =====================================================================
# CONFIGURATION
# =====================================================================

BASE_PATH    = Path(__file__).parent.parent / "results"
OUTPUT_DIR   = Path(__file__).parent / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

# ---- Global visual style ----
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
plt.rcParams.update({
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.35,
    'axes.labelweight':   'bold',
    'axes.titleweight':   'bold',
    'axes.titlesize':     13,
    'axes.labelsize':     11,
    'xtick.labelsize':    9,
    'ytick.labelsize':    9,
    'legend.fontsize':    9,
    'legend.framealpha':  0.92,
    'legend.edgecolor':   '#cccccc',
    'figure.facecolor':   'white',
    'axes.facecolor':     '#fafafa',
})

# ---- Color palettes ----
TOOL_COLORS = {
    'Ansible':        '#4e79a7',
    'Chef':           '#f28e2b',
    'Cloudformation': '#e15759',
    'Opentofu':       '#59a14f',
    'Pulumi':         '#76b7b2',
    'Puppet':         '#edc948',
    'Terraform':      '#b07aa1',
}

TOPO_COLORS = {
    'Fat-Tree':   '#3d7ebf',
    'Leaf-Spine': '#e05a5a',
}

def _tool_color(tool):
    return TOOL_COLORS.get(tool, '#aaaaaa')


def _set_ylim_headroom(ax, vals_list, factor=0.22, min_zero=True):
    """Set y-axis upper limit with breathing room above the highest bar."""
    max_val = max(max(v) if hasattr(v, '__iter__') else v for v in vals_list)
    bottom = 0 if min_zero else (min(min(v) if hasattr(v, '__iter__') else v for v in vals_list) * 0.9)
    ax.set_ylim(bottom, max_val * (1 + factor))
    return max_val

def _set_xlim_headroom(ax, vals_list, factor=0.18):
    """Set x-axis upper limit with breathing room for horizontal bars."""
    max_val = max(max(v) if hasattr(v, '__iter__') else v for v in vals_list)
    ax.set_xlim(0, max_val * (1 + factor))
    return max_val

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
# GRAPH 01: Duration Comparison (4 sub-metrics, grouped bars)
# =====================================================================

def plot_duration_comparison(ft, ls):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Duration Metrics Comparison: Fat-Tree vs Leaf-Spine',
                 fontsize=16, fontweight='bold', y=0.99)

    metrics = [
        ('duration_total_sec',    'Total Duration (s)',       (0, 0)),
        ('duration_install_sec',  'Install Duration (s)',     (0, 1)),
        ('duration_topology_sec', 'Topology Duration (s)',    (1, 0)),
        ('convergence_sec',       'Convergence (s)',          (1, 1)),
    ]

    for metric, title, (row, col) in metrics:
        ax = axes[row, col]
        tools     = sorted(ft.keys())
        ft_means  = [ft[t][metric].mean() for t in tools]
        ls_means  = [ls[t][metric].mean() for t in tools]
        ft_stds   = [ft[t][metric].std()  for t in tools]
        ls_stds   = [ls[t][metric].std()  for t in tools]

        x, w = np.arange(len(tools)), 0.35

        b1 = ax.bar(x - w/2, ft_means, w, yerr=ft_stds, capsize=4,
                    label='Fat-Tree',   color=TOPO_COLORS['Fat-Tree'],
                    alpha=0.85, edgecolor='white',
                    error_kw={'linewidth': 1.2, 'capthick': 1.2, 'ecolor': '#444'})
        b2 = ax.bar(x + w/2, ls_means, w, yerr=ls_stds, capsize=4,
                    label='Leaf-Spine', color=TOPO_COLORS['Leaf-Spine'],
                    alpha=0.85, edgecolor='white',
                    error_kw={'linewidth': 1.2, 'capthick': 1.2, 'ecolor': '#444'})

        ax.set_title(title, pad=10)
        ax.set_ylabel('Seconds')
        ax.set_xticks(x)
        ax.set_xticklabels(tools, rotation=35, ha='right')
        ax.yaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right')

        max_top = max(m + s for m, s in zip(ft_means + ls_means, ft_stds + ls_stds))
        ax.set_ylim(0, max_top * 1.25)

        for bars, means, stds in [(b1, ft_means, ft_stds), (b2, ls_means, ls_stds)]:
            for bar, val, std in zip(bars, means, stds):
                ax.text(bar.get_x() + bar.get_width()/2,
                        val + std + max_top * 0.015,
                        f'{val:.1f}', ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold')

    plt.tight_layout(pad=2.0, h_pad=3.5, w_pad=2.5)
    plt.savefig(OUTPUT_DIR / '01_duration_comparison.png', dpi=300, bbox_inches='tight')
    print("  ✓ 01_duration_comparison.png")
    plt.close()

# =====================================================================
# GRAPH 02: Duration Breakdown (Stacked Horizontal Bars)
# =====================================================================

def plot_total_duration_ranking(ft, ls):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    C_INST = '#3498db'
    C_TOPO = '#2ecc71'
    C_CONV = '#e74c3c'

    for idx, (topo_name, data) in enumerate([('Fat-Tree', ft), ('Leaf-Spine', ls)]):
        ax = axes[idx]
        tools = sorted(data.keys())
        inst      = [data[t]['duration_install_sec'].mean()  for t in tools]
        topo      = [data[t]['duration_topology_sec'].mean() for t in tools]
        conv      = [data[t]['convergence_sec'].mean()       for t in tools]
        inst_sd   = [data[t]['duration_install_sec'].std()   for t in tools]
        topo_sd   = [data[t]['duration_topology_sec'].std()  for t in tools]
        conv_sd   = [data[t]['convergence_sec'].std()        for t in tools]
        total     = [inst[i] + topo[i] + conv[i] for i in range(len(tools))]
        total_sd  = [np.sqrt(inst_sd[i]**2 + topo_sd[i]**2 + conv_sd[i]**2) for i in range(len(tools))]

        order    = sorted(range(len(tools)), key=lambda i: total[i])
        tools    = [tools[i]    for i in order]
        inst     = [inst[i]     for i in order]
        topo     = [topo[i]     for i in order]
        conv     = [conv[i]     for i in order]
        inst_sd  = [inst_sd[i]  for i in order]
        topo_sd  = [topo_sd[i]  for i in order]
        conv_sd  = [conv_sd[i]  for i in order]
        total    = [total[i]    for i in order]
        total_sd = [total_sd[i] for i in order]

        y     = np.arange(len(tools))
        left2 = [inst[i] + topo[i] for i in range(len(tools))]

        ax.barh(y, inst, label='Installation', color=C_INST, alpha=0.85, edgecolor='white', height=0.55)
        ax.barh(y, topo, left=inst,             label='Topology',    color=C_TOPO, alpha=0.85, edgecolor='white', height=0.55)
        ax.barh(y, conv, left=left2,            label='Convergence', color=C_CONV, alpha=0.85, edgecolor='white', height=0.55)

        ax.set_yticks(y)
        ax.set_yticklabels(tools, fontsize=10, fontweight='bold')
        ax.set_xlabel('Duration (seconds)')
        ax.set_ylabel('IaC Tools')
        ax.set_title(f'{topo_name}', pad=10)
        ax.xaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)

        max_total = max(total)
        ax.set_xlim(0, max_total * 1.28)

        # Total label with std at end of bar
        for i, (tot, sd) in enumerate(zip(total, total_sd)):
            ax.text(tot + max_total * 0.012, i, f'{tot:.1f}s (±{sd:.1f})',
                    va='center', fontsize=8, fontweight='bold')

        # Labels inside segments with std (only if wide enough)
        for i in range(len(tools)):
            threshold = max_total * 0.07
            if inst[i] > threshold:
                ax.text(inst[i]/2, i, f'{inst[i]:.0f}s\n(±{inst_sd[i]:.1f})',
                        va='center', ha='center', fontsize=7, color='white', fontweight='bold')
            if topo[i] > threshold:
                ax.text(inst[i] + topo[i]/2, i, f'{topo[i]:.0f}s\n(±{topo_sd[i]:.1f})',
                        va='center', ha='center', fontsize=7, color='white', fontweight='bold')
            if conv[i] > threshold:
                ax.text(inst[i] + topo[i] + conv[i]/2, i, f'{conv[i]:.0f}s\n(±{conv_sd[i]:.1f})',
                        va='center', ha='center', fontsize=7, color='white', fontweight='bold')

        ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout(pad=2.0, w_pad=4.0)
    plt.savefig(OUTPUT_DIR / '02_duration_breakdown.png', dpi=300, bbox_inches='tight')
    print("  ✓ 02_duration_breakdown.png")
    plt.close()

# =====================================================================
# GRAPH 03: Resource Usage (CPU & Memory, grouped bars)
# =====================================================================

def plot_resource_usage(ft, ls):
    """Avg CPU and Avg RAM as grouped horizontal bars — same layout as graph 02."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    C_CPU = '#3498db'
    C_RAM = '#e74c3c'

    for idx, (topo_name, data) in enumerate([('Fat-Tree', ft), ('Leaf-Spine', ls)]):
        ax = axes[idx]
        tools  = sorted(data.keys())
        cpu    = [data[t]['cpu_avg_pct'].mean() for t in tools]
        ram    = [data[t]['mem_avg_pct'].mean() for t in tools]
        cpu_sd = [data[t]['cpu_avg_pct'].std()  for t in tools]
        ram_sd = [data[t]['mem_avg_pct'].std()  for t in tools]

        # Sort by CPU avg (ascending)
        order  = sorted(range(len(tools)), key=lambda i: cpu[i])
        tools  = [tools[i]  for i in order]
        cpu    = [cpu[i]    for i in order]
        ram    = [ram[i]    for i in order]
        cpu_sd = [cpu_sd[i] for i in order]
        ram_sd = [ram_sd[i] for i in order]

        y = np.arange(len(tools))
        h = 0.35

        bars_cpu = ax.barh(y + h/2, cpu, h, label='Avg CPU (%)',
                           color=C_CPU, alpha=0.85, edgecolor='white')
        bars_ram = ax.barh(y - h/2, ram, h, label='Avg RAM (%)',
                           color=C_RAM, alpha=0.85, edgecolor='white')

        # Error bars at the end of each bar
        ax.errorbar(cpu, y + h/2, xerr=cpu_sd, fmt='none',
                    ecolor='#333333', elinewidth=1.4, capsize=4, capthick=1.4)
        ax.errorbar(ram, y - h/2, xerr=ram_sd, fmt='none',
                    ecolor='#333333', elinewidth=1.4, capsize=4, capthick=1.4)

        ax.set_yticks(y)
        ax.set_yticklabels(tools, fontsize=10, fontweight='bold')
        ax.set_xlabel('Usage (%)')
        ax.set_ylabel('IaC Tools')
        ax.set_title(f'{topo_name}', pad=10)
        ax.xaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)

        max_val   = max(max(cpu), max(ram))
        max_with_sd = max(c + s for c, s in zip(cpu + ram, cpu_sd + ram_sd))
        ax.set_xlim(0, min(max_with_sd * 1.18, 108))
        threshold = max_val * 0.10

        # Label inside bar: value (±sd)
        for bars, vals, sds in [(bars_cpu, cpu, cpu_sd), (bars_ram, ram, ram_sd)]:
            for bar, val, sd in zip(bars, vals, sds):
                bar_y = bar.get_y() + bar.get_height() / 2
                if val > threshold:
                    ax.text(val / 2, bar_y,
                            f'{val:.1f}% (±{sd:.1f})',
                            va='center', ha='center',
                            fontsize=7.5, color='white', fontweight='bold')

        ax.legend(loc='lower right', framealpha=0.9)

    plt.tight_layout(pad=2.0, w_pad=4.0)
    plt.savefig(OUTPUT_DIR / '03_resource_usage.png', dpi=300, bbox_inches='tight')
    print("  ✓ 03_resource_usage.png")
    plt.close()

# =====================================================================
# GRAPH 04: Network I/O
# =====================================================================

def plot_network_io(ft, ls):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Network I/O: Download (RX) vs Upload (TX)',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, (topo_name, data) in enumerate([('Fat-Tree', ft), ('Leaf-Spine', ls)]):
        ax = axes[idx]
        tools = sorted(data.keys())
        rx    = [data[t]['net_rx_mb'].mean() for t in tools]
        tx    = [data[t]['net_tx_mb'].mean() for t in tools]

        x, w = np.arange(len(tools)), 0.35

        b1 = ax.bar(x - w/2, rx, w, label='RX (MB)', color='#27ae60', alpha=0.85, edgecolor='white')
        b2 = ax.bar(x + w/2, tx, w, label='TX (MB)', color='#e67e22', alpha=0.85, edgecolor='white')

        ax.set_title(f'{topo_name}', pad=10)
        ax.set_ylabel('Data Transferred (MB)')
        ax.set_xticks(x)
        ax.set_xticklabels(tools, rotation=35, ha='right')
        ax.yaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right')

        max_val = max(max(rx), max(tx)) if rx and tx else 1
        ax.set_ylim(0, max_val * 1.22)

        for bars, vals in [(b1, rx), (b2, tx)]:
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + max_val * 0.012,
                        f'{val:.0f}', ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold')

    plt.tight_layout(pad=2.0, w_pad=3.0)
    plt.savefig(OUTPUT_DIR / '04_network_io.png', dpi=300, bbox_inches='tight')
    print("  ✓ 04_network_io.png")
    plt.close()

# =====================================================================
# GRAPH 05: Convergence Time Analysis
# =====================================================================

def plot_convergence_analysis(ft, ls):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Convergence Time per IaC Tool',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, (topo_name, data) in enumerate([('Fat-Tree', ft), ('Leaf-Spine', ls)]):
        ax = axes[idx]
        tools  = sorted(data.keys())
        means  = [data[t]['convergence_sec'].mean() for t in tools]
        stds   = [data[t]['convergence_sec'].std()  for t in tools]
        colors = [_tool_color(t) for t in tools]

        bars = ax.bar(tools, means, yerr=stds, capsize=5,
                      color=colors, alpha=0.85, edgecolor='white', linewidth=0.8,
                      error_kw={'linewidth': 1.2, 'capthick': 1.2, 'ecolor': '#555'})

        ax.set_title(f'{topo_name}', pad=10)
        ax.set_ylabel('Convergence Time (s)')
        ax.set_xticks(range(len(tools)))
        ax.set_xticklabels(tools, rotation=35, ha='right')
        ax.yaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)

        max_top = max(m + s for m, s in zip(means, stds)) if means else 1
        ax.set_ylim(0, max_top * 1.28)

        for bar, mean, std in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width()/2,
                    mean + std + max_top * 0.02,
                    f'{mean:.1f}s', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    plt.tight_layout(pad=2.0, w_pad=3.0)
    plt.savefig(OUTPUT_DIR / '05_convergence_analysis.png', dpi=300, bbox_inches='tight')
    print("  ✓ 05_convergence_analysis.png")
    plt.close()

# =====================================================================
# GRAPH 06: Box Plot — Total Duration
# =====================================================================

def plot_performance_distribution(ft, ls):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Distribution of Total Duration per Tool (Box Plot)',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, (topo_name, data) in enumerate([('Fat-Tree', ft), ('Leaf-Spine', ls)]):
        ax = axes[idx]
        tools     = sorted(data.keys())
        box_data  = [data[t]['duration_total_sec'].values for t in tools]
        colors    = [_tool_color(t) for t in tools]

        bp = ax.boxplot(
            box_data, tick_labels=tools, patch_artist=True,
            showmeans=True, meanline=True,
            whiskerprops=dict(linewidth=1.5, color='#666'),
            capprops=dict(linewidth=1.5, color='#666'),
            medianprops=dict(color='white', linewidth=2.2),
            meanprops=dict(color='#222', linewidth=1.8, linestyle='--'),
            flierprops=dict(marker='o', markerfacecolor='#aaa', markersize=3.5, alpha=0.5, linestyle='none'),
            boxprops=dict(linewidth=1.5),
        )
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_title(f'{topo_name}', pad=10)
        ax.set_ylabel('Duration (seconds)')
        ax.tick_params(axis='x', rotation=35)
        ax.yaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)

        all_vals = np.concatenate(box_data)
        clean = all_vals[np.isfinite(all_vals)]
        ax.set_ylim(max(0, clean.min() * 0.88), clean.max() * 1.12)

    plt.tight_layout(pad=2.0, w_pad=3.0)
    plt.savefig(OUTPUT_DIR / '06_performance_distribution.png', dpi=300, bbox_inches='tight')
    print("  ✓ 06_performance_distribution.png")
    plt.close()

# =====================================================================
# GRAPH 07: Efficiency Score (Horizontal Bars)
# =====================================================================

def plot_efficiency_score(ft, ls):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Composite Efficiency Score (Lower = Better)',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, (topo_name, data) in enumerate([('Fat-Tree', ft), ('Leaf-Spine', ls)]):
        ax = axes[idx]
        tools  = sorted(data.keys())
        scores = []
        for t in tools:
            df = data[t]
            dur  = df['duration_total_sec'].mean() / 100
            cpu  = df['cpu_avg_pct'].mean() / 100
            mem  = df['mem_avg_pct'].mean() / 100
            scores.append((dur * 0.5 + cpu * 0.25 + mem * 0.25) * 100)

        colors = [_tool_color(t) for t in tools]
        bars   = ax.barh(tools, scores, color=colors, alpha=0.85,
                         edgecolor='white', linewidth=0.8, height=0.55)

        ax.set_xlabel('Efficiency Score')
        ax.set_title(f'{topo_name}', pad=10)
        ax.xaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)

        max_score = max(scores) if scores else 1
        ax.set_xlim(0, max_score * 1.20)

        for bar, score in zip(bars, scores):
            ax.text(score + max_score * 0.012,
                    bar.get_y() + bar.get_height()/2,
                    f'{score:.2f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout(pad=2.0, w_pad=3.0)
    plt.savefig(OUTPUT_DIR / '07_efficiency_score.png', dpi=300, bbox_inches='tight')
    print("  ✓ 07_efficiency_score.png")
    plt.close()

# =====================================================================
# GRAPH 08: Box Plot — Duration Sub-phases (Fat-Tree vs Leaf-Spine)
# =====================================================================

def plot_boxplot_duration_phases(ft, ls):
    metrics = [
        ('duration_total_sec',    'Total Duration (s)'),
        ('duration_install_sec',  'Install Duration (s)'),
        ('duration_topology_sec', 'Topology Duration (s)'),
        ('convergence_sec',       'Convergence (s)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Duration Distribution per Phase — Fat-Tree vs Leaf-Spine (Box Plot)',
                 fontsize=16, fontweight='bold', y=0.995)

    for (metric, title), ax in zip(metrics, axes.flat):
        rows = []
        for topo_name, data in [('Fat-Tree', ft), ('Leaf-Spine', ls)]:
            for tool, df in data.items():
                for val in df[metric].values:
                    rows.append({'Tool': tool, 'Topology': topo_name, 'Value': val})
        df_plot = pd.DataFrame(rows)
        tools_order = sorted(df_plot['Tool'].unique())

        sns.boxplot(
            data=df_plot, x='Tool', y='Value', hue='Topology',
            order=tools_order,
            palette={'Fat-Tree': TOPO_COLORS['Fat-Tree'], 'Leaf-Spine': TOPO_COLORS['Leaf-Spine']},
            ax=ax, width=0.55, linewidth=1.2,
            flierprops=dict(marker='o', markersize=3, alpha=0.45),
        )

        ax.set_title(title, pad=10)
        ax.set_xlabel('')
        ax.set_ylabel('Seconds')
        ax.set_xticks(range(len(tools_order)))
        ax.set_xticklabels(tools_order, rotation=35, ha='right')
        ax.yaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(title='Topology', bbox_to_anchor=(1.01, 1), loc='upper left',
                  borderaxespad=0)

        ymax = df_plot['Value'].max()
        ax.set_ylim(0, ymax * 1.14)

    plt.tight_layout(pad=2.0, h_pad=4.0, w_pad=4.5)
    plt.savefig(OUTPUT_DIR / '08_boxplot_duration_phases.png', dpi=300, bbox_inches='tight')
    print("  ✓ 08_boxplot_duration_phases.png")
    plt.close()

# =====================================================================
# GRAPH 09: Box Plot — CPU & Memory (Fat-Tree vs Leaf-Spine)
# =====================================================================

def plot_boxplot_resources(ft, ls):
    metrics = [
        ('cpu_avg_pct', 'Avg CPU (%)'),
        ('cpu_max_pct', 'Max CPU (%)'),
        ('mem_avg_pct', 'Avg RAM (%)'),
        ('mem_max_pct', 'Max RAM (%)'),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('CPU & Memory Distribution — Fat-Tree vs Leaf-Spine (Box Plot)',
                 fontsize=16, fontweight='bold', y=0.995)

    for (metric, title), ax in zip(metrics, axes.flat):
        rows = []
        for topo_name, data in [('Fat-Tree', ft), ('Leaf-Spine', ls)]:
            for tool, df in data.items():
                for val in df[metric].values:
                    rows.append({'Tool': tool, 'Topology': topo_name, 'Value': val})
        df_plot = pd.DataFrame(rows)
        tools_order = sorted(df_plot['Tool'].unique())

        sns.boxplot(
            data=df_plot, x='Tool', y='Value', hue='Topology',
            order=tools_order,
            palette={'Fat-Tree': TOPO_COLORS['Fat-Tree'], 'Leaf-Spine': TOPO_COLORS['Leaf-Spine']},
            ax=ax, width=0.55, linewidth=1.2,
            flierprops=dict(marker='o', markersize=3, alpha=0.45),
        )

        ax.set_title(title, pad=10)
        ax.set_xlabel('')
        ax.set_ylabel(title)
        ax.set_xticks(range(len(tools_order)))
        ax.set_xticklabels(tools_order, rotation=35, ha='right')
        ax.yaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(title='Topology', bbox_to_anchor=(1.01, 1), loc='upper left',
                  borderaxespad=0)

        ymax = df_plot['Value'].max()
        ax.set_ylim(0, min(ymax * 1.14, 108))

    plt.tight_layout(pad=2.0, h_pad=4.0, w_pad=4.5)
    plt.savefig(OUTPUT_DIR / '09_boxplot_resources.png', dpi=300, bbox_inches='tight')
    print("  ✓ 09_boxplot_resources.png")
    plt.close()

# =====================================================================
# GRAPH 10: Line Chart — Duration Trend over Iterations
# =====================================================================

def plot_line_duration_trend(ft, ls):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Duration Trend over Iterations (rolling avg, window=10)',
                 fontsize=16, fontweight='bold', y=0.995)

    panels = [
        ('Fat-Tree',   ft, 'duration_total_sec', 'Total Duration (s)',  (0, 0)),
        ('Leaf-Spine', ls, 'duration_total_sec', 'Total Duration (s)',  (0, 1)),
        ('Fat-Tree',   ft, 'convergence_sec',    'Convergence (s)',     (1, 0)),
        ('Leaf-Spine', ls, 'convergence_sec',    'Convergence (s)',     (1, 1)),
    ]

    for topo_name, data, metric, ylabel, (row, col) in panels:
        ax = axes[row, col]
        tools = sorted(data.keys())

        for tool in tools:
            vals   = data[tool][metric].values
            iters  = np.arange(1, len(vals) + 1)
            color  = _tool_color(tool)
            # Raw line (faint)
            ax.plot(iters, vals, linewidth=0.8, alpha=0.25, color=color)
            # Smoothed trend (bold)
            window = min(10, len(vals))
            smoothed = pd.Series(vals).rolling(window=window, center=True, min_periods=1).mean()
            ax.plot(iters, smoothed, linewidth=2.0, color=color, label=tool)

        ax.set_title(f'{topo_name} — {ylabel}', pad=10)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)

        all_vals = np.concatenate([data[t][metric].values for t in tools])
        ax.set_ylim(max(0, all_vals.min() * 0.88), all_vals.max() * 1.14)
        ax.set_xlim(1, max(len(data[t][metric]) for t in tools))

        ax.legend(loc='upper right', framealpha=0.9, fontsize=8, ncol=2)

    plt.tight_layout(pad=2.0, h_pad=3.5, w_pad=3.0)
    plt.savefig(OUTPUT_DIR / '10_line_duration_trend.png', dpi=300, bbox_inches='tight')
    print("  ✓ 10_line_duration_trend.png")
    plt.close()

# =====================================================================
# GRAPH 11: Line Chart — CPU & Memory Trend over Iterations
# =====================================================================

def plot_line_resource_trend(ft, ls):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('CPU & Memory Trend over Iterations (rolling avg, window=10)',
                 fontsize=16, fontweight='bold', y=0.995)

    panels = [
        ('Fat-Tree',   ft, 'cpu_avg_pct', 'Avg CPU (%)',  (0, 0)),
        ('Leaf-Spine', ls, 'cpu_avg_pct', 'Avg CPU (%)',  (0, 1)),
        ('Fat-Tree',   ft, 'mem_avg_pct', 'Avg RAM (%)',  (1, 0)),
        ('Leaf-Spine', ls, 'mem_avg_pct', 'Avg RAM (%)',  (1, 1)),
    ]

    for topo_name, data, metric, ylabel, (row, col) in panels:
        ax = axes[row, col]
        tools = sorted(data.keys())

        for tool in tools:
            vals   = data[tool][metric].values
            iters  = np.arange(1, len(vals) + 1)
            color  = _tool_color(tool)
            ax.plot(iters, vals, linewidth=0.8, alpha=0.25, color=color)
            window   = min(10, len(vals))
            smoothed = pd.Series(vals).rolling(window=window, center=True, min_periods=1).mean()
            ax.plot(iters, smoothed, linewidth=2.0, color=color, label=tool)

        ax.set_title(f'{topo_name} — {ylabel}', pad=10)
        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)

        all_vals = np.concatenate([data[t][metric].values for t in tools])
        ax.set_ylim(max(0, all_vals.min() * 0.88), min(all_vals.max() * 1.14, 108))
        ax.set_xlim(1, max(len(data[t][metric]) for t in tools))

        ax.legend(loc='upper right', framealpha=0.9, fontsize=8, ncol=2)

    plt.tight_layout(pad=2.0, h_pad=3.5, w_pad=3.0)
    plt.savefig(OUTPUT_DIR / '11_line_resource_trend.png', dpi=300, bbox_inches='tight')
    print("  ✓ 11_line_resource_trend.png")
    plt.close()

# =====================================================================
# GRAPH 12: Scatter Plots — Key Metrics vs Total Provisioning Time
# =====================================================================

def plot_scatter_metrics(ft, ls):
    """6-panel scatter plot: each point is one iteration.
    Color = tool, marker shape = topology (o = Fat-Tree, x = Leaf-Spine)."""
    from matplotlib.lines import Line2D

    x_metrics = [
        ('cpu_avg_pct', 'Avg CPU (%)'),
        ('mem_avg_pct', 'Avg RAM (%)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    tools_sorted = sorted(set(list(ft.keys()) + list(ls.keys())))
    all_dur = np.concatenate([d['duration_total_sec'].values
                              for data in [ft, ls] for d in data.values()])
    y_min, y_max = all_dur.min() * 0.92, all_dur.max() * 1.08

    for (metric, xlabel), ax in zip(x_metrics, axes):
        for tool in tools_sorted:
            color = _tool_color(tool)
            if tool in ft:
                ax.scatter(ft[tool][metric], ft[tool]['duration_total_sec'],
                           marker='o', color=color, s=32, alpha=0.60, linewidths=0)
            if tool in ls:
                ax.scatter(ls[tool][metric], ls[tool]['duration_total_sec'],
                           marker='x', color=color, s=32, alpha=0.60, linewidths=1.4)

        ax.set_xlabel(xlabel)
        ax.set_ylabel('Total Duration (s)')
        ax.yaxis.grid(True, alpha=0.35, linestyle='--')
        ax.xaxis.grid(True, alpha=0.35, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_ylim(y_min, y_max)

        all_x = np.concatenate([d[metric].values
                                 for data in [ft, ls] for d in data.values()])
        ax.set_xlim(all_x.min() * 0.92, all_x.max() * 1.06)

    # ---- Legend inside Avg RAM (%) panel (axes[1]), upper left ----
    iac_title  = Line2D([0], [0], color='none', label='IaC Tools')
    tool_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=_tool_color(t),
               markersize=9, label=t)
        for t in tools_sorted
    ]
    sep = Line2D([0], [0], color='none', label='Topology')
    topo_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#555',
               markersize=9, label='Fat-Tree'),
        Line2D([0], [0], marker='x', color='#555', markersize=9,
               linewidth=0, markeredgewidth=1.8, label='Leaf-Spine'),
    ]

    leg = axes[1].legend(
        handles=[iac_title] + tool_handles + [sep] + topo_handles,
        title=None,
        loc='upper right',
        borderaxespad=0.8,
        framealpha=0.92,
        edgecolor='#cccccc',
    )
    # Bold both section headers
    for text in leg.get_texts():
        if text.get_text() in ('IaC Tools', 'Topology'):
            text.set_fontweight('bold')

    plt.tight_layout(pad=2.0, h_pad=2.8, w_pad=2.5)
    plt.savefig(OUTPUT_DIR / '12_scatter_metrics.png', dpi=300, bbox_inches='tight')
    print("  ✓ 12_scatter_metrics.png")
    plt.close()

# =====================================================================
# MAIN
# =====================================================================

def main():
    print("\n" + "="*65)
    print("IaC Tools Comparative Analysis — Graphics Generator")
    print("="*65 + "\n")

    print("Loading data...")
    ft = load_topology_data('fat-tree')
    ls = load_topology_data('leaf-spine')

    if not ft or not ls:
        print("\n✗ Error: Could not load data from both topologies")
        return 1

    print(f"\n✓ Fat-Tree:   {len(ft)} tools")
    print(f"✓ Leaf-Spine: {len(ls)} tools")
    print(f"\nGenerating 12 graphics...\n")

    try:
        plot_duration_comparison(ft, ls)
        plot_total_duration_ranking(ft, ls)
        plot_resource_usage(ft, ls)
        plot_network_io(ft, ls)
        plot_convergence_analysis(ft, ls)
        plot_performance_distribution(ft, ls)
        plot_efficiency_score(ft, ls)
        plot_boxplot_duration_phases(ft, ls)
        plot_boxplot_resources(ft, ls)
        plot_line_duration_trend(ft, ls)
        plot_line_resource_trend(ft, ls)
        plot_scatter_metrics(ft, ls)

        print("\n" + "="*65)
        print("✓ All 12 graphics generated successfully!")
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
