"""Shared sizes, colors, markers, and save paths for paper figures."""

import logging
import pathlib

import matplotlib.pyplot as plt

# Silence fontTools warnings about font timestamps during PDF export.
logging.getLogger('fontTools').setLevel(logging.ERROR)

# The paper text width is 397.485 TeX pt, or 5.5 in.
WIDTH_IN = 5.5

# Type sizes in points.
PT = {
    'panel': 9.0,
    'axis': 8.0,
    'legend': 8.0,
    'tick': 7.0,
    'annot': 8.0,
}

# Quantity colors. Red is reserved for critical signal strength.
PALETTE = {
    'theory': '#000000',
    'isotropic': '#5F5F5F',
    'ceiling': '#000000',
    'threshold': '#C1121F',
    'x': '#0072B2',
    'y': '#B86E00',
    'tcga': '#0072B2',
    'pbmc': '#CC79A7',
    'complete': '#882255',
    'grid': '#BBBBBB',
    'grouped': '#767676',
}

# Continuous-parameter colormaps.
CMAPS = {
    'theta': 'Blues',
    'rho_y': 'viridis',
    'r': 'plasma',
    'N': 'cividis',
    'cv': 'viridis',
    'R2': 'magma',
}

# Categorical colors exclude red.
CATEGORICAL = ('#2E7D6E', '#AA4499', '#8F7C1F', '#332288', '#3E7E9C', '#117733')

# Quantity markers.
MARKERS = {'x': 'o', 'y': 's', 'tcga': 'o', 'pbmc': 's', 'leverage': '^'}

# Line patterns by role.
DASHES = {
    'theory': (0, ()),            # solid
    'theory_alt': (0, (3.5, 1.4)),  # long dashes
    'isotropic': (0, (1, 2.6)),   # sparse dots
    'ceiling': (0, (4, 1.2, 1, 1.2)),  # dash-dot
    'threshold': (0, (1, 1.2)),   # tight dots
}


# Paper figure rcParams.
MANUSCRIPT_RCPARAMS = {
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'pdf.fonttype': 42,   # TrueType in PDF
    'ps.fonttype': 42,    # TrueType in PostScript
}

# Figure output directory, resolved independently of the current directory.
FIGURE_DIR = pathlib.Path(__file__).resolve().parents[1] / 'figures'


def use_manuscript_style() -> None:
    """Apply the paper figure rcParams."""
    plt.rcParams.update(MANUSCRIPT_RCPARAMS)
    plt.rcParams.update({
        'font.size': PT['axis'],
        'axes.labelsize': PT['axis'],
        'axes.titlesize': PT['axis'],
        'xtick.labelsize': PT['tick'],
        'ytick.labelsize': PT['tick'],
        'legend.fontsize': PT['legend'],
        'axes.linewidth': 0.6,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.major.size': 2.5,
        'ytick.major.size': 2.5,
    })


def figure(height_pt: float, ncols: int = 1):
    """Create a paper-width figure with height in points."""
    return plt.subplots(1, ncols, figsize=(WIDTH_IN, height_pt / 72.0),
                        constrained_layout=True)


def share_y(axes, label: str) -> None:
    """Give a row of panels one y-axis: one label, ticks on the first only."""
    for ax in axes[1:]:
        ax.sharey(axes[0])
        ax.tick_params(labelleft=False)
    axes[0].set_ylabel(label)


def halo_text(ax, x, y, text, color, **kwargs):
    """Draw text with a thin white stroke behind it."""
    import matplotlib.patheffects as pe

    kwargs.setdefault('fontsize', PT['annot'])
    kwargs.setdefault('zorder', 9)
    t = ax.text(x, y, text, color=color, **kwargs)
    t.set_path_effects([pe.withStroke(linewidth=1.6, foreground='white')])


def threshold_line(ax, x, label: str, y: float = 0.62, **kwargs):
    """Draw a red dotted vertical line at ``x`` with a rotated label on its left."""
    kwargs.setdefault('color', PALETTE['threshold'])
    kwargs.setdefault('ls', DASHES['threshold'])
    kwargs.setdefault('lw', 1.0)
    kwargs.setdefault('zorder', 2)
    ax.axvline(x, **kwargs)
    ax.text(x, y, f'{label} ', color=PALETTE['threshold'], fontsize=PT['annot'],
            rotation=90, ha='right', va='center', transform=ax.get_xaxis_transform(),
            zorder=kwargs['zorder'] + 1)


def param_colors(name: str, values, vmin=None, vmax=None, lo: float = 0.15,
                 hi: float = 0.88, norm=None):
    """Return parameter colors and their matching scalar mappable."""
    import numpy as np
    from matplotlib.colors import ListedColormap, Normalize

    values = np.asarray(values, dtype=float)
    vmin = float(values.min()) if vmin is None else float(vmin)
    vmax = float(values.max()) if vmax is None else float(vmax)
    base = plt.get_cmap(CMAPS[name])
    trimmed = ListedColormap(base(np.linspace(lo, hi, 256)))
    if norm is None:
        norm = Normalize(vmin, vmax)
    return trimmed(norm(values)), plt.cm.ScalarMappable(cmap=trimmed, norm=norm)


def panel_label(ax, label: str) -> None:
    """Add a bold panel label at the upper left."""
    ax.set_title(label, loc='left', fontsize=PT['panel'], fontweight='bold')


def save_figure(fig, name: str) -> None:
    """Write ``fig`` to figures/<name>.pdf."""
    FIGURE_DIR.mkdir(exist_ok=True)
    fig.savefig(FIGURE_DIR / f'{name}.pdf', dpi=300)
    print(f'Wrote figures/{name}.pdf')
