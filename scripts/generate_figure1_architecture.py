#!/usr/bin/env python3
"""
Generate Figure 1: System Architecture Diagram for the MSD Inflection Detection Pipeline.

This script creates a publication-quality flowchart showing the complete pipeline
from data ingestion through evaluation.
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def create_architecture_diagram(output_path: Path, dpi: int = 300):
    """
    Create the system architecture diagram for Figure 1.

    The diagram shows:
    - Data ingestion from OpenAlex
    - Citation graph construction and Leiden clustering
    - Lineage tracking and time series generation
    - Dual-pathway inflection labeling
    - Feature engineering (55 features)
    - LightGBM classifier with isotonic calibration
    - Persistence filtering and evaluation
    """
    # Okabe-Ito color palette (color-blind friendly)
    colors = {
        'data': '#0173B2',      # Blue - data sources/storage
        'process': '#029E73',   # Green - processing steps
        'model': '#DE8F05',     # Orange - ML components
        'output': '#CC78BC',    # Purple - outputs/results
        'label': '#CA9161',     # Brown - labeling
        'bg': '#F5F5F5',        # Light gray background
        'arrow': '#333333',     # Dark gray arrows
        'text': '#000000',      # Black text
    }

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), dpi=dpi)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.set_aspect('equal')
    ax.axis('off')

    # Box style parameters
    box_style = "round,rounding_size=0.3"

    def draw_box(x, y, width, height, text, color, fontsize=11):
        """Draw a rounded rectangle with centered text."""
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle=box_style,
            facecolor=color,
            edgecolor='black',
            linewidth=1.5,
            alpha=0.9
        )
        ax.add_patch(box)

        # Handle multiline text
        lines = text.split('\n')
        total_height = len(lines) * fontsize * 0.014
        for i, line in enumerate(lines):
            line_y = y + total_height/2 - (i + 0.5) * fontsize * 0.014
            ax.text(x, line_y, line, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', color='white')

    def draw_arrow(start, end, color=colors['arrow'], style='->', connectionstyle='arc3,rad=0'):
        """Draw an arrow between two points."""
        ax.annotate('', xy=end, xytext=start,
                   arrowprops={
                       'arrowstyle': style,
                       'color': color,
                       'lw': 2,
                       'connectionstyle': connectionstyle
                   })

    def draw_small_box(x, y, width, height, text, color, fontsize=9):
        """Draw a smaller box for sub-components."""
        box = FancyBboxPatch(
            (x - width/2, y - height/2), width, height,
            boxstyle="round,rounding_size=0.15",
            facecolor=color,
            edgecolor='black',
            linewidth=1,
            alpha=0.85
        )
        ax.add_patch(box)
        # Handle multiline text in small boxes
        lines = text.split('\n')
        total_height = len(lines) * fontsize * 0.012
        for i, line in enumerate(lines):
            line_y = y + total_height/2 - (i + 0.5) * fontsize * 0.012
            ax.text(x, line_y, line, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', color='white')

    # ============================================================
    # ROW 1: Data Sources and Initial Processing (y = 7)
    # ============================================================

    # OpenAlex API
    draw_box(1.5, 7, 1.9, 0.6, "OpenAlex API\n(183K papers)", colors['data'])

    # Citation Graph
    draw_box(4.5, 7, 1.9, 0.6, "Citation Graph\nConstruction", colors['process'])

    # Leiden Clustering
    draw_box(7.5, 7, 2.0, 0.6, "Leiden Clustering\n(CPM, γ=0.001)", colors['process'])

    # Lineage Registry
    draw_box(10.5, 7, 1.9, 0.6, "Lineage Registry\n(5,179 fronts)", colors['data'])

    # Arrows Row 1
    draw_arrow((2.45, 7), (3.55, 7))
    draw_arrow((5.45, 7), (6.55, 7))
    draw_arrow((8.5, 7), (9.55, 7))

    # ============================================================
    # ROW 2: Time Series and Labeling (y = 5)
    # ============================================================

    # Time Series Aggregation
    draw_box(1.5, 5, 1.9, 0.6, "Quarterly\nTime Series", colors['data'])

    # Dual-Pathway Label box (larger container to fit sub-boxes)
    box = FancyBboxPatch(
        (5.5 - 4.0/2, 5 - 1.5/2), 4.0, 1.5,
        boxstyle=box_style,
        facecolor=colors['label'],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.9
    )
    ax.add_patch(box)
    # Title at top of box
    ax.text(5.5, 5.55, "Dual-Pathway Inflection Labeling", ha='center', va='center',
           fontsize=10, fontweight='bold', color='white')

    # Sub-boxes for labeling pathways (centered inside container)
    draw_small_box(4.4, 4.75, 1.6, 0.55, "Logistic Fit\n(77.3%)", colors['label'], fontsize=9)
    draw_small_box(6.6, 4.75, 1.6, 0.55, "Derivative\n(22.7%)", colors['label'], fontsize=9)

    # Labels output
    draw_box(10.5, 5, 1.9, 0.6, "Inflection Labels\n(538 total)", colors['output'])

    # Arrows Row 2
    draw_arrow((10.5, 6.7), (10.5, 5.3))  # From lineage to labels (vertical)
    draw_arrow((1.5, 6.7), (1.5, 5.3))    # From data to time series
    draw_arrow((2.45, 5), (3.6, 5))        # Time series to labeling
    draw_arrow((7.4, 5), (9.55, 5))        # Labeling to labels

    # ============================================================
    # ROW 3: Feature Engineering (y = 3)
    # ============================================================

    # Feature Engineering main box - draw box only, text positioned separately
    box = FancyBboxPatch(
        (4 - 4.5/2, 3 - 1.1/2), 4.5, 1.1,
        boxstyle=box_style,
        facecolor=colors['process'],
        edgecolor='black',
        linewidth=1.5,
        alpha=0.9
    )
    ax.add_patch(box)
    # Title at top of box
    ax.text(4, 3.35, "Feature Engineering (55 Features)", ha='center', va='center',
           fontsize=10, fontweight='bold', color='white')

    # Sub-boxes for feature types (positioned lower, larger fonts)
    draw_small_box(2.8, 2.65, 1.9, 0.5, "Core (20)\nGrowth, Novelty", colors['process'], fontsize=9)
    draw_small_box(5.2, 2.65, 1.9, 0.5, "Context (35)\nField-Normalized", colors['process'], fontsize=9)

    # Arrows to Feature Engineering
    draw_arrow((1.5, 4.6), (1.5, 3.5), connectionstyle='arc3,rad=0')
    ax.annotate('', xy=(1.5, 3.5), xytext=(1.5, 4.6),
               arrowprops={'arrowstyle': '->', 'color': colors['arrow'], 'lw': 2})
    draw_arrow((1.5, 3.3), (1.5, 3.3))

    # Arrow from time series to features
    draw_arrow((1.5, 4.6), (2.3, 3.6), connectionstyle='arc3,rad=-0.2')

    # Arrow from labels to features
    draw_arrow((10.5, 4.6), (6.5, 3.6), connectionstyle='arc3,rad=0.3')

    # ============================================================
    # ROW 4: Model and Output (y = 1)
    # ============================================================

    # LightGBM box
    draw_box(4, 1, 1.9, 0.6, "LightGBM\n(100 trees)", colors['model'])

    # Isotonic Calibration
    draw_box(7, 1, 1.9, 0.6, "Isotonic\nCalibration", colors['model'])

    # Persistence Filter
    draw_box(10, 1, 1.9, 0.6, "Persistence\nFiltering", colors['process'])

    # Arrows Row 4
    draw_arrow((4, 2.45), (4, 1.3))        # Features to LightGBM
    draw_arrow((4.95, 1), (6.05, 1))       # LightGBM to Calibration
    draw_arrow((7.95, 1), (9.05, 1))       # Calibration to Persistence

    # ============================================================
    # Output metrics annotation
    # ============================================================

    # Add metrics annotation
    metrics_text = "Output: PR-AUC 0.971 (retro)\nPrecision 83.6%, Recall 95.7%"
    ax.text(10, 0.2, metrics_text, ha='center', va='top',
           fontsize=8, style='italic', color=colors['text'])

    # ============================================================
    # Legend
    # ============================================================

    legend_elements = [
        mpatches.Patch(facecolor=colors['data'], edgecolor='black', label='Data'),
        mpatches.Patch(facecolor=colors['process'], edgecolor='black', label='Processing'),
        mpatches.Patch(facecolor=colors['label'], edgecolor='black', label='Labeling'),
        mpatches.Patch(facecolor=colors['model'], edgecolor='black', label='ML Model'),
        mpatches.Patch(facecolor=colors['output'], edgecolor='black', label='Output'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', frameon=True,
             fontsize=8, ncol=5, bbox_to_anchor=(0.05, -0.02))

    # Title
    ax.text(6, 7.7, "Figure 1: MSD Inflection Detection Pipeline Architecture",
           ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()

    print(f"Generated: {output_path}")
    return output_path


def main():
    """Main entry point."""
    output_dir = Path(__file__).resolve().parents[1] / "_local" / "psc" / "reporting" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "fig1_architecture.png"

    print("=" * 60)
    print("Generating Figure 1: System Architecture Diagram")
    print("=" * 60)

    create_architecture_diagram(output_path, dpi=300)

    # Verify file was created
    if output_path.exists():
        size_kb = output_path.stat().st_size / 1024
        print(f"\nSUCCESS: {output_path}")
        print(f"Size: {size_kb:.1f} KB")
    else:
        print(f"\nERROR: Failed to create {output_path}")

    print("=" * 60)


if __name__ == "__main__":
    main()
