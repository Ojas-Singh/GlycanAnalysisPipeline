from pathlib import Path
from typing import List, Dict, Any
import logging
import json

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import lib.config as config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_plot_elements(data: pd.DataFrame, popp_list: List[int]) -> tuple[Dict, List[Line2D]]:
    """Create plot elements for distribution visualization."""
    data.columns = (data.columns
                   .str.replace('phi', 'ϕ')
                   .str.replace('psi', 'ψ')
                   .str.replace('omega', 'ω'))
    
    torsions = data.columns.tolist()
    line_positions = {torsion: data.loc[popp_list, torsion].values for torsion in torsions}
    line_colors = ["#1B9C75", "#D55D02", "#746FB1", "#E12886", "#939242"]
    
    legend_elements = [
        Line2D([0], [0], color=col, lw=4, linestyle="-", label=f"cluster {i}") 
        for i, col in enumerate(line_colors)
    ]
    
    return line_positions, legend_elements, line_colors, torsions

def plot_distribution(input_csv: Path, output_filepath: Path, popp_list: List[int]) -> None:
    """Plot torsion angle distributions.
    
    Args:
        input_csv: Path to input CSV file
        output_filepath: Path to save output plot
        popp_list: List of population indices
    """
    try:
        data = pd.read_csv(input_csv)
        data = data.drop(columns=['i', 'cluster'])
        
        line_positions, legend_elements, line_colors, torsions = create_plot_elements(data, popp_list)
        
        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .06, label, fontweight="bold", fontsize=32, color=color,
                   ha="left", va="center", transform=ax.transAxes)

        def add_colored_lines(x, label, **kwargs):
            for pos, col in zip(line_positions[label], line_colors):
                plt.axvline(pos, color=col, linestyle="-", linewidth=4, ymax=0.25)

        df = pd.concat([
            pd.DataFrame({'Value': data[i], 'Dataset': i}) for i in torsions
        ])
        
        g = sns.FacetGrid(df, row="Dataset", hue="Dataset", 
                         aspect=20, height=1, 
                         palette=sns.cubehelix_palette(len(torsions), rot=-.25, light=.7),
                         xlim=(-240, 200))
        
        g.map(add_colored_lines, "Value")
        g.map_dataframe(sns.kdeplot, "Value", bw_adjust=1, clip_on=False,
                       fill=True, alpha=1, linewidth=4, multiple='stack')
        g.refline(y=0, linewidth=4, linestyle="-", color=None, clip_on=False)
        g.map(label, "Value")
        
        g.figure.subplots_adjust(hspace=-.75)
        g.set_titles("")
        g.set(yticks=[], 
              xticks=[-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180],
              ylabel="")
        
        for ax in g.axes.flat:
            ax.tick_params(axis='x', labelsize=24)
        g.despine(bottom=True, left=True)
        
        plt.savefig(output_filepath, transparent=True, dpi=450)
        plt.close(g)
        logger.info(f"Plot saved to {output_filepath}")
        
    except Exception as e:
        logger.error(f"Failed to create plot: {str(e)}")
        pass

def main() -> None:
    """Process all directories and create distribution plots."""
    start_dir = Path(config.data_dir)
    logger.info(f"Processing directories in {start_dir}")
    
    for pack_dir in start_dir.rglob("pack"):
        csv_filepath = pack_dir / "torsions.csv"
        output_filepath = pack_dir / "dist.svg"

        if output_filepath.exists():
            logger.info(f"Skipping {pack_dir}: dist.svg already exists")
            continue

        if csv_filepath.exists():
            logger.info(f"Processing {csv_filepath}")
            try:
                with open(pack_dir / "info.json", 'r') as file:
                    data = json.load(file)

                popp_list = data.get('popp', [])
                plot_distribution(csv_filepath, output_filepath, popp_list)
            except Exception as e:
                logger.error(f"Error processing {pack_dir}: {str(e)}")

if __name__ == "__main__":
    main()