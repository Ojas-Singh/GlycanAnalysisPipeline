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
        
        num_torsions = len(torsions)
        # Adjust height based on the number of torsion angles
        # Increase height for fewer rows, with a minimum height of 1
        facet_height = max(1, 3 - num_torsions * 0.25) 
        
        # Adjust aspect ratio for fewer rows to make the plot less wide
        if num_torsions <= 5:
            facet_aspect = 10 # Lower aspect ratio for 1-3 rows
        else:
            facet_aspect = 20 # Default aspect ratio

        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .06, label, fontweight="bold", fontsize=32, color=color,
                   ha="left", va="center", transform=ax.transAxes)

        def add_colored_lines(x, label, **kwargs):
            # Ensure line_positions[label] has enough values for line_colors
            positions = line_positions.get(label, [])
            for i, pos in enumerate(positions):
                if i < len(line_colors): # Avoid index error if more positions than colors
                    plt.axvline(pos, color=line_colors[i], linestyle="-", linewidth=4, ymax=0.25)

        df = pd.concat([
            pd.DataFrame({'Value': data[i], 'Dataset': i}) for i in torsions
        ])
        
        g = sns.FacetGrid(df, row="Dataset", hue="Dataset", 
                         aspect=facet_aspect, height=facet_height, # Use calculated aspect and height
                         palette=sns.cubehelix_palette(len(torsions), rot=-.25, light=.7),
                         xlim=(-240, 200))
        
        # Pass only "Value" to map. 'label' will be passed as a keyword argument 
        # by FacetGrid based on the 'row' mapping.
        g.map(add_colored_lines, "Value") 
        g.map_dataframe(sns.kdeplot, "Value", bw_adjust=1, clip_on=False,
                       fill=True, alpha=1, linewidth=4, multiple='stack')
        g.refline(y=0, linewidth=4, linestyle="-", color=None, clip_on=False)
        
        # The 'label' function expects 'label' as the third argument, 
        # which FacetGrid provides as a keyword argument based on 'row'.
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
        plt.close(g.figure) # Pass the figure associated with the FacetGrid
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