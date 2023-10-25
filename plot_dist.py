import os,json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import config
sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})





def plot_function(input_csv, output_filepath, popp_list):
    # This function will plot the data from the CSV and save the plot to the given filepath
    # For demonstration purposes, I'll create a simple plot. You can modify this as needed.
    
    data = pd.read_csv(input_csv)
    data = data.drop(columns=['i', 'cluster'])

    # Replacing 'phi', 'psi', and 'omega' with their lowercase symbols
    data.columns = data.columns.str.replace('phi', 'ϕ')
    data.columns = data.columns.str.replace('psi', 'ψ')
    data.columns = data.columns.str.replace('omega', 'ω')

    # Displaying the updated column names
    kk = data.columns.tolist()
    line_positions_dict = {torsion: data.loc[popp_list, torsion].values for torsion in kk}
    line_colors = ["#1B9C75", "#D55D02", "#746FB1", "#E12886", "#939242"]
    # Create legend elements
    legend_elements = [Line2D([0], [0], color=col, lw=4, linestyle="-", label=f"cluster {i}") 
                    for i, col in enumerate(line_colors)]
    

    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .06, label, fontweight="bold",fontsize=32, color=color,
                ha="left", va="center", transform=ax.transAxes)

    def add_colored_lines_per_kde(x, label, **kwargs):
        """Adjusted function to add colored vertical lines to the plots based on the specific torsion angle."""
        for pos, col in zip(line_positions_dict[label], line_colors):
            plt.axvline(pos, color=col, linestyle="-", linewidth=4, ymax=0.25)


    pal = sns.cubehelix_palette(len(kk), rot=-.25, light=.7)
    # a_2_3_φ = [i[0] for i in np.asarray(data[['2_3_φ']])]
    # Adjusting the data preparation to avoid duplicate axis issues
    df = pd.concat([
    
        pd.DataFrame({'Value': data[i], 'Dataset': i}) for i in kk
    ])
    
    g = sns.FacetGrid(df, row="Dataset", hue="Dataset", aspect=20, height=1, palette=pal, xlim=(-240, 200))
    bw = 1

    g.map(add_colored_lines_per_kde, "Value")
    g.map_dataframe(sns.kdeplot, "Value", bw_adjust=bw, clip_on=False, fill=True, alpha=1, linewidth=4, multiple='stack')
    g.refline(y=0, linewidth=4, linestyle="-", color=None, clip_on=False)

    # Adjusting labels
    g.map(label, "Value")

    # Setting the subplots to overlap and removing unnecessary details
    g.figure.subplots_adjust(hspace=-.75)
    g.set_titles("")
    # g.axes[0, 0].legend(handles=legend_elements, loc='upper right',fontsize=32, bbox_to_anchor=(1, 1.1), frameon=False)

    g.set(yticks=[],xticks=[-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180], ylabel="")
    desired_font_size = 24  # or any other desired size

    # ... [rest of your code]

    # Set xlabel font size
    # g.set_xlabels(label="Angle (degrees)", fontsize=desired_font_size)

    # Set xtick font size
    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelsize=desired_font_size)
    g.despine(bottom=True, left=True)

    plt.savefig(output_filepath,transparent=True ,dpi=450)


    # data.plot()
    # plt.savefig(output_filepath)

# Starting directory
# start_dir = "/mnt/database/DB"  # Replace with the path to your directory
start_dir = config.data_dir

for dirpath, dirnames, filenames in os.walk(start_dir):
    # Checking if the folder name is "pack"
    if os.path.basename(dirpath) == "pack":
        # Constructing the path to the "torsions.csv" file
        csv_filepath = os.path.join(dirpath, "torsions.csv")
        if os.path.exists(csv_filepath):
            print(csv_filepath)
            # If the CSV file exists, create the plot
            output_filepath = os.path.join(dirpath, "dist.svg")
            # Reading and loading the JSON data
            
            try:
                if not os.path.exists(output_filepath):
                    with open(os.path.join(dirpath, "info.json"), 'r') as file:
                        data = json.load(file)

                    # Extracting the 'popp' list
                    popp_list = data.get('popp', [])
                    plot_function(csv_filepath, output_filepath , popp_list)
            except:
                print("error!")

# Labeling function



