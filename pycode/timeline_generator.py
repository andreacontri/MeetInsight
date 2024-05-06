import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DateFormatter

def create_timeline_figure(df):
    # Convert the time columns to datetime objects
    df['StartTime'] = pd.to_datetime(df['StartTime'], format='%H:%M:%S.%f')
    df['EndTime'] = pd.to_datetime(df['EndTime'], format='%H:%M:%S.%f')
    
    # Get the unique speakers and sort them
    speakers = df['Speaker'].unique()
    speakers = sorted(speakers)

    # Calculate the figure height based on the number of speakers
    # Base height per speaker plus some padding
    base_height_per_speaker = 1  # Adjust this
    fig_height = len(speakers) * base_height_per_speaker
    
    # Create the figure and axis with dynamic height
    fig, ax = plt.subplots(figsize=(8, fig_height))
    
    # Remove the spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Calculate the y-position for each speaker
    speaker_positions = {speaker: i for i, speaker in enumerate(speakers)}
    
    # Keep track of patches for legend
    legend_patches = []
    

    # Create a colormap object
    cmap = plt.get_cmap('Set2')
    colors = cmap(np.linspace(0, 1, len(speakers)))
        
    # Plot the turns for each speaker
    for idx, speaker in enumerate(speakers):
        speaker_df = df[df['Speaker'] == speaker]
        color = colors[idx]  # Get color from colormap
        for _, row in speaker_df.iterrows():
            start = row['StartTime']
            end = row['EndTime']
            ax.add_patch(patches.Rectangle((start, speaker_positions[speaker] - 0.15), end - start, 0.5, color=f'C{speaker_positions[speaker]}', alpha=0.5))
        # Add patch to legend list
        legend_patches.append(patches.Patch(color=color, label=speaker, alpha=0.5))
    
    # Add legend
    ax.legend(handles=legend_patches, loc='upper right')
    
    # Remove y-axis labels
    ax.set_yticks([])
    
    # Set the x-axis limits to the full conversation duration
    min_time = df['StartTime'].min()
    max_time = df['EndTime'].max()
    ax.set_xlim(min_time, max_time)
    
    # Set the y-axis limits and ticks
    ax.set_ylim(-0.5, len(speakers) - 0.5)
    ax.set_yticks(list(range(len(speakers))))
    ax.set_yticklabels(speakers)
    
    # Format the x-axis labels to show time
    time_format = DateFormatter('%H:%M:%S')
    ax.xaxis.set_major_formatter(time_format)
    
    # Optionally, increase the number of ticks if needed
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    
    # Add labels and title
    ax.set_title('Timeline of Conversation')
    
    return fig


# # Debugging
# from vtt_formatting import format_VTT
# vtt_filename = "data/example_transcripts.vtt"
# # with open(vtt_filename, 'r', encoding='utf-8') as file:
# #     vtt_content = file.read()
# df, formatted_content = format_VTT(vtt_filename)
# create_timeline_figure(df)
# plt.show()