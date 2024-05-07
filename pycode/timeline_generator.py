import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DateFormatter

def create_timeline_figure(df):
    """
    Generate a timeline figure illustrating the durations each speaker spoke in a conversation.
    This function uses the Matplotlib library to create a timeline from a DataFrame that contains
    start and end times along with speaker identifiers.

    Args:
        df (pd.DataFrame): DataFrame containing the columns 'StartTime', 'EndTime', and 'Speaker',
        where time columns are in the format HH:MM:SS.sss.

    Returns:
        matplotlib.figure.Figure: The matplotlib figure object representing the timeline.
    """
    # Convert start and end times in the DataFrame to datetime objects
    df['StartTime'] = pd.to_datetime(df['StartTime'], format='%H:%M:%S.%f')
    df['EndTime'] = pd.to_datetime(df['EndTime'], format='%H:%M:%S.%f')
    
    print(df.head())
    # Extract and sort unique speakers
    speakers = sorted(df['Speaker'].unique())
    
    # Define the base height per speaker and calculate total figure height
    base_height_per_speaker = 1  # Height per speaker in inches
    fig_height = len(speakers) * base_height_per_speaker
    
    # Create a figure and axis with dynamically calculated height
    fig, ax = plt.subplots(figsize=(8, fig_height))
    
    # Hide unnecessary spines (borders)
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_visible(False)
    
    # Map speakers to vertical positions on the plot
    speaker_positions = {speaker: idx for idx, speaker in enumerate(speakers)}

    # Create a color map for the speakers
    cmap = plt.get_cmap('Set2')
    colors = cmap(np.linspace(0, 1, len(speakers)))
    
    # Initialize legend patches for adding to the legend later
    legend_patches = []

    # Plot each speaker's speaking durations as rectangles on the timeline
    for idx, speaker in enumerate(speakers):
        speaker_df = df[df['Speaker'] == speaker]
        color = colors[idx]
        for _, row in speaker_df.iterrows():
            start = row['StartTime']
            end = row['EndTime']
            ax.add_patch(
                patches.Rectangle(
                    (start, speaker_positions[speaker] - 0.15), end - start, 0.5, color=color, alpha=0.5
                )
            )
        legend_patches.append(patches.Patch(color=color, label=speaker, alpha=0.5))
    
    # Add a legend to the plot
    ax.legend(handles=legend_patches, loc='upper right')

    # Format the x-axis with appropriate time labels and set axis limits
    ax.xaxis.set_major_formatter(DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.set_xlim(df['StartTime'].min(), df['EndTime'].max())

    # Set y-axis ticks and labels
    ax.set_yticks(range(len(speakers)))
    ax.set_yticklabels(speakers)
    ax.set_ylim(-0.5, len(speakers) - 0.5)

    # Remove y-axis tick marks
    ax.tick_params(axis='y', which='both', left=False)

    # Add a title to the plot
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