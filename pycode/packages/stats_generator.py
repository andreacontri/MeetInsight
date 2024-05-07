import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_stats_figure(df):
    """
    Generate a figure containing pie charts for various statistical data about speakers.
    
    Parameters:
    - df (pandas.DataFrame): DataFrame containing 'StartTime', 'EndTime', 'Speaker', and 'Text' columns.
    
    Returns:
    - matplotlib.figure.Figure: Figure object containing the plotted data.
    """

    # Convert time columns to datetime objects for easier manipulation
    df['StartTime'] = pd.to_datetime(df['StartTime'], format='%H:%M:%S.%f')
    df['EndTime'] = pd.to_datetime(df['EndTime'], format='%H:%M:%S.%f')

    # Calculate duration for each record in seconds
    df['Duration'] = (df['EndTime'] - df['StartTime']).dt.total_seconds()

    # Calculate total duration for normalization purposes
    total_duration = df['Duration'].sum()

    # Compute total duration per speaker and convert to percentage of total
    speaker_durations = df.groupby('Speaker')['Duration'].sum()
    speaker_percentages = (speaker_durations / total_duration) * 100

    # Count the number of turns each speaker has
    speaker_turns = df.groupby('Speaker')['Text'].count()

    # Calculate average duration of turns for each speaker
    speaker_avg_turns = df.groupby('Speaker')['Duration'].mean()
    
    # Find the longest duration of turn per speaker
    speaker_longest_run = df.groupby('Speaker')['Duration'].max()

    # Compile stats into a list for plotting
    stats = [speaker_percentages, speaker_turns, speaker_avg_turns, speaker_longest_run]
    
    # Titles for each subplot
    titles = ['Talking Time for Each Speaker (%)',
                'Number of Turns for Each Speaker',
                'Average Length of Turns for Each Speaker',
                'Longest Run for Each Speaker']
    
    # Create subplot layout with 2 rows and 2 columns
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
    axes = axes.flatten()

    # Use a colormap for consistent coloring across the pies
    cmap = plt.get_cmap('Set2')
    colors = cmap(np.linspace(0, 1, len(speaker_durations)))

    def format(pct, alldata, i):
        """
        Internal function to format the text displayed on each pie chart slice.

        Parameters:
        - pct (float): The percentage of the pie slice.
        - alldata (array-like): Data array of the pie being processed.
        - i (int): Index of the pie chart being processed.

        Returns:
        - str: Formatted string to display on the slice.
        """
        # Only display the label if the percentage is 5% or more
        if pct < 5:
            return ""
        match i:
            case 0:
                return "{:.1f}%".format(pct)
            case 1:
                return str(int(pct / 100. * np.sum(alldata)))
            case _:
                absolute = round(pct/100. * sum(stats[i]), 1)
                return "{:.1f}s".format(absolute)

    # Plot each statistic as a pie chart
    for i in range(len(stats)):
        stats[i].plot(kind='pie',
                        ax=axes[i],
                        labels=[None]*len(stats[i]),  # Disable slice labels
                        autopct=lambda pct: format(pct, stats[i], i),
                        pctdistance=1.25,
                        labeldistance=1.3,
                        radius=.9,
                        colors=colors)
        
        axes[i].set_ylabel('')
        axes[i].set_title(titles[i]) # Set the title for the subplots

    # Create a shared legend at the bottom of the figure
    legend_labels = [f'Speaker {j+1}' for j in range(len(stats[0]))]
    fig.legend(legend_labels, loc='lower center', bbox_to_anchor=(0.5, 0), ncol=len(legend_labels), title="Speakers")

    return fig


# # # Debugging
# from vtt_formatting import format_VTT
# vtt_filename = "data/example_transcripts.vtt"
# # with open(vtt_filename, 'r', encoding='utf-8') as file:
# #     vtt_content = file.read()
# df, formatted_content = format_VTT(vtt_filename)
# create_stats_figure(df)
# plt.show()