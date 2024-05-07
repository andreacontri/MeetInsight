import webvtt
import re
import pandas as pd

def format_VTT(file_path):
    """
    Read and process a VTT file to format its timestamps and extract data into a DataFrame.
    This function also saves a new VTT file with formatted timestamps.

    Args:
        file_path (str): The path to the VTT file that needs to be formatted and processed.

    Returns:
        tuple: A tuple containing a DataFrame with the VTT data and the formatted VTT content as a string.
    """
    # Open the VTT file and read its content
    with open(file_path, 'r', encoding='utf-8') as file:
        vtt_content = file.read()

    # Helper function to format timestamps in VTT entries
    def format_time_stamp(time_stamp):
        match = re.match(r"(\d+):(\d+):(\d+)\.(\d+) --> (\d+):(\d+):(\d+)\.(\d+)", time_stamp)
        if match:
            parts = list(map(int, match.groups()))
            formatted_parts = "{:02}:{:02}:{:02}.{:03d} --> {:02}:{:02}:{:02}.{:03d}".format(*parts)
            return formatted_parts
        return time_stamp

    # Format the content of the VTT file by updating its timestamps
    formatted_content = "\n".join(
        format_time_stamp(line) if '-->' in line else line
        for line in vtt_content.splitlines()
    )

    # Write the formatted content to a new file
    formatted_output = 'formatted_output.vtt'
    with open(formatted_output, "w", encoding="utf-8") as file:
        file.write(formatted_content)

    # Read the formatted VTT file using webvtt-py to parse its contents
    captions = webvtt.read(formatted_output)
    data = {
        'StartTime': [cap.start for cap in captions],
        'EndTime': [cap.end for cap in captions],
        'Text': [cap.text for cap in captions],
        'Speaker': [cap.raw_text.split(' ')[0].strip('<v>') for cap in captions]
    }

    # Create a DataFrame from the parsed data
    df = pd.DataFrame(data)

    return df, formatted_content
