import webvtt
import re
import pandas as pd

def format_VTT(file_path):
    # print(file_path)

    with open(file_path) as file:
        vtt_content = file.read()

    # print(vtt_content)
    # Function to format the time stamps
    def format_time_stamp(time_stamp):
        # Match the time stamp pattern
        match = re.match(r"(\d+):(\d+):(\d+)\.(\d+) --> (\d+):(\d+):(\d+)\.(\d+)", time_stamp)
        if match:
            # Extract the individual components of the time stamp
            hours1, minutes1, seconds1, milliseconds1 = map(int, match.groups()[:4])
            hours2, minutes2, seconds2, milliseconds2 = map(int, match.groups()[4:])

            # Format the time stamps
            formatted_time1 = f"{hours1:03d}:{minutes1:02d}:{seconds1:02d}.{milliseconds1:03d}"
            formatted_time2 = f"{hours2:03d}:{minutes2:02d}:{seconds2:02d}.{milliseconds2:03d}"

            # Combine the formatted time stamps
            return f"{formatted_time1} --> {formatted_time2}"
        else:
            return time_stamp

    # Process the .vtt file content
    formatted_content = "\n".join(
        format_time_stamp(line) if "-->" in line else line
        for line in vtt_content.splitlines()
    )


    # Save the formatted content to a new .vtt file
    with open("data/formatted_output.vtt", "w", encoding="utf-8") as file:
        file.write(formatted_content)
        
    formatted_output = 'data/formatted_output.vtt'

    # print()
    # print("Formatted .vtt file saved as '" + formatted_output + "'.")
    # print()

    import webvtt
    start=[]
    end=[]
    text=[]
    speaker=[]
    for caption in webvtt.read(formatted_output):
        start.append(caption.start)
        end.append(caption.end)
        text.append(caption.text)
        speaker.append(caption.raw_text)


    df = pd.DataFrame(list(zip(start,end,text,speaker)), columns = ['StartTime', 'EndTime',"Text","Speaker"])
    listx=df['Speaker'].str.split('>', n=1, expand=True)
    df["Speaker"]=listx[0]
    df["Speaker"]=df["Speaker"].str.replace("<v ","")
    # print(df.head())
    return df, formatted_content
