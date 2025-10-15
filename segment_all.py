from moviepy import VideoFileClip
import os
import pandas as pd
from pathlib import Path

def read_file_to_dictionary(filepath, delimiter=' '):
    """
    Reads a file with two fields per line into a dictionary.

    Args:
        filepath (str): The path to the file.
        delimiter (str): The character(s) used to separate the fields in each line.

    Returns:
        dict: A dictionary where the first field of each line is the key
              and the second field is the value.
    """
    result_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            # Remove leading/trailing whitespace and split the line by the delimiter
            parts = line.strip().split(delimiter, 1) 
            # Ensure there are exactly two parts
            if parts[1] == "duration":
                continue
            if len(parts) == 2:
                key = Path(parts[0]).stem
                value = float(parts[1])
                result_dict[key] = value
            else:
                # Optionally handle lines that don't conform to the expected format
                print(f"Skipping malformed line: {line.strip()}")
    return result_dict

def extract_video_segments(input_dir, output_dir, offset_file, duration_file):
    """
    Extracts 300-second segments from all MKV files in the input directory
    and saves them to the output directory.

    Args:
        input_dir (str): Path to the directory containing the input MP4 files.
        output_dir (str): Path to the directory where the segmented clips will be saved.
        segment_duration (int, optional): The duration of each segment in seconds. Defaults to 300.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read in the offset file 
    offset_dict = read_file_to_dictionary(offset_file)

    # Read in the segment durations
    duration_dict = read_file_to_dictionary(duration_file, delimiter=",")
    
    
    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        print(f"****Video file: {filename} original****")
        if filename.endswith(".mp4") or filename.endswith(".mkv") or filename.endswith(".webm"):
            input_filepath = os.path.join(input_dir, filename)
            file_stem = os.path.splitext(filename)[0]
            print(f"****Video file: {input_filepath} Stem: {file_stem}")
            try:
                with VideoFileClip(input_filepath) as video:
                    # Define segment start times

                    segment_labels = ["_c_01", "_c_02", "_c_03"]
               
                    for i, segment_label in enumerate(segment_labels):
                        # Ensure the video is long enough for the segment
                        file_key = file_stem + segment_label
                        if not file_key in offset_dict:
                            print(f"{file_key} not found in list of offsets")
                            exit() 
                        if not file_key in duration_dict:
                            print(f"{file_key} not found in list of durations")
                            exit()
                        start_time = offset_dict[file_key]
                        segment_duration = duration_dict[file_key]
                        if start_time + segment_duration <= video.duration:
                            # Extract the subclip
                            subclip = video.subclipped(start_time, start_time + segment_duration)

                            # Construct the output filename
                            output_filename = f"{file_stem}{segment_labels[i]}.mp4" 
                            output_filepath = os.path.join(output_dir, output_filename)

                            # Write the subclip to the output file
                            subclip.write_videofile(output_filepath, codec="libx264", audio_codec="aac")
                            print(f"Segment extracted and saved: {output_filepath} {start_time} {segment_duration}")
                        else:
                            print(f"Skipping segment {i+1} for {filename}: Video is too short for the requested start time.")
                            exit()
                            
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Example usage
if __name__ == "__main__":
    
    input_directory = "/scratch/map22-share/AVA-AVD/AVA-AVD/dataset/videos"
    output_directory = "/scratch/map22-share/AVA-AVD/AVA-AVD/dataset/videos_new2"
    offset_file = "/scratch/map22-share/AVA-AVD/AVA-AVD/dataset/offsets"
    duration_file = "/scratch/map22-share/pyav/allwaves.txt.durations"

    extract_video_segments(input_directory, output_directory, offset_file, duration_file)
