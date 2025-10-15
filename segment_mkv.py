from moviepy import VideoFileClip
import os

def extract_video_segments(input_dir, output_dir, segment_duration=300):
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

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".mkv"):
            input_filepath = os.path.join(input_dir, filename)
            file_stem = os.path.splitext(filename)[0]

            try:
                with VideoFileClip(input_filepath) as video:
                    # Define segment start times
                    start_times = [900, 1200, 1500] 
                    segment_labels = ["_c_01", "_c_02", "_c_03"]

                    for i, start_time in enumerate(start_times):
                        # Ensure the video is long enough for the segment
                        if start_time + segment_duration <= video.duration:
                            # Extract the subclip
                            subclip = video.subclipped(start_time, start_time + segment_duration)

                            # Construct the output filename
                            output_filename = f"{file_stem}{segment_labels[i]}.mp4" 
                            output_filepath = os.path.join(output_dir, output_filename)

                            # Write the subclip to the output file
                            subclip.write_videofile(output_filepath, codec="libx264", audio_codec="aac")
                            print(f"Segment extracted and saved: {output_filepath}")
                        else:
                            print(f"Skipping segment {i+1} for {filename}: Video is too short for the requested start time.")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Example usage
if __name__ == "__main__":
    
    input_directory = "/scratch/map22-share/AVA-AVD/AVA-AVD/dataset/videos"
    output_directory = "/scratch/map22-share/AVA-AVD/AVA-AVD/dataset/videos_new"

    extract_video_segments(input_directory, output_directory)
