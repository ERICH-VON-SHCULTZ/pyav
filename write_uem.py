import os
from pathlib import Path
import wave

def process_files_in_directory(input_dir, output_dir, wav_dir):
    """
    Reads all files from an input directory, applies a modification,
    and writes the modified files to a new output directory.

    Args:
        input_dir (str): The path to the directory containing the input files.
        output_dir (str): The path to the directory where modified files will be saved.
    """

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):

        if "c_01" in filename:
            offset = 0.
        elif "c_02" in filename:
            offset = 0.
        elif "c_03" in filename:
            offset = 0.
        else:
            print("Bad filename " + filename)
            exit()
    
            
        input_filepath = Path(os.path.join(input_dir, filename))

        # Check if it's a file (and not a subdirectory)
        if os.path.isfile(input_filepath):

            output_filepath = os.path.join(output_dir, input_filepath.stem + ".uem")
            with open(output_filepath, 'w') as outfile:
                wav_filepath = os.path.join(wav_dir, input_filepath.stem + ".wav")

                with wave.open(wav_filepath, 'r') as wf:
                    num_frames = wf.getnframes()
                    frame_rate = wf.getframerate()
                    duration = num_frames / float(frame_rate) - .01
                
                outfile.write(f"{input_filepath.stem} NA 0. {duration:.2f}")


# Example Usage:
if __name__ == "__main__":
    source_directory = "/scratch/map22-share/AVA-AVD/AVA-AVD/dataset/rttms"
    destination_directory = "/scratch/map22-share/AVA-AVD/AVA-AVD/dataset/uems"
    wav_directory = "/scratch/map22-share/AVA-AVD/AVA-AVD/dataset/waves"

    process_files_in_directory(source_directory, destination_directory, wav_directory)
