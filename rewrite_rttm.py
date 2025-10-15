import os

def process_files_in_directory(input_dir, output_dir):
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
            offset = 900.
        elif "c_02" in filename:
            offset = 1200.
        elif "c_03" in filename:
            offset = 1500.
        else:
            print("Bad filename " + filename)
            exit()
    
            
        input_filepath = os.path.join(input_dir, filename)

        # Check if it's a file (and not a subdirectory)
        if os.path.isfile(input_filepath):
            with open(input_filepath, 'r') as infile:
                content = infile.readlines()

            modified_content = []
            for line in content:
                linesplit = line.strip().split()
                linesplit[3] = f"{float(linesplit[3]) - offset:.6f}"
                modified_content.append(" ".join(linesplit) + "\n")
                
            output_filepath = os.path.join(output_dir, filename)
            with open(output_filepath, 'w') as outfile:
                outfile.writelines(modified_content)

            print(f"Processed '{filename}' and saved to '{output_filepath}'")

# Example Usage:
if __name__ == "__main__":
    source_directory = "/scratch/map22-share/AVA-AVD/AVA-AVD/dataset/rttms"
    destination_directory = "/scratch/map22-share/AVA-AVD/AVA-AVD/dataset/rttms_new"

    process_files_in_directory(source_directory, destination_directory)
