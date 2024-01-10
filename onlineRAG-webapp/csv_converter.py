import os
import pandas as pd

def excel_to_csv(input_folder, output_folder):
    try:
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # Iterate through each file in the input folder
        for file_name in os.listdir(input_folder):
            if file_name.endswith(".xlsx") or file_name.endswith(".xls"):
                # Construct full paths for input and output files
                excel_file_path = os.path.join(input_folder, file_name)
                csv_file_path = os.path.join(output_folder, file_name.replace(".xlsx", ".csv").replace(".xls", ".csv"))

                # Read Excel file and write to CSV
                df = pd.read_excel(excel_file_path)
                df.to_csv(csv_file_path, index=False)

                print(f"Conversion successful. {file_name} converted to {csv_file_path}")

    except Exception as e:
        print(f"Error: {e}")

# Example usage:
input_folder_path = 'testing_files'
output_folder_path = 'testing_files'
excel_to_csv(input_folder_path, output_folder_path)
