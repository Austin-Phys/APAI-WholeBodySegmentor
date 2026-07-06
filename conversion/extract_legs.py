import os
import zipfile

# Define the root directory containing the patient folders
ROOT_DIR = r"C:\Research\ASHA\leg_data"

def extract_zips_in_folders(base_path):
    # Verify the root directory exists
    if not os.path.exists(base_path):
        print(f"Error: The path {base_path} does not exist.")
        return

    # Loop through everything inside the root directory
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        
        # Check if the item is a directory (e.g., P001, P002)
        if os.path.isdir(item_path):
            print(f"Checking folder: {item}")
            
            # Look for zip files inside this subfolder
            for file in os.listdir(item_path):
                if file.lower().endswith('.zip'):
                    zip_path = os.path.join(item_path, file)
                    
                    # Create a folder name for the extraction target (optional, prevents clutter)
                    # This extracts into a folder named after the zip file, inside the P00X folder
                    extract_target_dir = os.path.join(item_path, os.path.splitext(file)[0])
                    os.makedirs(extract_target_dir, exist_ok=True)
                    
                    print(f"  Extracting {file} -> {extract_target_dir}")
                    
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall(extract_target_dir)
                    except zipfile.BadZipFile:
                        print(f"  [ERROR] {file} is corrupted or not a valid zip file.")
                    except Exception as e:
                        print(f"  [ERROR] Failed to extract {file}: {e}")

if __name__ == "__main__":
    extract_zips_in_folders(ROOT_DIR)
