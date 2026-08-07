import os
import shutil

def make_clear_folder(folder_path):
    """
    Create a folder if it does not exist, or clear it if it does.
    
    Args:
        folder_path (str): Path to the folder to create or clear.
    """
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist