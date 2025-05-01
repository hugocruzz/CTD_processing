import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from readers import IdronautReader, SeabirdReader, RBRReader, ExoReader
from processors import process_raw_data
from visualize import create_profile_plot
from processors import process_raw_data, CTDFormatter
import shutil  # Add this import at the top of the file


def segment_profiles(pressure_series: pd.Series, prominence: float = 5, distance: int = 10):
    """
    Segment profiles from a pressure series, using local minima as delimiters.

    Args:
        pressure_series (pd.Series): The pressure data.
        prominence (float): Minimum prominence of peaks to consider.
        distance (int): Minimum distance between peaks.

    Returns:
        List[tuple]: List of (start_index, end_index) for each profile.
    """
    # Smooth the pressure data to reduce noise
    smoothed_pressure = savgol_filter(pressure_series, window_length=21, polyorder=2)

    # Find local minima (valleys)
    minima_indices, _ = find_peaks(-smoothed_pressure, prominence=prominence, distance=distance)

    # Add the start and end of the dataset as delimiters
    delimiters = np.concatenate(([0], minima_indices, [len(pressure_series) - 1]))

    # Create profiles based on the delimiters
    profiles = [(delimiters[i], delimiters[i + 1]) for i in range(len(delimiters) - 1)]

    return profiles

def find_logbook(data_dir: str) -> str:
    """
    Search for a file named 'logbook.csv' in all subfolders of the given directory.

    Args:
        data_dir (str): The base directory to search in.

    Returns:
        str: The full path to the logbook file if found.

    Raises:
        FileNotFoundError: If no logbook file is found.
    """
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower() == "logbook.csv":
                return os.path.join(root, file)
    raise FileNotFoundError("No 'logbook.csv' file found in the directory or its subfolders.")

def process_ctd_file(filepath, ctd_type, data_dir, Level1_output, Level2_output, Level2B_output, logbook_path=None):
    """Process a single CTD file and handle multiple profiles, respecting subfolder structure."""
    # If logbook_path is not provided, find it
    if logbook_path is None:
        logbook_path = find_logbook(data_dir)

    # Read the logbook
    logbook = pd.read_csv(logbook_path, sep=';', encoding='utf-8')

    readers = {
        'idronaut': IdronautReader,
        'seabird': SeabirdReader,
        'rbr': RBRReader,
        'exo': ExoReader
    }
    
    reader = readers[ctd_type](filepath, ctd_type)
    df = reader.read()

    # Segment profiles based on pressure
    if 'pressure_dbar' not in df.columns:
        raise ValueError(f"Pressure column not found in file: {filepath}")
    
    profiles = segment_profiles(df['pressure_dbar'], prominence=10, distance=50)
    if not profiles:
        print(f"No valid profiles found in file: {filepath}")
        return

    # Ensure the Level1 output folder exists
    os.makedirs(Level1_output, exist_ok=True)

    # Export profiles to Level1 folder
    for i, (start, end) in enumerate(profiles):
        profile_df = df.iloc[start:end + 1].copy()
        numeric_cols = profile_df.select_dtypes(include=['number', 'float', 'int']).columns
        profile_df[numeric_cols] = profile_df[numeric_cols].apply(pd.to_numeric, errors='coerce')

        try:
            base_name = profile_df["timestamp"].iloc[0].strftime("%Y%m%d_%H%M%S")
        except:
            base_name = os.path.splitext(os.path.basename(filepath))[0]
        profile_outfile = os.path.join(Level1_output, f"{base_name}_{i + 1}.csv")
        profile_df.to_csv(profile_outfile, index=False)
        print(f"Saved profile {i + 1} to {profile_outfile}")

    # Process the data and save to Level2 folder
    df = process_raw_data(df, ctd_type)
    os.makedirs(Level2_output, exist_ok=True)
    processed_outfile = os.path.join(Level2_output, os.path.basename(filepath))
    df.to_csv(processed_outfile, index=False)
    print(f"Processed data saved to {processed_outfile}")

    # Match profiles with logbook entries and copy to Level2B folder
    os.makedirs(Level2B_output, exist_ok=True)
    for i, (start, end) in enumerate(profiles):
        profile_outfile = os.path.join(Level1_output, f"{base_name}_{i + 1}.csv")
        matched_row = logbook[logbook['CTD file'] == os.path.basename(profile_outfile)]
        if not matched_row.empty:
            station = matched_row.iloc[0]['Station']
            station_folder = os.path.join(Level2B_output, station)
            os.makedirs(station_folder, exist_ok=True)

            # Copy the profile to the station folder
            station_outfile = os.path.join(station_folder, os.path.basename(profile_outfile))
            if os.path.exists(station_outfile):
                print(f"File already exists at destination: {station_outfile}. Overwriting.")
                os.remove(station_outfile)  # Remove the existing file if overwriting is desired
            shutil.copy(profile_outfile, station_outfile)
            print(f"Copied profile {profile_outfile} to {station_outfile}")

def get_ctd_type(filename: str) -> str:
    """
    Determine CTD type from filename.
    
    Args:
        filename: Name of the CTD file
        
    Returns:
        str: 'seabird', 'idronaut', or 'rbr'
        
    Raises:
        ValueError: If CTD type cannot be determined
    """
    # Get file extension
    extension = os.path.splitext(filename)[1].lower()
    
    if extension == '.cnv':
        return 'seabird'
    elif extension == '.txt' and 'idronaut' in filename.lower():
        return 'idronaut'
    elif extension == '.txt' and ('_data' in filename.lower() and ('rbr' in os.path.dirname(filename).lower())):
        return 'rbr'
    elif extension == ".csv" and "kor" in filename.lower():
        return 'exo'
    elif extension == '.txt' and ('suboceanexperiment' in filename.lower()):
        print("this is a subocean file, skipping")
    else:
        return None
    
def process_all_files(directory: str, ship: str, Level1_output, Level2_output, Level2B_output, logbook_path=None) -> None:
    """Process all CTD files in directory."""
    print(f"Replicating directory structure from {directory} to {Level1_output}")
    
    all_files = []
    for ext in ['.cnv', '.txt', '.csv']:
        search_pattern = os.path.join(directory, f"**/*{ext}")
        all_files.extend(glob.glob(search_pattern, recursive=True))
    
    if not all_files:
        print(f"No CTD files found in {directory}")
        return
    
    for file in all_files:
        ctd_type = get_ctd_type(file)
        if ctd_type is None:
            print(f"Could not determine CTD type for {file}")
            continue
        print(f"Processing {file} as {ctd_type}")
        process_ctd_file(file, ctd_type, directory, Level1_output, Level2_output, Level2B_output, logbook_path)

if __name__ == "__main__":
    campaign = "LacNOX/20250408_Lexplore_spatial"
    data_dir = fr"C:\Users\cruz\Documents\SENSE\SubOcean\data\raw\{campaign}"
    Level1_output = os.path.join("data", "Level1", campaign) 
    Level2_output = os.path.join("data", "Level2", campaign)
    Level2B_output = os.path.join("data", "Level2B", campaign)
    
    # If logbook_path is not provided, it will be searched automatically
    logbook_path = None  # Set to None to trigger automatic search
    
    process_all_files(data_dir, campaign, Level1_output, Level2_output, Level2B_output, logbook_path)