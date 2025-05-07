import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from readers import IdronautReader, SeabirdReader, RBRReader, ExoReader, GF23Reader
from processors import process_raw_data
from visualize import create_profile_plot
from processors import process_raw_data, CTDFormatter
import shutil
from utils import (find_files_by_extension, save_profile, 
                  generate_profile_filename, extract_profiles_from_data)
from datetime import datetime


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

def load_logbook(data_dir: str, logbook_path: str = None) -> pd.DataFrame:
    """
    Load the logbook file.
    
    Args:
        data_dir: Base directory to search if logbook_path is None
        logbook_path: Path to the logbook file, if known
        
    Returns:
        DataFrame containing logbook data or None if not found
    """
    try:
        if logbook_path is None:
            logbook_path = find_logbook(data_dir)
        return pd.read_csv(logbook_path, sep=';', encoding='utf-8')
    except FileNotFoundError:
        print("No logbook.csv found. Processing data without station restructuring.")
        return None

def get_reader(filepath, ctd_type):
    """
    Initialize the appropriate reader for the given CTD type.

    Args:
        filepath (str): Path to the CTD file.
        ctd_type (str): Type of CTD ('idronaut', 'seabird', 'rbr', 'exo', 'GF23').

    Returns:
        BaseReader: An instance of the appropriate reader class.
    """
    readers = {
        'idronaut': IdronautReader,
        'seabird': SeabirdReader,
        'rbr': RBRReader,
        'exo': ExoReader,
        'GF23': GF23Reader
    }

    if ctd_type not in readers:
        raise ValueError(f"Unsupported CTD type: {ctd_type}")

    return readers[ctd_type](filepath, ctd_type)

def extract_date_from_data(filepath, ctd_type):
    """Extract date from the time column in the dataframe."""
    try:
        reader = get_reader(filepath, ctd_type)
        df = reader.read()

        # Check for timestamp or date columns in the dataframe
        if 'datetime' in df.columns:
            return pd.to_datetime(df['datetime'].iloc[0]).strftime('%Y-%m-%d')
        elif 'date' in df.columns:
            return pd.to_datetime(df['date'].iloc[0]).strftime('%Y-%m-%d')
        else:
            print(f"No date/time column found in {filepath}, using file properties")
            # Fall back to file modification time
            file_time = os.path.getmtime(filepath)
            return datetime.fromtimestamp(file_time).strftime('%Y-%m-%d')

    except Exception as e:
        print(f"Error extracting date from {filepath}: {e}")
        # If all fails, use today's date
        return datetime.now().strftime('%Y-%m-%d')

def process_ctd_file(filepath, ctd_type, data_dir, Level1_output, Level2_output, Level2B_output, processing_mode=None):
    """Process a single CTD file and handle multiple profiles, respecting subfolder structure."""

    reader = get_reader(filepath, ctd_type)
    df = reader.read()

    # Extract profiles based on processing mode
    if processing_mode == "segment":
        profiles_data = extract_profiles_from_data(df, filepath)
    else:
        # For None or "concatenate", treat entire file as one profile
        profile_filename = os.path.splitext(os.path.basename(filepath))[0]
        profiles_data = [(df, profile_filename)]

    if not profiles_data:
        print(f"No valid profiles found in file: {filepath}")
        return df, filepath  # Return the dataframe for potential concatenation

    # Process each profile
    for i, (profile_df, profile_filename) in enumerate(profiles_data):
        # Save raw profile to Level1
        level1_path = save_profile(profile_df, Level1_output, profile_filename)
        print(f"Saved profile {i + 1} to {level1_path}")

        # Process and save to Level2
        processed_df = process_raw_data(profile_df, ctd_type)
        level2_path = save_profile(processed_df, Level2_output, profile_filename)
        print(f"Processed profile {i + 1} saved to {level2_path}")

    return df, filepath  # Return the dataframe for potential concatenation


def get_ctd_type(filename: str) -> str:
    """
    Determine CTD type from filename.
    
    Args:
        filename: Name of the CTD file
        
    Returns:
        str: 'seabird', 'idronaut', 'rbr', or 'exo'
        
    Returns None if type cannot be determined
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
    elif extension == ".txt" and ".TXT" in os.path.basename(filename):
        return 'GF23'
    elif extension == '.txt' and ('suboceanexperiment' in filename.lower()):
        print("this is a subocean file, skipping")
    else:
        return None
    
def process_all_files(directory: str, Level1_output, Level2_output, Level2B_output, processing_mode=None) -> None:
    """
    Process all CTD files in directory.
    
    Args:
        directory: Directory containing CTD files
        Level1_output: Output directory for Level1 data
        Level2_output: Output directory for Level2 data
        Level2B_output: Output directory for Level2B data
        processing_mode: "concatenate", "segment", or None (default processing)
    """
    print(f"Replicating directory structure from {directory} to {Level1_output}")

    # Find all CTD files
    all_files = find_files_by_extension(directory, ['.cnv', '.txt', '.csv'], recursive=True)

    if not all_files:
        print(f"No CTD files found in {directory}")
        return

    if processing_mode == "concatenate":
        # Group files by date and CTD type
        grouped_files = {}

        for file in all_files:
            ctd_type = get_ctd_type(file)
            if ctd_type is None:
                print(f"Could not determine CTD type for {file}")
                continue

            # Extract date from dataframe instead of filepath
            date_str = extract_date_from_data(file, ctd_type)
            key = (date_str, ctd_type)

            if key not in grouped_files:
                grouped_files[key] = []
            grouped_files[key].append(file)

        # Process each group
        for (date_str, ctd_type), files in grouped_files.items():
            if len(files) == 0:
                continue

            print(f"Concatenating {len(files)} {ctd_type} files for {date_str}")

            # Read and concatenate data
            dfs = []
            for file in files:
                reader = get_reader(file, ctd_type)
                df = reader.read()
                dfs.append(df)

            if not dfs:
                print(f"No valid data for {date_str}, {ctd_type}")
                continue

            # Concatenate data
            concatenated_df = pd.concat(dfs, ignore_index=True)

            # Sort by time or depth to ensure proper ordering
            if 'datetime' in concatenated_df.columns:
                concatenated_df.sort_values('datetime', inplace=True)
            elif 'depth_m' in concatenated_df.columns:
                concatenated_df.sort_values('depth_m', inplace=True)

            # Generate filename for the concatenated data
            concat_filename = f"{date_str}_{ctd_type}_concatenated.csv"

            # Save concatenated raw data to Level1
            level1_path = save_profile(concatenated_df, Level1_output, concat_filename)
            print(f"Saved concatenated profile to {level1_path}")

            # Process and save concatenated data to Level2
            processed_df = process_raw_data(concatenated_df, ctd_type)
            level2_path = save_profile(processed_df, Level2_output, concat_filename)
            print(f"Processed concatenated profile saved to {level2_path}")
    else:
        # Process files individually
        for file in all_files:
            ctd_type = get_ctd_type(file)
            if ctd_type is None:
                print(f"Could not determine CTD type for {file}")
                continue
            print(f"Processing {file} as {ctd_type}")
            process_ctd_file(file, ctd_type, directory, Level1_output, Level2_output, Level2B_output, processing_mode)

if __name__ == "__main__":
    #campaign = "LacNOX/"
    #campaign = "Forel/"
    #campaign = "Greenfjord 2023\casts"
    campaign  =  "Forel"
    campaign  =  "Sanna"
    #data_dir = fr"C:\Users\cruz\Documents\SENSE\SubOcean\data\raw\{campaign}"
    data_dir = fr"C:\Users\cruz\Documents\SENSE\CTD_processing\data\Level0\{campaign}"
    Level1_output = os.path.join("data", "Level1", campaign) 
    Level2_output = os.path.join("data", "Level2", campaign)
    Level2B_output = os.path.join("data", "Level2B", campaign)
    
    # Set processing_mode to:
    # - "concatenate": Combine all profiles by day and CTD type
    # - "segment": Extract multiple profiles from each file (as before)
    # - None: Process each file as a single profile
    processing_mode = "concatenate"  # Change as needed
    processing_mode=  None
    process_all_files(data_dir, Level1_output, Level2_output, Level2B_output, processing_mode)
    
    print("\nProfile processing complete!")
    print("To organize profiles by station, run match_profiles.py separately.")