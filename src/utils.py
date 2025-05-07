import os
import glob
import pandas as pd
import shutil
from typing import List, Tuple, Dict, Optional, Union

def find_files_by_extension(directory: str, extensions: List[str], recursive: bool = True) -> List[str]:
    """
    Find files with specific extensions in a directory.
    
    Args:
        directory: Base directory to search
        extensions: List of file extensions to look for (e.g., ['.txt', '.csv'])
        recursive: Whether to search in subdirectories
        
    Returns:
        List of file paths matching the criteria
    """
    all_files = []
    if recursive:
        for root, _, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    all_files.append(os.path.join(root, file))
    else:
        for ext in extensions:
            search_pattern = os.path.join(directory, f"*{ext}")
            all_files.extend(glob.glob(search_pattern))
    
    return all_files

def save_profile(df: pd.DataFrame, output_dir: str, filename: str) -> str:
    """
    Save a profile DataFrame to the specified output directory.
    
    Args:
        df: DataFrame containing profile data
        output_dir: Directory where the file should be saved
        filename: Name of the output file
        
    Returns:
        Path to the saved file
    """
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, filename)
    #if output_file have no extension, add .csv
    if not output_file.endswith('.csv'):
        output_file += '.csv'
        
    df.to_csv(output_file, index=False)
    return output_file

def match_profile_with_logbook(profile_filename: str, 
                              logbook: pd.DataFrame, 
                              output_dir: str,
                              profile_source_path: str) -> Optional[str]:
    """
    Match a profile with a logbook entry and copy it to the appropriate output directory.
    
    Args:
        profile_filename: Name of the profile file
        logbook: DataFrame containing logbook entries
        output_dir: Base output directory
        profile_source_path: Source path of the profile file to copy
        
    Returns:
        Path to the copied file or None if no match found
    """
    if logbook is None:
        return None
        
    matched_row = logbook[logbook['CTD file'] == profile_filename]
    if not matched_row.empty:
        # Determine the output structure
        campaign = matched_row.iloc[0]['campaign'] if "campaign" in matched_row.columns else ""
        station = matched_row.iloc[0]['Station']
        suffix = f"{campaign}/{station}/" if campaign else f"{station}/"
        
        # Create output directory
        station_folder = os.path.join(output_dir, suffix)
        os.makedirs(station_folder, exist_ok=True)
        
        # Copy the file
        station_outfile = os.path.join(station_folder, profile_filename)
        if os.path.exists(station_outfile):
            os.remove(station_outfile)
        shutil.copy(profile_source_path, station_outfile)
        return station_outfile
    
    return None

def generate_profile_filename(profile_df: pd.DataFrame, source_filepath: str, index: int) -> str:
    """
    Generate a consistent filename for a profile.
    
    Args:
        profile_df: DataFrame containing profile data
        source_filepath: Original source file path
        index: Profile index number
        
    Returns:
        Generated filename
    """
    try:
        base_name = profile_df["timestamp"].iloc[0].strftime("%Y%m%d_%H%M%S")
    except:
        base_name = os.path.splitext(os.path.basename(source_filepath))[0]
    
    return f"{base_name}_{index + 1}.csv"

def extract_profiles_from_data(df: pd.DataFrame, source_filepath: str) -> List[Tuple[pd.DataFrame, str]]:
    """
    Extract profiles from data and create filenames.
    
    Args:
        df: DataFrame containing CTD data
        source_filepath: Original source file path
        
    Returns:
        List of tuples (profile_df, profile_filename)
    """
    if 'pressure_dbar' not in df.columns:
        raise ValueError(f"Pressure column not found in file: {source_filepath}")
    
    from main import segment_profiles  # Import locally to avoid circular imports
    
    profiles = segment_profiles(df['pressure_dbar'], prominence=10, distance=50)
    if not profiles:
        return []
        
    result = []
    for i, (start, end) in enumerate(profiles):
        # Extract profile segment
        profile_df = df.iloc[start:end + 1].copy()
        numeric_cols = profile_df.select_dtypes(include=['number', 'float', 'int']).columns
        profile_df[numeric_cols] = profile_df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Generate filename
        profile_filename = generate_profile_filename(profile_df, source_filepath, i)
        
        result.append((profile_df, profile_filename))
        
    return result