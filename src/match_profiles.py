import os
import pandas as pd
import shutil
import glob
from typing import List, Dict, Optional, Union

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

def load_logbook(logbook_path: str) -> pd.DataFrame:
    """
    Load the logbook file.
    
    Args:
        logbook_path: Path to the logbook file
        
    Returns:
        DataFrame containing logbook data
    """
    return pd.read_csv(logbook_path, sep=';', encoding='utf-8')

def find_all_profiles(level2_dir: str) -> List[str]:
    """
    Find all profile files in the Level2 directory.
    
    Args:
        level2_dir: Directory containing Level2 profiles
        
    Returns:
        List of profile file paths
    """
    all_profiles = []
    for root, dirs, files in os.walk(level2_dir):
        for file in files:
            if file.endswith('.csv'):
                all_profiles.append(os.path.join(root, file))
    return all_profiles

def find_file_in_directory(directory: str, filename: str) -> Optional[str]:
    """
    Find a file in a directory tree by name, handling partial matches.
    
    Args:
        directory: Directory to search in
        filename: Filename to find (can be partial)
        
    Returns:
        Full path to the file if found, None otherwise
    """
    for root, dirs, files in os.walk(directory):
        # Try exact match first
        if filename in files:
            return os.path.join(root, filename)
        
        # Try case-insensitive match
        matches = [f for f in files if f.lower() == filename.lower()]
        if matches:
            return os.path.join(root, matches[0])
            
        # Try partial match (filename is contained in actual filename)
        matches = [f for f in files if filename in f]
        if matches:
            return os.path.join(root, matches[0])
    
    return None

def find_related_files(raw_data_dir: str, logbook_entry: pd.Series) -> Dict[str, str]:
    """
    Find all related files mentioned in the logbook entry.
    
    Args:
        raw_data_dir: Base directory for raw data
        logbook_entry: A row from the logbook
        
    Returns:
        Dictionary mapping file type to file path
    """
    related_files = {}
    
    # Look for SubOcean files
    if 'Subocean File' in logbook_entry and pd.notna(logbook_entry['Subocean File']):
        subocean_file = logbook_entry['Subocean File']
        file_path = find_file_in_directory(raw_data_dir, subocean_file)
        if file_path:
            related_files['subocean_log'] = file_path.replace(".txt", ".log")
            related_files["subocean"] = file_path.replace(".log", ".txt")
    
    # Look for Exo files
    for exo_column in ['Exo file', 'Exo file 2']:
        if exo_column in logbook_entry and pd.notna(logbook_entry[exo_column]):
            exo_file = logbook_entry[exo_column]
            file_path = find_file_in_directory(raw_data_dir, exo_file)
            if file_path:
                related_files[exo_column.lower().replace(' ', '_')] = file_path
    
    return related_files

def create_station_folder(level2b_dir: str, station: str, campaign: str = '') -> str:
    """
    Create station folder in Level2B directory.
    
    Args:
        level2b_dir: Base directory for Level2B output
        station: Station identifier
        campaign: Campaign name (optional)
        
    Returns:
        Path to the created folder
    """
    if campaign:
        station_folder = os.path.join(level2b_dir, campaign, station)
    else:
        station_folder = os.path.join(level2b_dir, station)
    
    os.makedirs(station_folder, exist_ok=True)
    return station_folder

def copy_file_to_station(source_path: str, station_folder: str) -> str:
    """
    Copy a file to a station folder, overwriting if necessary.
    
    Args:
        source_path: Path to source file
        station_folder: Destination folder
        
    Returns:
        Path to the copied file
    """
    filename = os.path.basename(source_path)
    target_path = os.path.join(station_folder, filename)
    
    if os.path.exists(target_path):
        print(f"File already exists at destination: {target_path}. Overwriting.")
        os.remove(target_path)
    
    shutil.copy(source_path, target_path)
    print(f"Copied file to {target_path}")
    
    return target_path

def match_profile_with_logbook_entry(profile_path: str, ctd_file: str) -> bool:
    """
    Check if a profile matches a CTD file entry in the logbook.
    
    Args:
        profile_path: Path to the profile file
        ctd_file: CTD file name from logbook
        
    Returns:
        True if the profile matches the CTD file, False otherwise
    """
    profile_name = os.path.basename(profile_path)
    
    # Try exact match
    if profile_name == ctd_file:
        return True
    
    # Try match with profile index (e.g., 20250409_084749_2.csv matches 20250409_084749_1.csv)
    base_profile_name = '_'.join(profile_name.split('_')[:-1])
    if base_profile_name and ctd_file.startswith(base_profile_name):
        return True
    
    return False

def match_and_copy_profiles(level2_dir: str, raw_data_dir: str, level2b_dir: str, logbook_path: str) -> None:
    """
    Match Level2 profiles with logbook entries and copy them to Level2B.
    
    Args:
        level2_dir: Directory containing Level2 profiles
        raw_data_dir: Directory containing raw data files
        level2b_dir: Output directory for Level2B
        logbook_path: Path to the logbook file
    """
    # Load logbook
    try:
        logbook = load_logbook(logbook_path)
    except Exception as e:
        print(f"Error loading logbook: {e}")
        return
    
    # Find all profile files
    all_profiles = find_all_profiles(level2_dir)
    if not all_profiles:
        print(f"No profiles found in {level2_dir}")
        return
    
    print(f"Found {len(all_profiles)} profiles in Level2")
    
    # Track matched profiles
    matched_profiles = set()
    
    # Process each logbook entry
    for _, entry in logbook.iterrows():
        station = entry['Station']
        
        # Skip entries without a CTD file
        if 'CTD file' not in entry or pd.isna(entry['CTD file']):
            print(f"Warning: No CTD file specified for Station {station}")
            continue
            
        ctd_file = entry['CTD file']
        campaign = entry.get('campaign', '')
        
        # Create station folder
        station_folder = create_station_folder(level2b_dir, station, campaign)
        
        # Find matching profiles
        matched = False
        for profile_path in all_profiles:
            if match_profile_with_logbook_entry(profile_path, ctd_file):
                # Copy profile to Level2B
                copy_file_to_station(profile_path, station_folder)
                matched = True
                matched_profiles.add(profile_path)
                
                # Find and copy related files
                related_files = find_related_files(raw_data_dir, entry)
                for file_type, file_path in related_files.items():
                    copy_file_to_station(file_path, station_folder)
                    print(f"Copied {file_type} file for station {station}")
                
                break
        
        if not matched:
            print(f"Warning: No matching profile found for {ctd_file} (Station {station})")
    
    # Report unmatched profiles
    unmatched = [p for p in all_profiles if p not in matched_profiles]
    if unmatched:
        print(f"\n{len(unmatched)} profiles were not matched with any logbook entry:")
        for p in unmatched:
            print(f"  - {os.path.basename(p)}")
    
    print(f"\nMatched {len(matched_profiles)} out of {len(all_profiles)} profiles")
    print("Profile matching and restructuring complete.")

if __name__ == "__main__":
    # Set paths
    campaign = "Forel/"  # Update this as needed
    data_dir = fr"C:\Users\cruz\Documents\SENSE\SubOcean\data\raw\{campaign}"
    level2_dir = os.path.join("data", "Level2", campaign)
    level2b_dir = os.path.join("data", "Level2B", campaign)
    
    # Find logbook
    try:
        logbook_path = find_logbook(data_dir)
        print(f"Found logbook at: {logbook_path}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    
    # Match profiles with logbook and organize into Level2B
    match_and_copy_profiles(level2_dir, data_dir, level2b_dir, logbook_path)