import os
import glob
import json
import numpy as np
import pandas as pd
import xarray as xr
# Add src to path
import sys
from pathlib import Path
src_dir = Path.cwd().parent / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

def process_folder(ctd_base, subfolder_path, output_base, split_profile=True):
    """
    Process one subfolder by:
       - Loading CTD files from the folder
       - Processing each CTD file individually
       - Optionally splitting SubOcean profiles (upward and downward)
       - Cleaning SubOcean data by flagging/removing rows where "Error Standard" > 1 (using DataCleaner)
       - Formatting CTD profiles and updating JSON using the update_experiment_title function
    """
    # Create relative path from the CTD base to maintain folder structure in output
    rel_path = os.path.relpath(subfolder_path, ctd_base)
    output_folder = os.path.join(output_base, rel_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    ctd_files = glob.glob(os.path.join(subfolder_path, "*.csv"))
    if not ctd_files:
        print(f"No CTD file found in {subfolder_path}")
        return

    for ctd_file in ctd_files:
        print(f"Processing CTD file: {ctd_file}")

        # Read and process CTD data once (they are used in all updates)
        try:
            ctd_df = pd.read_csv(ctd_file)
        except: 
            print(f"Error reading {ctd_file}. Skipping this file.")
            continue
        #If "timestamp" in columns, convert timestamp to "%Y/%m/%d" with column name:"yyyy_mm_dd" and %H:%M:%S" with column name "%H_%M_%S" format
        if "timestamp" in ctd_df.columns:
            ctd_df["date_mm_dd_yyyy"] = pd.to_datetime(ctd_df["timestamp"]).dt.strftime("%m/%d/%Y")
            ctd_df["time_hh_mm_ss"] = pd.to_datetime(ctd_df["timestamp"]).dt.strftime("%H:%M:%S")
            ctd_df.drop(columns=["timestamp"], inplace=True)
        else:
            ctd_df["date_mm_dd_yyyy"] = pd.to_datetime(ctd_df["date_mm_dd_yyyy"], format="%d/%m/%Y").dt.strftime("%m/%d/%Y")
            ctd_df["time_hh_mm_ss"] = pd.to_datetime(ctd_df["time_hh_mm_ss"], format="%H:%M:%S").dt.strftime("%H:%M:%S")
            

        ctd_ds = ctd_df.to_xarray()
        # Rename and set coordinate
        CTD_pressure_col = "pressure_dbar"  # rename "pressure_dbar" to "Pres"
        ctd_ds = ctd_ds.rename_vars({CTD_pressure_col: "Pres"})
        ctd_ds = ctd_ds.swap_dims({'index': 'Pres'})
        ctd_ds = ctd_ds.set_coords('Pres')
        ctd_ds = ctd_ds.drop_vars('index')
        # Convert pressure in dbar to psi for CTD
        ctd_ds["Pres"] = ctd_ds["Pres"] * 1.45038

        if split_profile:
            max_pressure_ctd = ctd_ds["Pres"].argmax()

            # Downward CTD portion
            ctd_ds_downard = ctd_ds.isel(Pres=slice(None, max_pressure_ctd.values))
            ctd_ds_downard_unique = ctd_ds_downard.copy()
            ctd_ds_downard_unique["Oxygen_percent"] = ctd_ds_downard_unique["oxygen_saturation_percent"]
            ctd_ds_downard_unique = ctd_ds_downard_unique.rename_vars({"Pres": "PrdE",
                                                                        "temperature_C": "Tv2C",
                                                                        "salinity_psu": "Sal2",
                                                                        "Oxygen_percent": "Sbeox2PS"})
            ctd_df_downard = ctd_ds_downard_unique[["Tv2C", "Sal2", "Sbeox2PS", "PrdE"]].to_dataframe()
            ctd_df_downard.reset_index(drop=True, inplace=True)
            ctd_df_downard = ctd_df_downard.drop_duplicates(subset='PrdE', keep='first')
            ctd_df_downard.sort_values(by=["PrdE"], inplace=True)

            # Upward CTD portion
            ctd_ds_upward = ctd_ds.isel(Pres=slice(max_pressure_ctd.values, None))
            if len(ctd_ds_upward.Pres) > 1:
                ctd_ds_upward_unique = ctd_ds_upward.copy()
                ctd_ds_upward_unique["Oxygen_percent"] = ctd_ds_upward_unique["oxygen_saturation_percent"]
                ctd_ds_upward_unique = ctd_ds_upward_unique.rename_vars({"Pres": "PrdE",
                                                                          "temperature_C": "Tv2C",
                                                                          "salinity_psu": "Sal2",
                                                                          "Oxygen_percent": "Sbeox2PS"})
                ctd_df_upward = ctd_ds_upward_unique[["Tv2C", "Sal2", "Sbeox2PS", "PrdE"]].to_dataframe()
                ctd_df_upward.reset_index(drop=True, inplace=True)
                ctd_df_upward = ctd_df_upward.drop_duplicates(subset='PrdE', keep='first')
                ctd_df_upward.sort_values(by=["PrdE"], inplace=True)
            else:
                ctd_df_upward = None
        else:
            # Process the entire profile as a single dataset
            ctd_ds_unique = ctd_ds.copy()
            ctd_ds_unique["Oxygen_percent"] = ctd_ds_unique["oxygen_saturation_percent"]
            ctd_ds_unique = ctd_ds_unique.rename_vars({"Pres": "PrdE",
                                                       "temperature_C": "Tv2C",
                                                       "salinity_psu": "Sal2",
                                                       "Oxygen_percent": "Sbeox2PS",
                                                       "time_hh_mm_ss": "hh:mm:ss",
                                                       "date_mm_dd_yyyy": "mm/dd/yyyy"})
            ctd_df_downard = ctd_ds_unique[["Tv2C", "Sal2", "Sbeox2PS", "PrdE","hh:mm:ss", "mm/dd/yyyy"]].to_dataframe()
            ctd_df_downard
            #reformat the date to yyyy_mm_dd
            ctd_df_downard.reset_index(drop=True, inplace=True)

            #ctd_df_downard = ctd_df_downard.drop_duplicates(subset='PrdE', keep='first')
            #ctd_df_downard.sort_values(by=["PrdE"], inplace=True)
            ctd_df_upward = None

        # Format CTD files and export them to output folder
        if split_profile:
            formatted_ctd_down = os.path.basename(ctd_file).replace(".csv", "_downward_formatted.asc")
            CTD_file_path_downward = os.path.join(output_folder, formatted_ctd_down)
            if ctd_df_downard["Sbeox2PS"].mean() < 0:
                print("WARNING: Oxygen values are negative, check the data!")
                print(ctd_file)
            ctd_df_downard.to_csv(CTD_file_path_downward, sep='\t', index=False)
            print(f"Exported formatted downward CTD to {CTD_file_path_downward}")

            if ctd_df_upward is not None:
                formatted_ctd_up = os.path.basename(ctd_file).replace(".csv", "_upward_formatted.asc")
                CTD_file_path_upward = os.path.join(output_folder, formatted_ctd_up)
                ctd_df_upward.to_csv(CTD_file_path_upward, sep='\t', index=False)
                print(f"Exported formatted upward CTD to {CTD_file_path_upward}")
        else:
            if "timestamp" in ctd_df.columns:
                #suffix = ctd_df["timestamp"].iloc[0][:18].replace("-","_").replace(":","_") + "_formatted.asc"
                suffix = "_formatted.asc"
            else:
                suffix = "_formatted.asc"
            formatted_ctd = os.path.basename(ctd_file).replace(".csv", suffix)
            
            CTD_file_path = os.path.join(output_folder, formatted_ctd)
            if ctd_df_downard["Sbeox2PS"].mean() < 0:
                print("WARNING: Oxygen values are negative, check the data!")
                print(ctd_file)
            ctd_df_downard.to_csv(CTD_file_path, sep='\t', index=False)
            print(f"Exported formatted CTD to {CTD_file_path}")


def find_all_folders_with_csv(base_path):
    """
    Recursively find all folders containing CSV files
    """
    folders_with_csv = []
    
    for root, dirs, files in os.walk(base_path):
        print(f"Checking folder: {root}")  # Debugging statement
        # Check if there are any CSV files in this directory
        csv_files = [file for file in files if file.lower().endswith('.csv')]
        if csv_files:
            print(f"Found CSV files: {csv_files}")  # Debugging statement
            folders_with_csv.append(root)
            
    return folders_with_csv

        
def main():
    # Base paths; modify if needed
    campaign_name = "LacNOX"  # Change as needed
    #campaign_name = "SubOcean++"
    ctd_path = f"C:/Users/cruz/Documents/SENSE/CTD_processing/data/Level2/{campaign_name}"
    output_folder = f"C:/Users/cruz/Documents/SENSE/CTD_processing/data/a2ps_format/{campaign_name}"
    
    # Find all folders containing CSV files
    folderlist = find_all_folders_with_csv(ctd_path)
    
    print(f"Found {len(folderlist)} folders with CSV files:")
    for folder in folderlist:
        print(f"- {folder}")
    
    # Process each folder
    for folder_path in folderlist:
        print(f"\nProcessing folder: {folder_path}")
        process_folder(ctd_path, folder_path, output_folder, split_profile=False)


if __name__ == '__main__':
    main()
