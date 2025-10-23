"""Data processing functions for CTD data."""

import numpy as np
import pandas as pd
import gsw
from config import COLUMN_MAPPINGS, SITE_CONFIG
import pandas as pd
import numpy as np
import seawater as sw
import os
import glob
import xarray as xr

class CTDFormatter:
    """Class to format CTD data into A2PS-compatible format, with time integration."""

    def __init__(self, split_profile=True):
        self.split_profile = split_profile

    def format_folder(self, input_folder, output_folder):
        """
        Format all CTD files in a folder into A2PS-compatible format, preserving subfolder structure.
        """
        for root, dirs, files in os.walk(input_folder):
            rel_path = os.path.relpath(root, input_folder)
            out_dir = os.path.join(output_folder, rel_path)
            os.makedirs(out_dir, exist_ok=True)
            ctd_files = [f for f in files if f.lower().endswith('.csv')]
            for ctd_file in ctd_files:
                self.format_file(os.path.join(root, ctd_file), out_dir)

    def format_file(self, ctd_file, output_folder):
        """
        Format a single CTD file into A2PS-compatible format, with time columns if present.
        """
        try:
            ctd_df = pd.read_csv(ctd_file)
        except Exception as e:
            print(f"Error reading {ctd_file}: {e}")
            return

        # Handle time columns
        if "timestamp" in ctd_df.columns:
            ctd_df["timestamp"] = pd.to_datetime(ctd_df["timestamp"])
            ctd_df = ctd_df.set_index("timestamp")
            ctd_df = ctd_df.resample("1S").interpolate(method="linear")
            ctd_df = ctd_df.reset_index()
            ctd_df["date_mm_dd_yyyy"] = pd.to_datetime(ctd_df["timestamp"]).dt.strftime("%m/%d/%Y")
            ctd_df["time_hh_mm_ss"] = pd.to_datetime(ctd_df["timestamp"]).dt.strftime("%H:%M:%S")
            #Interpolate to new date range of 1 second interval 

            ctd_df["time_hh_mm_ss"] = pd.to_datetime(ctd_df["time_hh_mm_ss"]).dt.strftime("%H:%M:%S")
            ctd_df.drop(columns=["timestamp"], inplace=True)
        elif "date_mm_dd_yyyy" in ctd_df.columns:
            #Interpolate to new date range of 1 second interval 
            ctd_df["date_mm_dd_yyyy"] = pd.to_datetime(ctd_df["date_mm_dd_yyyy"], format="%d/%m/%Y")
            ctd_df["time_hh_mm_ss"] = pd.to_datetime(ctd_df["time_hh_mm_ss"], format="%H:%M:%S")
            ctd_df = ctd_df.set_index(["date_mm_dd_yyyy", "time_hh_mm_ss"])
            ctd_df = ctd_df.resample("1S").interpolate(method="linear")
            ctd_df = ctd_df.reset_index()

            ctd_df["date_mm_dd_yyyy"] = pd.to_datetime(ctd_df["date_mm_dd_yyyy"], format="%d/%m/%Y").dt.strftime("%m/%d/%Y")
            ctd_df["time_hh_mm_ss"] = pd.to_datetime(ctd_df["time_hh_mm_ss"], format="%H:%M:%S").dt.strftime("%H:%M:%S")
        else:
            # If no time, force split profile for interpolation
            #self.split_profile = True
            print("No time columns found, splitting profile for interpolation.")

        ctd_ds = ctd_df.to_xarray()
        
        # Find pressure column - try different possible names
        pressure_col = None
        for col_name in ["pressure_dbar", "Pres", "pressure", "Press"]:
            if col_name in ctd_ds.variables:
                pressure_col = col_name
                break
        
        if pressure_col is None:
            print(f"Warning: No pressure column found in {ctd_file}")
            print(f"Available columns: {list(ctd_ds.variables.keys())}")
            return
            
        # Rename pressure column to "Pres" if it's not already named that
        if pressure_col != "Pres":
            ctd_ds = ctd_ds.rename_vars({pressure_col: "Pres"})
        
        ctd_ds = ctd_ds.swap_dims({'index': 'Pres'})
        ctd_ds = ctd_ds.set_coords('Pres')
        ctd_ds = ctd_ds.drop_vars('index')
        ctd_ds["Pres"] = ctd_ds["Pres"] * 1.45038  # dbar to psi

        if self.split_profile:
            max_pressure_idx = ctd_ds["Pres"].argmax()
            # Downward
            downward_ds = ctd_ds.isel(Pres=slice(None, max_pressure_idx.values))
            downward_df = self._prepare_dataframe(downward_ds)
            self._save_formatted_file(downward_df, ctd_file, output_folder, "_downward_formatted.asc")
            # Upward
            upward_ds = ctd_ds.isel(Pres=slice(max_pressure_idx.values, None))
            if len(upward_ds.Pres) > 1:
                upward_df = self._prepare_dataframe(upward_ds)
                self._save_formatted_file(upward_df, ctd_file, output_folder, "_upward_formatted.asc")
        else:
            # Entire profile, include time columns if present
            formatted_df = self._prepare_dataframe(ctd_ds, include_time=True, sort_by_pressure=False)
            self._save_formatted_file(formatted_df, ctd_file, output_folder, "_formatted.asc")

    def _prepare_dataframe(self, ctd_ds, include_time=False, sort_by_pressure=True):
        """Prepare DataFrame for A2PS formatting, optionally including time columns."""
        # Define potential rename mappings
        potential_renames = {
            "temperature_C": "Tv2C",
            "salinity_psu": "Sal2",
            "oxygen_saturation_percent": "Sbeox2PS",
            "Pres": "PrdE",
            "pressure_dbar": "PrdE"  # Alternative pressure column name
        }
        
        # Only include renames for columns that actually exist in the dataset
        rename_dict = {}
        available_vars = list(ctd_ds.variables.keys())
        
        for original_name, new_name in potential_renames.items():
            if original_name in available_vars:
                rename_dict[original_name] = new_name
        
        # Determine which columns will be available after renaming
        base_columns = []
        if "Tv2C" in rename_dict.values():
            base_columns.append("Tv2C")
        if "Sal2" in rename_dict.values():
            base_columns.append("Sal2")
        if "Sbeox2PS" in rename_dict.values():
            base_columns.append("Sbeox2PS")
        if "PrdE" in rename_dict.values():
            base_columns.append("PrdE")
        
        if include_time:
            # Add time columns if present
            if "time_hh_mm_ss" in ctd_ds.variables and "date_mm_dd_yyyy" in ctd_ds.variables:
                rename_dict["time_hh_mm_ss"] = "hh:mm:ss"
                rename_dict["date_mm_dd_yyyy"] = "mm/dd/yyyy"
                columns = base_columns + ["hh:mm:ss", "mm/dd/yyyy"]
            else:
                columns = base_columns
        else:
            columns = base_columns
        
        # Only proceed with renaming if we have columns to rename
        if rename_dict:
            ctd_ds = ctd_ds.rename_vars(rename_dict)
        
        # Only select columns that actually exist after renaming
        available_columns = [col for col in columns if col in ctd_ds.variables]
        
        if not available_columns:
            print(f"Warning: No standard CTD columns found in dataset")
            print(f"Available variables: {list(ctd_ds.variables.keys())}")
            # Return a dataframe with whatever variables are available
            df = ctd_ds.to_dataframe()
        else:
            df = ctd_ds[available_columns].to_dataframe()
        
        df.reset_index(drop=True, inplace=True)
        #Drop rows if more than 3 value is NaN
        df = df.dropna()

        if sort_by_pressure:
            if 'PrdE' in df.columns:
                df = df.drop_duplicates(subset='PrdE', keep='first')
                df.sort_values(by=["PrdE"], inplace=True)
            else:
                print("Warning: No pressure column (PrdE) found for sorting")
        else:
            if 'hh:mm:ss' in df.columns and 'mm/dd/yyyy' in df.columns:
                # Combine date and time into a single datetime object for proper sorting
                df['datetime'] = pd.to_datetime(df['mm/dd/yyyy'] + ' ' + df['hh:mm:ss'])
                df.sort_values(by="datetime", inplace=True)
                df.drop(columns=['datetime'], inplace=True)
            elif 'hh:mm:ss' in df.columns:
                # Fallback to sorting by time if date is not available
                df.sort_values(by=["hh:mm:ss"], inplace=True)
            else:
                print("Warning: No time column (hh:mm:ss) found for sorting")
        return df

    def _save_formatted_file(self, df, ctd_file, output_folder, suffix):
        """Save the formatted DataFrame to a file."""
        formatted_file = os.path.basename(ctd_file).replace(".csv", suffix)
        output_path = os.path.join(output_folder, formatted_file)
        if "Sbeox2PS" in df.columns and df["Sbeox2PS"].mean() < 0:
            print("WARNING: Oxygen values are negative, check the data!")
            print(ctd_file)
        df.to_csv(output_path, sep='\t', index=False)
        print(f"Exported formatted CTD to {output_path}")

def process_raw_data(df, ctd_type):
    """Process raw CTD data with all corrections and quality checks."""
    df_copy = df.copy()
    df = clean_air_data(df, ctd_type)
    if df.empty:
        print("No valid data found after removing air data, could not perform calculations ")
        return df_copy
    df = calculate_ocean_params(df, ctd_type)
    df = identify_downcast(df, ctd_type)
    df = quality_check_ph(df, ctd_type)
    
    return df

def conductivity_to_salinity_unesco(conductivity, temperature=15):
    """
    Convert conductivity (S/m) to salinity (PSU) using the PSS-78 equation.

    Parameters:
    - conductivity (float): Conductivity in S/m.
    - temperature (float, optional): Temperature in °C (default is 15°C for standard seawater).

    Returns:
    - float: Salinity in PSU.
    """
    # Standard conductivity at 35 PSU, 15°C, atmospheric pressure
    C_35_15_0 = 4.2914  # S/m

    # Compute conductivity ratio
    R = conductivity / C_35_15_0

    # Coefficients from UNESCO PSS-78
    a0, a1, a2, a3, a4, a5 = 0.0080, -0.1692, 25.3851, 14.0941, -7.0261, 2.7081

    # Compute salinity using PSS-78 equation
    S = (a0 +
         a1 * R**0.5 +
         a2 * R +
         a3 * R**1.5 +
         a4 * R**2 +
         a5 * R**2.5)

    return S

def get_parameter_name(ctd_type: str, param_type: str, standardized: bool = True) -> str:
    """
    Get parameter name for given CTD type and parameter type.
    
    Args:
        ctd_type: Type of CTD ('idronaut', 'seabird', etc.)
        param_type: Parameter to look up (e.g., 'conductivity')
        standardized: If True, return standardized name instead of raw name
        
    Returns:
        str: Column name (raw or standardized based on flag)
    """
    ctd_type = ctd_type.lower()
    
    # Common parameter mappings to standardized column names
    PARAM_MAPPING = {
        'conductivity': 'conductivity_mS_per_m',
        'pressure': 'pressure_dbar',
        'temperature': 'temperature_C',
        'salinity': 'salinity_psu',
        'oxygen_saturation': 'oxygen_saturation_percent',
        'oxygen_concentration': 'oxygen_concentration_ml_per_L',
        'dissolved_o2_saturation': 'oxygen_saturation_percent',  # Map RBR dissolved O2 saturation
        'dissolved_o2_concentration': 'dissolved_o2_concentration',
        'depth': 'depth_m',
        'ph': 'ph',
        'turbidity': 'turbidity_NTU',
        'PAR': 'PAR_umol_m2_s',
        'chlorophyll': 'chlorophyll_mg_m3',
        'fluorescence': 'fluorescence_rfu'
    }
    
    # If looking for standardized name and it's a common parameter, return directly
    if standardized and param_type in PARAM_MAPPING:
        return PARAM_MAPPING[param_type]
    
    # Otherwise search in mappings
    if not standardized:
        # Looking for raw name based on standardized name
        for raw_name, (std_name, _) in COLUMN_MAPPINGS[ctd_type].items():
            if param_type.lower() in std_name.lower():
                return raw_name
    else:
        # Try partial match on standardized names
        for raw_name, (std_name, _) in COLUMN_MAPPINGS[ctd_type].items():
            if param_type.lower() in std_name.lower():
                return std_name
    
    return None

def clean_air_data(df: pd.DataFrame, ctd_type: str, threshold_cond=None) -> pd.DataFrame:
    """Remove air measurements and apply corrections."""
    ctd_type = ctd_type.lower()
    
    # Set threshold_cond based on ctd_type
    if threshold_cond is None:
        threshold_cond = 5 if ctd_type == "exo" else 0.15

    # Get standardized column names directly
    cond_col = 'conductivity_mS_per_m'
    pres_col = 'pressure_dbar'
    o2_col = 'oxygen_saturation_percent'
    o2_conc = "oxygen_concentration_ml_per_L"
    par_col = 'PAR_umol_m2_s'
    chla_col = 'Chl(a)Phy-EthrinPhy-Cyanin'
    
    # Check if columns exist, fall back to parameter lookup if not
    if cond_col not in df.columns:
        cond_col = get_parameter_name(ctd_type, 'conductivity', standardized=True)
    if pres_col not in df.columns:
        pres_col = get_parameter_name(ctd_type, 'pressure', standardized=True)
    if o2_col not in df.columns:
        o2_col = get_parameter_name(ctd_type, 'oxygen_saturation', standardized=True)
    if par_col not in df.columns:
        par_col = get_parameter_name(ctd_type, 'PAR', standardized=True)
    if chla_col not in df.columns:
        chla_col = get_parameter_name(ctd_type, 'Chl(a)', standardized=True)
        if chla_col not in df.columns:
            chla_col = get_parameter_name(ctd_type, 'chlorophyll', standardized=True)
            if chla_col not in df.columns:
                chla_col = get_parameter_name(ctd_type, 'fluorescence', standardized=True)
    if o2_conc not in df.columns:
        o2_conc = get_parameter_name(ctd_type, 'oxygen_concentration', standardized=True)

    # Check for required columns - pressure is mandatory, conductivity is optional
    # Some RBR files may not have conductivity measurements
    has_conductivity = cond_col is not None and cond_col in df.columns
    has_pressure = pres_col is not None and pres_col in df.columns
    
    if not has_pressure:
        raise ValueError(
            f"Missing required pressure column for CTD type {ctd_type}\n"
            f"Looking for: pressure_dbar\n"
            f"Found columns: {df.columns.tolist()}"
        )
    
    if not has_conductivity:
        print(f"Warning: No conductivity column found for CTD type {ctd_type}")
        print(f"Available columns: {df.columns.tolist()}")
        print("Skipping conductivity-based air data removal")
    
    # Check which optional columns are available
    has_oxygen = o2_col is not None and o2_col in df.columns
    has_par = par_col is not None and par_col in df.columns
    has_chla = chla_col is not None and chla_col in df.columns
    has_o2_conc = o2_conc is not None and o2_conc in df.columns
    # Add debug print
    print(f"Processing columns: pressure={pres_col}")
    if has_conductivity:
        print(f"Conductivity column: {cond_col}")
    if has_oxygen:
        print(f"Oxygen column found: {o2_col}")
    else:
        print("No oxygen column found - skipping oxygen corrections")
    if has_par:
        print(f"PAR column found: {par_col}")
    if has_chla:
        print(f"Chlorophyll column found: {chla_col}")
    if has_o2_conc:
        print(f"Oxygen concentration column found: {o2_conc}")    
    # Process data using standardized column names
    # If no conductivity, try to identify air data using pressure (close to surface)
    if has_conductivity:
        df_air = df[df[cond_col] < threshold_cond]
    else:
        # For instruments without conductivity, use pressure threshold for air detection
        # Air measurements typically have pressure close to atmospheric (near 0 dbar)
        pressure_threshold = 2.0  # dbar - measurements within 2 dbar of surface
        df_air = df[df[pres_col] < pressure_threshold]
        print(f"Using pressure threshold {pressure_threshold} dbar for air detection (no conductivity available)")
    
    if df_air.empty:
        print("No air data found, skipping corrections")
        return df
        
    # Calculate pressure offset
    ctd_pres_offset = df_air[pres_col].median()
    
    # Calculate oxygen offset only if oxygen column exists
    ctd_O2_offset = 0  # Default value
    if has_oxygen:
        ctd_O2_offset = df_air[o2_col].median() - 100
        if np.abs(ctd_O2_offset) > 50:
            ctd_O2_offset = 0
            print("Error with offsetting Oxygen data with air, the oxygen in the air is badly measured")


    # Filter out air data
    if has_conductivity:
        df = df[df[cond_col] > threshold_cond].copy()
    else:
        # Use pressure threshold instead of conductivity
        pressure_threshold = 2.0  # dbar
        df = df[df[pres_col] >= pressure_threshold].copy()
        print(f"Filtered air data using pressure threshold {pressure_threshold} dbar")
    
    # Apply corrections
    df[pres_col] = df[pres_col] - ctd_pres_offset
    
    if has_oxygen:
        old_o2 = df[o2_col].copy()
        df[o2_col] = df[o2_col] - ctd_O2_offset
        if df[o2_col].mean() < 0:
            print("Error: Negative mean oxygen saturation after correction")
    if has_o2_conc:
        #Recalcuate oxygen concentration based on corrected saturation using the general formula with old and new saturation Oxygen_concentration  = oxygen_saturation_percent * solubility / 100 where solubility stay the same
        df[o2_conc] = df[o2_conc]*df[o2_col]/old_o2 
    else:
        df[o2_conc] = df.apply(
            lambda row: calculate_oxygen_mgl(
                row['temperature_C'], 
                row['salinity_psu'], 
                row[o2_col]
            ), axis=1
        )

    # Filter df[pres_col] < 0
    df = df[df[pres_col] > 0].copy()
    
    # Calculate PAR average in the air (optional)
    if has_par and par_col in df_air.columns:
        par_avg_air = df_air[par_col].mean()
        df['PAR_avg_air'] = par_avg_air
    else:
        if has_par:
            print("PAR column is missing in the air data.")
    
    # Handle chlorophyll offset (optional)
    if has_chla and chla_col in df_air.columns:
        # Change the calculation to take the value under 50m, do the median and assign it to the offset.
        chla_offset_50m = df[df[pres_col] > 50][chla_col].mean()
        if not np.isnan(chla_offset_50m):
            df[chla_col] = df[chla_col] - chla_offset_50m
        
    return df

def identify_downcast(df: pd.DataFrame, ctd_type: str) -> pd.DataFrame:
    """
    Identify downcast portion of profile.
    
    Args:
        df: DataFrame containing CTD data
        ctd_type: Type of CTD ('idronaut' or 'seabird')
        
    Returns:
        DataFrame with added 'is_downcast' column
    """
    # Check for depth or pressure in standardized column names
    if 'depth_m' in df.columns:
        depth_col = 'depth_m'
    else:
        depth_col = 'pressure_dbar'
    
    if depth_col not in df.columns:
        raise ValueError(
            f"Could not find depth or pressure column\n"
            f"Available columns: {df.columns.tolist()}"
        )
    
    max_depth_idx = df[depth_col].idxmax()
    df["is_downcast"] = df.index <= max_depth_idx
    
    return df

def quality_check_ph(df: pd.DataFrame, ctd_type: str) -> pd.DataFrame:
    """Apply quality control to pH measurements."""
    # Use direct column name
    ph_col = 'ph'
    
    # Check if pH column exists - first try direct match
    if ph_col in df.columns:
        df.loc[(df[ph_col] < 6) | (df[ph_col] > 9), ph_col] = np.nan
        return df
    
    # Try to find pH column using COLUMN_MAPPINGS if direct match failed
    ctd_type = ctd_type.lower()
    for raw_name, (std_name, _) in COLUMN_MAPPINGS[ctd_type].items():
        if 'ph' == std_name.lower():
            if raw_name in df.columns:
                df.loc[(df[raw_name] < 6) | (df[raw_name] > 9), raw_name] = np.nan
                print(f"Applied pH quality check to column: {raw_name}")
                break
    
    return df



def find_mld(temp, dens, depth, thresh_temp=0.2, thresh_dens=0.03, thresh_depth=1):
    """Calculate mixed layer depth using temperature and density criteria."""

    #Filter depth > 1, adjust temp, dens 
    temp = temp[depth > thresh_depth]
    dens = dens[depth > thresh_depth]
    depth = depth[depth > thresh_depth]

    temp_surf = temp.iloc[0]
    dens_surf = dens.iloc[0]
    
    mld_temp = depth[abs(temp - temp_surf) > thresh_temp].iloc[0] if any(abs(temp - temp_surf) > thresh_temp) else np.nan
    mld_dens = depth[abs(dens - dens_surf) > thresh_dens].iloc[0] if any(abs(dens - dens_surf) > thresh_dens) else np.nan
    
    return pd.Series({'mld_temp': mld_temp, 'mld_dens': mld_dens})

def calculate_oxygen_mgkg(temp: float, sal: float, o2sat: float) -> float:
    """
    Calculate oxygen in mg/kg using the provided formula.
    
    Args:
        temp: Temperature in °C
        sal: Salinity in PSU
        o2sat: Oxygen saturation in %
        
    Returns:
        float: Oxygen concentration in mg/kg
    """
    # Constants
    a0 = -138.74202
    a1 = 1.572288e5
    a2 = -6.637149e7
    a3 = 1.243678e10
    a4 = -8.621061e11
    b0 = 0.020573
    b1 = -12.142
    b2 = 2363.1
    
    # Convert temperature to Kelvin
    T = temp + 273.15
    
    # Calculate ln(CO)
    lnCO = (a0 + (a1/T) + (a2/(T*T)) + (a3/(T*T*T)) + (a4/(T*T*T*T)) - 
            (sal * (b0 + (b1/T) + (b2/(T*T)))))
    
    # Calculate final result
    return (o2sat * np.exp(lnCO)) / 100.0

def calculate_oxygen_mgl(temp: float, sal: float, o2sat: float) -> float:
    """
    Calculate oxygen in mg/L using the provided formula.
    
    Args:
        temp: Temperature in °C
        sal: Salinity in PSU
        o2sat: Oxygen saturation in %
        
    Returns:
        float: Oxygen concentration in mg/L
    """
    # Constants
    a1 = -173.4292
    a2 = 249.6339
    a3 = 143.3483
    a4 = -21.8492
    b1 = -0.033096
    b2 = 0.014259
    b3 = -0.0017
    cnv = 1.428
    
    # Calculate T1
    T = (temp + 273.15) / 100.0
    
    # Calculate capacity
    capac = cnv * np.exp((a1 + (a2 * (100.0/(temp+273.15))) + 
                         (a3*np.log(T)) + (a4*T)) + 
                        (sal * (b1 + (b2*T) + (b3*T*T))))
    
    # Calculate final result
    return (o2sat * capac) / 100.0

def calculate_ocean_params(df: pd.DataFrame, ctd_type: str) -> pd.DataFrame:
    """Calculate oceanographic parameters."""
    ctd_type = ctd_type.lower()
    
    # Use direct column names
    pres_col = 'pressure_dbar'
    temp_col = 'temperature_C'
    cond_col = 'conductivity_mS_per_m'
    sal_col = 'salinity_psu'
    o2_col = 'oxygen_saturation_percent'
    
    # Check columns exist
    has_pressure = pres_col in df.columns
    has_temperature = temp_col in df.columns
    has_conductivity = cond_col in df.columns
    has_salinity = sal_col in df.columns
    
    # Try to find alternative columns
    if not has_pressure:
        alt_col = get_parameter_name(ctd_type, 'pressure', standardized=True)
        if alt_col and alt_col in df.columns:
            pres_col = alt_col
            has_pressure = True
    
    if not has_temperature:
        alt_col = get_parameter_name(ctd_type, 'temperature', standardized=True)
        if alt_col and alt_col in df.columns:
            temp_col = alt_col
            has_temperature = True
    
    if not has_conductivity:
        alt_col = get_parameter_name(ctd_type, 'conductivity', standardized=True)
        if alt_col and alt_col in df.columns:
            cond_col = alt_col
            has_conductivity = True
    
    if not has_salinity:
        alt_col = get_parameter_name(ctd_type, 'salinity', standardized=True)
        if alt_col and alt_col in df.columns:
            sal_col = alt_col
            has_salinity = True
    
    # Check what we can calculate with available data
    if not has_pressure:
        raise ValueError(f"Pressure column is required but not found. Available columns: {df.columns.tolist()}")
    
    print(f"Available columns for ocean calculations:")
    print(f"  Pressure: {pres_col if has_pressure else 'MISSING'}")
    print(f"  Temperature: {temp_col if has_temperature else 'MISSING'}")
    print(f"  Conductivity: {cond_col if has_conductivity else 'MISSING'}")
    print(f"  Salinity: {sal_col if has_salinity else 'MISSING'}")
    
    # If we don't have basic CTD parameters, skip ocean parameter calculations
    if not (has_temperature and (has_conductivity or has_salinity)):
        print("Warning: Insufficient data for full ocean parameter calculations")
        print("Need at least temperature and either conductivity or salinity")
        return df
    
    # Add debug print
    print(f"Calculating ocean parameters using columns: {pres_col}, {temp_col}, {sal_col}, {o2_col}")
    
    # Ensure depth column exists
    if 'depth_m' not in df.columns:
        if pres_col in df.columns:
            # Calculate depth from pressure using GSW
            try:
                # Convert pressure to depth (negative values indicate depth below sea level)
                df['depth_m'] = -gsw.z_from_p(df[pres_col].to_numpy(), SITE_CONFIG['LATITUDE'])
                print("Depth column calculated from pressure.")
            except Exception as e:
                print(f"Error calculating depth from pressure: {e}")
                df['depth_m'] = np.nan
        else:
            raise ValueError("Pressure column is missing, cannot calculate depth.")
    
    # Convert to numpy arrays - only if columns exist
    p = df[pres_col].abs().to_numpy()
    
    # Calculate derived parameters only if we have the required data
    if has_temperature and has_salinity:
        temp = df[temp_col].to_numpy()
        sal = df[sal_col].to_numpy()
        
        SA = gsw.SA_from_SP(sal, p, SITE_CONFIG['LONGITUDE'], SITE_CONFIG['LATITUDE'])
        CT = gsw.CT_from_t(SA, temp, p)
        
        # Calculate derived parameters
        df['pot_temp_C'] = gsw.pt_from_CT(SA, CT)
        df['density_kg_m3'] = gsw.density.rho(SA, CT, p)
        
        # Calculate MLD if we have the required columns
        if has_temperature and 'density_kg_m3' in df.columns:
            mld = find_mld(df[temp_col], df['density_kg_m3'], df[pres_col])
            df['mld_temp'] = mld['mld_temp']
            df['mld_dens'] = mld['mld_dens']
        
        # Calculate oxygen solubility
        o2_sol_umol = gsw.O2sol(SA, CT, p, SITE_CONFIG['LONGITUDE'], SITE_CONFIG['LATITUDE'])
        o2_sol_mll = o2_sol_umol * 0.022391  # μmol/kg to mL/L
        o2_sol_mgl = o2_sol_mll * 1.42905    # mL/L to mg/L
        
        # Store solubility values
        df['o2_solubility_mll'] = o2_sol_mll
        df['o2_solubility_mgl'] = o2_sol_mgl
        
        # Check if oxygen column exists for concentration calculations
        if o2_col in df.columns:
            # Calculate oxygen concentrations using vectorized operations
            df['o2_mgkg_Weiss'] = df.apply(
                lambda row: calculate_oxygen_mgkg(
                    row[temp_col], 
                    row[sal_col], 
                    row[o2_col]
                ), axis=1
            )
            df['o2_mgl_Weiss'] = df.apply(
                lambda row: calculate_oxygen_mgl(
                    row[temp_col], 
                    row[sal_col], 
                    row[o2_col]
                ), axis=1
            )
        else:
            print("No oxygen saturation data available - skipping oxygen concentration calculations")
        
    else:
        print("Skipping oceanographic calculations - need temperature and salinity")
    '''
    # Calculate N² with proper handling of warnings
    try:
        # Initialize N2 column with NaN
        df['N2'] = np.nan

        # Remove duplicate pressure values that can cause division by zero
        unique_mask = np.diff(p) != 0
        if any(unique_mask):  # Only proceed if we have valid differences
            SA_clean = SA[:-1][unique_mask]
            CT_clean = CT[:-1][unique_mask]
            p_clean = p[:-1][unique_mask]

            # Calculate N² only for valid data points
            with np.errstate(divide='ignore', invalid='ignore'):
                N2, pmid = gsw.Nsquared(SA_clean, CT_clean, p_clean)

                # Replace invalid values with NaN
                N2 = np.where(np.isfinite(N2), N2, np.nan)

                # Assign N2 values to the DataFrame only for valid indexes
                for i, mid_p in enumerate(pmid):
                    closest_idx = (np.abs(p - mid_p)).argmin()
                    # Only assign if closest_idx is a valid index in df
                    if closest_idx in df.index:
                        df.at[closest_idx, 'N2'] = N2[i]

        # Optionally, drop rows with NaN index (if any were created, which shouldn't happen with this logic)
        df = df[df.index.notna()]

    except Exception as e:
        print(f"Warning: Error calculating N2: {e}")
        df['N2'] = np.nan
    '''
    if ctd_type == "seabird" and has_conductivity and has_temperature and has_salinity:
        # Only rename if not already renamed (prevent duplicate suffixes)
        seabird_col = sal_col + "_seabird"
        if sal_col in df.columns and seabird_col not in df.columns:
            df = df.rename(columns={sal_col: seabird_col})
        # Calculate new salinity using seawater library (Idronaut equation)
        # sw.salt expects conductivity RATIO (R), not absolute conductivity
        # R = measured_conductivity / reference_conductivity
        # Reference conductivity = 42.914 mS/cm (standard seawater: 35 PSU, 15°C, 0 dbar)
        # First convert mS/m to mS/cm, then calculate ratio
        cond_mScm = df[cond_col] / 100.0  # Convert mS/m to mS/cm
        conductivity_ratio = cond_mScm / 42.914  # Calculate conductivity ratio
        df[sal_col] = sw.salt(conductivity_ratio, df[temp_col], df[pres_col])
    elif ctd_type == "seabird" and not has_conductivity:
        print("Warning: Seabird-specific salinity calculation skipped - no conductivity data")

    return df