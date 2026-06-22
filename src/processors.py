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
from scipy.ndimage import uniform_filter1d

class CTDFormatter:
    """Class to format CTD data into A2PS-compatible format, with time integration."""

    def __init__(self, split_profile=True, sort_by_pressure=True, pressure_offset=0.0):
        self.split_profile = split_profile
        self.sort_by_pressure = sort_by_pressure
        self.pressure_offset = pressure_offset

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
            _num = ctd_df.select_dtypes(include="number")
            _str = ctd_df.select_dtypes(exclude="number")
            _num = _num.resample("1s").interpolate(method="linear")
            if not _str.empty:
                ctd_df = pd.concat([_num, _str.resample("1s").ffill()], axis=1)
            else:
                ctd_df = _num
            ctd_df = ctd_df.reset_index()
            ctd_df["date_mm_dd_yyyy"] = pd.to_datetime(ctd_df["timestamp"]).dt.strftime("%m/%d/%Y")
            ctd_df["time_hh_mm_ss"] = pd.to_datetime(ctd_df["timestamp"]).dt.strftime("%H:%M:%S")
            ctd_df["time_hh_mm_ss"] = pd.to_datetime(ctd_df["time_hh_mm_ss"]).dt.strftime("%H:%M:%S")
            ctd_df.drop(columns=["timestamp"], inplace=True)
        elif "date_mm_dd_yyyy" in ctd_df.columns:
            ctd_df["date_mm_dd_yyyy"] = pd.to_datetime(ctd_df["date_mm_dd_yyyy"], format="%d/%m/%Y")
            ctd_df["time_hh_mm_ss"] = pd.to_datetime(ctd_df["time_hh_mm_ss"], format="%H:%M:%S")
            ctd_df = ctd_df.set_index(["date_mm_dd_yyyy", "time_hh_mm_ss"])
            _num = ctd_df.select_dtypes(include="number")
            _str = ctd_df.select_dtypes(exclude="number")
            _num = _num.resample("1s").interpolate(method="linear")
            if not _str.empty:
                ctd_df = pd.concat([_num, _str.resample("1s").ffill()], axis=1)
            else:
                ctd_df = _num
            ctd_df = ctd_df.reset_index()
            ctd_df["date_mm_dd_yyyy"] = pd.to_datetime(ctd_df["date_mm_dd_yyyy"]).dt.strftime("%m/%d/%Y")
            ctd_df["time_hh_mm_ss"] = pd.to_datetime(ctd_df["time_hh_mm_ss"]).dt.strftime("%H:%M:%S")
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
        # Apply pressure offset (in dbar) before converting to psi
        if self.pressure_offset != 0.0:
            ctd_ds["Pres"] = ctd_ds["Pres"] + self.pressure_offset
        ctd_ds["Pres"] = ctd_ds["Pres"] * 1.45038  # dbar to psi

        pres_tag = "_Pres" if self.sort_by_pressure else ""
        if self.split_profile:
            max_pressure_idx = ctd_ds["Pres"].argmax()
            # Downward
            downward_ds = ctd_ds.isel(Pres=slice(None, max_pressure_idx.values))
            downward_df = self._prepare_dataframe(downward_ds)
            self._save_formatted_file(downward_df, ctd_file, output_folder, f"{pres_tag}_downward_formatted.asc")
            # Upward
            upward_ds = ctd_ds.isel(Pres=slice(max_pressure_idx.values, None))
            if len(upward_ds.Pres) > 1:
                upward_df = self._prepare_dataframe(upward_ds)
                self._save_formatted_file(upward_df, ctd_file, output_folder, f"{pres_tag}_upward_formatted.asc")
        else:
            # Entire profile, include time columns if present
            formatted_df = self._prepare_dataframe(ctd_ds, include_time=True, sort_by_pressure=self.sort_by_pressure)
            self._save_formatted_file(formatted_df, ctd_file, output_folder, f"{pres_tag}_formatted.asc")

    def _prepare_dataframe(self, ctd_ds, include_time=False, sort_by_pressure=True):
        """Prepare DataFrame for A2PS formatting, optionally including time columns.

        The first columns are always: Tv2C, Sal2, Sbeox2PS, PrdE [, hh:mm:ss, mm/dd/yyyy].
        All other columns present in the dataset are appended after those, in their
        original order, so no data is discarded.
        """
        potential_renames = {
            "temperature_C": "Tv2C",
            "salinity_psu": "Sal2",
            "oxygen_saturation_percent": "Sbeox2PS",
        }

        available_vars = list(ctd_ds.variables.keys())

        # Build rename dict for columns that actually exist
        rename_dict = {}
        for original_name, new_name in potential_renames.items():
            if original_name in available_vars:
                rename_dict[original_name] = new_name

        # Also rename time columns when requested
        if include_time:
            if "time_hh_mm_ss" in available_vars:
                rename_dict["time_hh_mm_ss"] = "hh:mm:ss"
            if "date_mm_dd_yyyy" in available_vars:
                rename_dict["date_mm_dd_yyyy"] = "mm/dd/yyyy"

        if rename_dict:
            ctd_ds = ctd_ds.rename_vars(rename_dict)

        # Convert FULL dataset to DataFrame (keeps all columns)
        df = ctd_ds.to_dataframe()

        # reset_index(drop=False) promotes "Pres" dimension to a regular column
        df.reset_index(drop=False, inplace=True)
        if "index" in df.columns:
            df.drop(columns=["index"], inplace=True)

        # Rename pressure dimension → PrdE
        if "Pres" in df.columns:
            df.rename(columns={"Pres": "PrdE"}, inplace=True)

        # Build column order: priority columns first, then everything else
        priority = ["Tv2C", "Sal2", "Sbeox2PS", "PrdE"]
        if include_time:
            priority += ["hh:mm:ss", "mm/dd/yyyy"]

        front_cols = [c for c in priority if c in df.columns]
        extra_cols = [c for c in df.columns if c not in front_cols]
        df = df[front_cols + extra_cols]

        # Drop rows that are NaN in the priority columns only
        priority_present = [c for c in ["Tv2C", "Sal2", "Sbeox2PS", "PrdE"] if c in df.columns]
        if priority_present:
            df = df.dropna(subset=priority_present)
        else:
            df = df.dropna()

        if sort_by_pressure:
            if 'PrdE' in df.columns:
                df = df.drop_duplicates(subset='PrdE', keep='first')
                df.sort_values(by=["PrdE"], inplace=True)
            else:
                print("Warning: No pressure column (PrdE) found for sorting")
        else:
            if 'hh:mm:ss' in df.columns and 'mm/dd/yyyy' in df.columns:
                df['datetime'] = pd.to_datetime(df['mm/dd/yyyy'] + ' ' + df['hh:mm:ss'])
                df.sort_values(by="datetime", inplace=True)
            elif 'hh:mm:ss' in df.columns:
                df.sort_values(by=["hh:mm:ss"], inplace=True)
            else:
                print("Warning: No time column (hh:mm:ss) found for sorting")

        return df

    # Minimum temperature threshold accepted by downstream software (A2PS)
    TV2C_MIN = -0.994

    def _save_formatted_file(self, df, ctd_file, output_folder, suffix):
        """Save the formatted DataFrame to a file."""
        formatted_file = os.path.basename(ctd_file).replace(".csv", suffix)
        output_path = os.path.join(output_folder, formatted_file)
        if "Sbeox2PS" in df.columns and df["Sbeox2PS"].mean() < 0:
            print("WARNING: Oxygen values are negative, check the data!")
            print(ctd_file)
        # ⚠️  TEMPORARY FIX — downstream software (A2PS) does not parse
        # temperatures below -0.994 °C correctly (fixed-width column overflow).
        # Values below this threshold are clamped to -0.994. This is an
        # APPROXIMATION — review when the software limitation is resolved.
        if "Tv2C" in df.columns:
            n_clamped = (df["Tv2C"] < self.TV2C_MIN).sum()
            if n_clamped > 0:
                print(
                    f"WARNING [Tv2C clamp]: {n_clamped} temperature values below "
                    f"{self.TV2C_MIN} °C were clamped to {self.TV2C_MIN} °C "
                    f"(TEMPORARY FIX for downstream software limit). File: {ctd_file}"
                )
                df = df.copy()
                df["Tv2C"] = df["Tv2C"].clip(lower=self.TV2C_MIN)
        df.to_csv(output_path, sep='\t', index=False, na_rep='NaN')
        print(f"Exported formatted CTD to {output_path}")

def process_raw_data(df, ctd_type):
    """Process raw CTD data with all corrections and quality checks."""
    df_copy = df.copy()
    df = quality_check_oxygen(df, ctd_type)
    df = clean_air_data(df, ctd_type)
    if df.empty:
        print("No valid data found after removing air data, could not perform calculations ")
        return df_copy
    df = calculate_ocean_params(df, ctd_type)
    df = identify_downcast(df, ctd_type)
    df = quality_check_ph(df, ctd_type)
    df = correct_chla_offset_and_pressure(df)
    
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
    o2_conc = "oxygen_concentration_mg_per_L"
    par_col = 'PAR_umol_m2_s'
    chla_col = 'Chl(a)Phy-EthrinPhy-Cyanin' 
    depth_col = 'depth_m'
    
    # Check if columns exist, fall back to parameter lookup if not
    if cond_col not in df.columns:
        # Search for conductivity column variants directly in the dataframe
        conductivity_variants = ['conductivity_mS_per_m', 'conductivity_mS_per_cm', 'conductivity']
        cond_col = None
        for variant in conductivity_variants:
            if variant in df.columns:
                cond_col = variant
                break
        
        # If still not found, try get_parameter_name as fallback
        if cond_col is None:
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
    if depth_col not in df.columns:
        depth_col = get_parameter_name(ctd_type, 'depth', standardized=True)

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
    '''
    #Make twinx to plot df[o2_col] and df[o2_conc] together for checking
    import matplotlib.pyplot as plt
    plt.figure()

    df[o2_conc].plot()
    df[o2_conc+"_idronaut_corrected"].plot()
    df[o2_conc+"Weiss"].plot()
    ax2 = plt.twinx()
    df[o2_col].plot(ax=ax2)
    plt.legend()'''
    # Filter df[pres_col] < 0
    df = df[df[pres_col] > 0].copy()
    if depth_col in df.columns: 
        df = df[df[depth_col] > 0].copy()
    df = correct_chla_offset_and_pressure(df)
    # Calculate PAR average in the air (optional)
    if has_par and par_col in df_air.columns:
        par_avg_air = df_air[par_col].mean()
        df['PAR_avg_air'] = par_avg_air
    else:
        if has_par:
            print("PAR column is missing in the air data.")
    
    return df

def correct_chla_offset_and_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply offset to chlorophyll columns based on deep water average and remove negative pressure readings.
    
    Args:
        df (pd.DataFrame): The input DataFrame with CTD data.
        
    Returns:
        pd.DataFrame: The processed DataFrame.
    """
    
    # Standardized pressure column name
    pres_col = 'pressure_dbar'
    
    if pres_col not in df.columns:
        print(f"Warning: Pressure column '{pres_col}' not found. Skipping chlorophyll offset and pressure correction.")
        return df
        
    # List of chlorophyll and related columns to apply offset
    chla_cols_to_offset = [
        "Trx-chl(a)", "Trx-Chl-a", "Pethr", "Pchan", 
        "Chl(a)", "Phy-Ethrin", "Phy-Cyanin"
    ]
    
    # Apply offset for each chlorophyll column found in the DataFrame
    for col in chla_cols_to_offset:
        if col in df.columns:
            # Calculate the mean from data at 50m depth or more
            deep_water_mean = df[df[pres_col] >= 50][col].min() #CHANGED TO HAVE NO NEGATIVE VALUES
            
            if pd.notna(deep_water_mean):
                # Apply the offset to the entire column
                df[col] = df[col] - deep_water_mean
                print(f"Applied offset of {deep_water_mean:.4f} to column '{col}'.")
            else:
                print(f"Warning: Could not calculate offset for '{col}' (no data >= 50m or all values are NaN).")

    # Remove rows with negative pressure after all corrections
    df = df[df[pres_col] >= 0].copy()
    
    return df

def identify_downcast(df: pd.DataFrame, ctd_type: str, smoothing_window: int = 10) -> pd.DataFrame:
    """
    Identify downcast vs upcast portions of profile(s) based on depth/pressure trends.
    Handles multiple casts in a single profile by detecting depth gradients.
    
    Args:
        df: DataFrame containing CTD data
        ctd_type: Type of CTD ('idronaut' or 'seabird')
        smoothing_window: Window size for smoothing depth data to detect trends (default: 10)
        
    Returns:
        DataFrame with added 'is_downcast' column (True=downcast, False=upcast, None=unclear)
    """
    # Check for depth or pressure in standardized column names
    if 'depth_m' in df.columns:
        depth_col = 'depth_m'
    elif 'pressure_dbar' in df.columns:
        depth_col = 'pressure_dbar'
    else:
        raise ValueError(
            f"Could not find depth or pressure column\n"
            f"Available columns: {df.columns.tolist()}"
        )
    
    # Create a copy of depth data
    depth_data = df[depth_col].copy()
    
    # Ensure depth data is numeric
    depth_data = pd.to_numeric(depth_data, errors='coerce')
    
    # Handle NaN values with forward/backward fill
    depth_data = depth_data.ffill().bfill()
    
    if depth_data.isna().all():
        print("Warning: All depth values are non-numeric or NaN. Setting all to None.")
        df['is_downcast'] = None
        return df
    
    # Convert to absolute values (depth/pressure should be positive)
    depth_data = np.abs(depth_data)
    
    # Smooth the depth data to reduce noise
    if len(depth_data) > smoothing_window:
        smoothed_depth = uniform_filter1d(depth_data.values.astype(float), size=smoothing_window, mode='nearest')
    else:
        smoothed_depth = depth_data.values
    
    # Calculate depth gradient (rate of change)
    if len(smoothed_depth) < 2:
        print("Warning: Not enough depth data to calculate gradient. Setting all to None.")
        df['is_downcast'] = None
        return df
    depth_gradient = np.gradient(smoothed_depth)
    
    # Initialize is_downcast array with None
    is_downcast = np.full(len(df), None, dtype=object)
    
    # Use a rolling window to determine local trend
    window_size = max(smoothing_window, 20)  # At least 20 points for trend detection
    half_window = window_size // 2
    
    # Vectorized gradient calculation for windows
    for i in range(len(depth_gradient)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(depth_gradient), i + half_window + 1)
        
        # Calculate average gradient in this window
        window_gradient = np.mean(depth_gradient[start_idx:end_idx])
        
        # Threshold for determining trend (in meters/dbar per sample)
        gradient_threshold = 0.001  # Small threshold to handle nearly flat sections
        
        if window_gradient > gradient_threshold:
            is_downcast[i] = True  # Depth increasing = downcast
        elif window_gradient < -gradient_threshold:
            is_downcast[i] = False  # Depth decreasing = upcast
        # else: leave as None for unclear/flat sections
    
    # Convert to pandas Series for easier manipulation
    is_downcast_series = pd.Series(is_downcast, index=df.index)
    
    # Fill small gaps (up to 10 points) where both neighbors agree
    gap_size = 10
    for i in range(1, len(is_downcast_series) - 1):
        if pd.isna(is_downcast_series.iloc[i]):
            # Look backwards for nearest non-null value
            prev_val = None
            for j in range(i - 1, max(-1, i - gap_size - 1), -1):
                if not pd.isna(is_downcast_series.iloc[j]):
                    prev_val = is_downcast_series.iloc[j]
                    break
            
            # Look forwards for nearest non-null value
            next_val = None
            for j in range(i + 1, min(len(is_downcast_series), i + gap_size + 1)):
                if not pd.isna(is_downcast_series.iloc[j]):
                    next_val = is_downcast_series.iloc[j]
                    break
            
            # Fill if both neighbors agree
            if prev_val is not None and next_val is not None and prev_val == next_val:
                is_downcast_series.iloc[i] = prev_val
    
    # If we still have too many None values, use fallback strategy
    non_null_count = is_downcast_series.count()
    total_count = len(is_downcast_series)
    
    if non_null_count > 0 and non_null_count < total_count * 0.5:  # Less than 50% classified
        # Use the most common classification for the whole profile
        most_common = is_downcast_series.mode()
        if len(most_common) > 0:
            default_value = most_common.iloc[0]
            print(f"Low classification rate ({non_null_count}/{total_count}). Using default: {default_value}")
            is_downcast_series = is_downcast_series.fillna(default_value)
    
    df['is_downcast'] = is_downcast_series

    # Final pass: propagate nearest known True/False to any remaining None sections
    # (covers flat segments missed by the small-gap filler and the 50% fallback)
    df['is_downcast'] = df['is_downcast'].ffill().bfill()
    # If the entire column is still None (e.g. no gradient at all), use explicit NaN
    if df['is_downcast'].isna().all():
        df['is_downcast'] = np.nan

    return df

def quality_check_ph(df: pd.DataFrame, ctd_type: str) -> pd.DataFrame:
    """Apply quality control to pH measurements by setting invalid values to NaN."""
    # Use direct column name first, else search mappings
    ph_col = 'ph'

    # desired valid pH range
    ph_min = 6.8
    ph_max = 8.6

    # Helper to apply filter on a given column name
    def _filter_ph_column(df_local: pd.DataFrame, col: str) -> pd.DataFrame:
        # Ensure numeric
        df_local[col] = pd.to_numeric(df_local[col], errors='coerce')
        
        # Count invalid values (excluding already NaN)
        invalid_mask = ~df_local[col].isna() & ((df_local[col] < ph_min) | (df_local[col] > ph_max))
        num_invalid = invalid_mask.sum()
        
        # Set invalid pH values to NaN (keep the rows, just invalidate the pH values)
        df_local.loc[invalid_mask, col] = np.nan
        
        if num_invalid > 0:
            print(f"Set {num_invalid} pH values in '{col}' to NaN (outside range [{ph_min}, {ph_max}]).")
        else:
            print(f"No pH values invalidated for column '{col}' (all within range [{ph_min}, {ph_max}]).")
        
        return df_local

    # Check if pH column exists - first try direct match
    if ph_col in df.columns:
        return _filter_ph_column(df, ph_col)

    # Try to find pH column using COLUMN_MAPPINGS if direct match failed
    ctd_type = ctd_type.lower()
    for raw_name, (std_name, _) in COLUMN_MAPPINGS[ctd_type].items():
        if 'ph' == std_name.lower():
            if raw_name in df.columns:
                print(f"Applying pH filter to column: {raw_name}")
                return _filter_ph_column(df, raw_name)

    # If no pH column found, return dataframe unchanged
    return df


def quality_check_oxygen(df: pd.DataFrame, ctd_type: str, 
                         o2_min: float = 0.0, 
                         o2_max: float = 200.0) -> pd.DataFrame:
    """
    Apply quality control to oxygen saturation measurements.
    Filter out unrealistic values that indicate sensor malfunction or air contamination.
    
    Natural oxygen saturation typically ranges from 80-110%, with extreme cases:
    - High productivity/algal blooms: up to ~130-140%
    - Values above 150% almost always indicate sensor errors or air bubbles
    - Values above 200% are physically unrealistic in natural waters
    
    Args:
        df: DataFrame containing CTD data
        ctd_type: Type of CTD instrument
        o2_min: Minimum acceptable O2 saturation (%) - default 0%
        o2_max: Maximum acceptable O2 saturation (%) - default 150%
                (Conservative threshold allowing extreme supersaturation)
    
    Returns:
        DataFrame with invalid oxygen values set to NaN (rows are preserved)
    """
    o2_col = 'oxygen_saturation_percent'
    o2_conc_col = "O2ppm"
    if o2_col not in df.columns:
        return df
    
    # Convert to numeric, handling any non-numeric entries
    df[o2_col] = pd.to_numeric(df[o2_col], errors='coerce')
    
    # Count values outside the range (excluding NaN)
    unrealistic_mask = df[o2_col].notna() & ((df[o2_col] < o2_min) | (df[o2_col] > o2_max))
    n_filtered = unrealistic_mask.sum()
    
    if n_filtered > 0:
        print(f"Warning: Setting {n_filtered} oxygen saturation values to NaN "
              f"(outside realistic range: {o2_min}-{o2_max}%)")
        
        # Set unrealistic values to NaN instead of dropping rows
        df.loc[unrealistic_mask, o2_col] = np.nan
        
        # Also invalidate corresponding concentration values if present
        if o2_conc_col in df.columns:
            df.loc[unrealistic_mask, o2_conc_col] = np.nan
    
    return df


def find_mld(temp, dens, depth, thresh_temp=0.2, thresh_dens=0.03, thresh_depth=1):
    """Calculate mixed layer depth using temperature and density criteria."""

    #Filter depth > 1, adjust temp, dens
    temp = temp[depth > thresh_depth]
    dens = dens[depth > thresh_depth]
    depth = depth[depth > thresh_depth]

    # Not enough data after depth filter
    if temp.empty or dens.empty:
        return pd.Series({'mld_temp': np.nan, 'mld_dens': np.nan})

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
        # Search for conductivity column variants directly in the dataframe
        conductivity_variants = ['conductivity_mS_per_m', 'conductivity_mS_per_cm', 'conductivity']
        for variant in conductivity_variants:
            if variant in df.columns:
                cond_col = variant
                has_conductivity = True
                print(f"Found conductivity column: {cond_col}")
                break
        
        # If still not found, try get_parameter_name as fallback
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
                #Filter depth_m < 0
                df = df[df['depth_m'] > 0].copy()

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
        #df['o2_solubility_mll'] = o2_sol_mll
        #df['o2_solubility_mgl'] = o2_sol_mgl
        
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
        
        # Check conductivity units and convert if needed
        if 'conductivity_mS_per_m' in cond_col:
            # Convert mS/m to mS/cm
            cond_mScm = df[cond_col] / 100.0
            print(f"Converting conductivity from mS/m to mS/cm for salinity calculation")
        elif 'conductivity_mS_per_cm' in cond_col:
            # Already in mS/cm
            cond_mScm = df[cond_col]
            print(f"Using conductivity in mS/cm for salinity calculation")
        else:
            # Assume mS/cm if unit is unclear
            cond_mScm = df[cond_col]
            print(f"Warning: Conductivity units unclear for column '{cond_col}', assuming mS/cm")
        
        conductivity_ratio = cond_mScm / 42.914  # Calculate conductivity ratio
        df[sal_col] = sw.salt(conductivity_ratio, df[temp_col], df[pres_col])
    elif ctd_type == "seabird" and not has_conductivity:
        print("Warning: Seabird-specific salinity calculation skipped - no conductivity data")

    return df