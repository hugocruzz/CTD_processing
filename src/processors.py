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
    """Class to format CTD data into A2PS-compatible format."""

    def __init__(self, split_profile=True):
        self.split_profile = split_profile

    def format_folder(self, input_folder, output_folder):
        """
        Format all CTD files in a folder into A2PS-compatible format.

        Args:
            input_folder: Folder containing segmented CTD profiles.
            output_folder: Folder to save formatted files.
        """
        os.makedirs(output_folder, exist_ok=True)
        ctd_files = glob.glob(os.path.join(input_folder, "*.csv"))
        if not ctd_files:
            print(f"No CTD files found in {input_folder}")
            return

        for ctd_file in ctd_files:
            print(f"Formatting CTD file: {ctd_file}")
            self.format_file(ctd_file, output_folder)

    def format_file(self, ctd_file, output_folder):
        """
        Format a single CTD file into A2PS-compatible format.

        Args:
            ctd_file: Path to the CTD file.
            output_folder: Folder to save the formatted file.
        """
        ctd_df = pd.read_csv(ctd_file)
        ctd_ds = ctd_df.to_xarray()

        # Rename and set coordinate
        ctd_ds = ctd_ds.rename_vars({"pressure_dbar": "Pres"})
        ctd_ds = ctd_ds.swap_dims({'index': 'Pres'})
        ctd_ds = ctd_ds.set_coords('Pres')
        ctd_ds = ctd_ds.drop_vars('index')
        ctd_ds["Pres"] = ctd_ds["Pres"] * 1.45038  # Convert pressure to psi

        if self.split_profile:
            self._split_and_format(ctd_ds, ctd_file, output_folder)
        else:
            self._format_entire_profile(ctd_ds, ctd_file, output_folder)

    def _split_and_format(self, ctd_ds, ctd_file, output_folder):
        """Split the profile into downward and upward portions and format them."""
        max_pressure_idx = ctd_ds["Pres"].argmax()

        # Downward portion
        downward_ds = ctd_ds.isel(Pres=slice(None, max_pressure_idx.values))
        downward_df = self._prepare_dataframe(downward_ds)
        self._save_formatted_file(downward_df, ctd_file, output_folder, suffix="_downward_formatted.asc")

        # Upward portion
        upward_ds = ctd_ds.isel(Pres=slice(max_pressure_idx.values, None))
        if len(upward_ds.Pres) > 1:
            upward_df = self._prepare_dataframe(upward_ds)
            self._save_formatted_file(upward_df, ctd_file, output_folder, suffix="_upward_formatted.asc")

    def _format_entire_profile(self, ctd_ds, ctd_file, output_folder):
        """Format the entire profile as a single dataset."""
        formatted_df = self._prepare_dataframe(ctd_ds)
        self._save_formatted_file(formatted_df, ctd_file, output_folder, suffix="_formatted.asc")

    def _prepare_dataframe(self, ctd_ds):
        """Prepare the DataFrame for formatting."""
        ctd_ds = ctd_ds.rename_vars({
            "temperature_C": "Tv2C",
            "salinity_psu": "Sal2",
            "oxygen_saturation_percent": "Sbeox2PS",
            "Pres": "PrdE"
        })
        df = ctd_ds[["Tv2C", "Sal2", "Sbeox2PS", "PrdE"]].to_dataframe()
        df.reset_index(drop=True, inplace=True)
        df = df.drop_duplicates(subset='PrdE', keep='first')
        df.sort_values(by=["PrdE"], inplace=True)
        return df

    def _save_formatted_file(self, df, ctd_file, output_folder, suffix):
        """Save the formatted DataFrame to a file."""
        formatted_file = os.path.basename(ctd_file).replace(".csv", suffix)
        output_path = os.path.join(output_folder, formatted_file)
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
        'depth': 'depth_m',
        'ph': 'ph',
        'turbidity': 'turbidity_NTU',
        'PAR': 'PAR'
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
    par_col = 'PAR_umol_m2_s'
    # Check if columns exist, fall back to parameter lookup if not
    if cond_col not in df.columns:
        cond_col = get_parameter_name(ctd_type, 'conductivity', standardized=True)
    if pres_col not in df.columns:
        pres_col = get_parameter_name(ctd_type, 'pressure', standardized=True)
    if o2_col not in df.columns:
        o2_col = get_parameter_name(ctd_type, 'oxygen_saturation', standardized=True)
    if par_col not in df.columns:
        par_col = get_parameter_name(ctd_type, 'PAR', standardized=True)
    if not all([cond_col, pres_col, o2_col]):
        raise ValueError(
            f"Missing required columns for CTD type {ctd_type}\n"
            f"Looking for: conductivity_mS_per_m, pressure_dbar, oxygen_saturation_percent\n"
            f"Found columns: {df.columns.tolist()}"
        )
    
    # Add debug print
    print(f"Processing columns: {cond_col}, {pres_col}, {o2_col}")
    
    # Process data using standardized column names
    df_air = df[df[cond_col] < threshold_cond]
    
    if df_air.empty:
        print("No air data found, skipping corrections")
        return df
        
    ctd_pres_offset = df_air[pres_col].median()
    ctd_O2_offset = df_air[o2_col].median() - 100
    
    if np.abs(ctd_O2_offset) > 50:
        ctd_O2_offset = 0
        print("Error with offsetting Oxygen data with air, the oxygen in the air is badly measured")
        
    df = df[df[cond_col] > threshold_cond].copy()
    df[pres_col] = df[pres_col] - ctd_pres_offset
    df[o2_col] = df[o2_col] - ctd_O2_offset
    
    if df[o2_col].mean() < 0:
        print("Error: Negative mean oxygen saturation after correction")
        
    # Filter df[pres_col] < 0
    df = df[df[pres_col] > 0].copy()
    
    # Calculate PAR average in the air
    if par_col in df_air.columns:
        par_avg_air = df_air[par_col].mean()
        df['PAR_avg_air'] = par_avg_air
    else:
        print("PAR column is missing in the air data.")
        
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
    missing_cols = []
    for col in [pres_col, temp_col, cond_col, sal_col]:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"Warning: Missing columns {missing_cols}. Trying to find alternative columns.")
        for col in missing_cols:
            param_type = col.split('_')[0]  # Extract base parameter name
            alt_col = get_parameter_name(ctd_type, param_type, standardized=True)
            if alt_col:
                print(f"Using {alt_col} instead of {col}")
                if col == pres_col:
                    pres_col = alt_col
                elif col == temp_col:
                    temp_col = alt_col
                elif col == cond_col:
                    cond_col = alt_col
                elif col == sal_col:
                    sal_col = alt_col
    
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
    
    # Convert to numpy arrays
    p = df[pres_col].abs().to_numpy()
    sal = df[sal_col].to_numpy()
    temp = df[temp_col].to_numpy()
    cond = df[cond_col].to_numpy()
    
    SA = gsw.SA_from_SP(sal, p, SITE_CONFIG['LONGITUDE'], SITE_CONFIG['LATITUDE'])
    CT = gsw.CT_from_t(SA, temp, p)
    
    # Calculate derived parameters
    df['pot_temp_C'] = gsw.pt_from_CT(SA, CT)
    df['density_kg_m3'] = gsw.density.rho(SA, CT, p)
    if pres_col in df.columns and 'density_kg_m3' in df.columns:
        mld = find_mld(df[temp_col], df['density_kg_m3'], df[pres_col])
        df['mld_temp'] = mld['mld_temp']
        df['mld_dens'] = mld['mld_dens']
    else:
        print("Required columns for MLD calculation are missing.")

    # Add missing oxygen calculations
    o2_sol_umol = gsw.O2sol(SA, CT, p, SITE_CONFIG['LONGITUDE'], SITE_CONFIG['LATITUDE'])
    o2_sol_mll = o2_sol_umol * 0.022391  # μmol/kg to mL/L
    o2_sol_mgl = o2_sol_mll * 1.42905    # mL/L to mg/L
    
    # Store solubility values with CTD type suffix
    df[f'o2_solubility_mll'] = o2_sol_mll
    df[f'o2_solubility_mgl'] = o2_sol_mgl

    if o2_col in df.columns:
        # Calculate oxygen concentrations using vectorized operations
        df[f'o2_mgkg'] = df.apply(
            lambda row: calculate_oxygen_mgkg(
                row[temp_col], 
                row[sal_col], 
                row[o2_col]
            ), axis=1
        )
        #Essayer de retrouve rla meme valeur 
        df[f'o2_mgl'] = df.apply(
            lambda row: calculate_oxygen_mgl(
                row[temp_col], 
                row[sal_col], 
                row[o2_col]
            ), axis=1
        )

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
                
                # Assign N2 values to the DataFrame
                if len(N2) == len(pmid):  # Ensure lengths match
                    # Map N2 values to the closest pressure levels in the original DataFrame
                    for i, mid_p in enumerate(pmid):
                        closest_idx = (np.abs(p - mid_p)).argmin()  # Find the closest pressure index
                        df.at[closest_idx, 'N2'] = N2[i]
                
    except Exception as e:
        print(f"Warning: Error calculating N2: {e}")
        df['N2'] = np.nan

    if ctd_type=="seabird":
        #Rename sal_col into sal_col+"_seabird"
        df = df.rename(columns={sal_col: sal_col+"_seabird"})
        df[sal_col] = sw.salt(df[cond_col]/42.914, df[temp_col], df[pres_col]) #This equation is the one used by the Idronaut CTD

    return df

