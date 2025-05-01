import pandas as pd
import numpy as np
import glob
import os
import plotly.subplots as sp
import plotly.graph_objects as go
import gsw
import scipy.signal as signal
import re
LATITUDE = 60.0  # Latitude of the sampling location
LONGITUDE = 60.0  # Longitude of the sampling location
# Define base parameters common to both CTD types
BASE_PARAMS = {
    'pressure': {'unit': 'dbar', 'standard_name': 'pressure'},
    'temperature': {'unit': 'C', 'standard_name': 'temperature'},
    'conductivity': {'unit': 'S_per_m', 'standard_name': 'conductivity'},
    'salinity': {'unit': 'psu', 'standard_name': 'salinity'},
    'oxygen_saturation': {'unit': 'percent', 'standard_name': 'oxygen_saturation'}
}

# Define CTD-specific mappings
COLUMN_MAPPING = {
    'idronaut': {
        'Pres': f"{BASE_PARAMS['pressure']['standard_name']}_{BASE_PARAMS['pressure']['unit']}",
        'Temp': f"{BASE_PARAMS['temperature']['standard_name']}_{BASE_PARAMS['temperature']['unit']}",
        'Cond': f"{BASE_PARAMS['conductivity']['standard_name']}_mS_per_cm",
        'Sal': f"{BASE_PARAMS['salinity']['standard_name']}_{BASE_PARAMS['salinity']['unit']}",
        'O2%': f"{BASE_PARAMS['oxygen_saturation']['standard_name']}_{BASE_PARAMS['oxygen_saturation']['unit']}",
        'O2ppm': 'oxygen_concentration_ppm'
    },
    'seabird': {
        'prdM': f"{BASE_PARAMS['pressure']['standard_name']}_{BASE_PARAMS['pressure']['unit']}",
        'depSM': 'depth_m',
        't090C': f"{BASE_PARAMS['temperature']['standard_name']}_{BASE_PARAMS['temperature']['unit']}",
        'c0S/m': f"{BASE_PARAMS['conductivity']['standard_name']}_{BASE_PARAMS['conductivity']['unit']}",
        'sal00': f"{BASE_PARAMS['salinity']['standard_name']}_{BASE_PARAMS['salinity']['unit']}",
        'oxsatML/L': 'oxygen_saturation_ml_L',
        'sbeox0PS': f"{BASE_PARAMS['oxygen_saturation']['standard_name']}_{BASE_PARAMS['oxygen_saturation']['unit']}",
        'ph': 'ph',
        'turbWETntu0': 'turbidity_NTU',
        'wetCDOM': 'cdom_mg_m3',
        'flECO-AFL': 'fluorescence_mg_m3',
        'scan': 'scan',
        'flag': 'flag'
    }
}

# Function to get standardized column name
def get_standard_name(param_type, ctd_type):
    """Get standardized column name based on parameter type and CTD type"""
    if param_type in BASE_PARAMS:
        return f"{BASE_PARAMS[param_type]['standard_name']}_{BASE_PARAMS[param_type]['unit']}"
    return None

def saturation_to_ppm(percent_saturation, C_sat):
    """
    Convert percentage saturation of O2 to ppm (mg/L).

    Parameters:
    - percent_saturation (float or array): Percentage saturation of O2.
    - C_sat (float or array): Oxygen solubility at the given temperature and pressure (mg/L).

    Returns:
    - float or array: O2 concentration in ppm (mg/L).
    """
    return (percent_saturation / 100) * C_sat

def o2_saturation(temp_c, pressure_hpa):
    """
    Calculate oxygen saturation concentration (C_sat) in mg/L based on temperature and pressure.

    Parameters:
    - temp_c (float): Temperature in degrees Celsius.
    - pressure_hpa (float): Atmospheric pressure in hPa.

    Returns:
    - float: Oxygen saturation concentration in mg/L.
    """
    # Constants for oxygen solubility calculation (Weiss 1970)
    A = -173.4292
    B = 249.6339
    C = 143.3483
    D = -21.8492

    # Convert temperature to Kelvin
    temp_k = temp_c + 273.15

    # Oxygen solubility at 1 atm in mg/L
    C_sat = np.exp(A + B * (100 / temp_k) + C * np.log(temp_k / 100) + D * (temp_k / 100))

    # Adjust for ambient pressure (hPa to atm)
    C_sat = C_sat * (pressure_hpa / 1013.25)
    return C_sat

def find_mld(temp, dens, depth, thresh_temp=0.2, thresh_dens=0.03):
    # Surface values
    temp_surf = temp.iloc[0]
    dens_surf = dens.iloc[0]
    
    # Find where difference exceeds threshold
    mld_temp = depth[abs(temp - temp_surf) > thresh_temp].iloc[0] if any(abs(temp - temp_surf) > thresh_temp) else np.nan
    mld_dens = depth[abs(dens - dens_surf) > thresh_dens].iloc[0] if any(abs(dens - dens_surf) > thresh_dens) else np.nan
    
    return pd.Series({'mld_temp': mld_temp, 'mld_dens': mld_dens})

class CTDReader:
    def __init__(self, filepath, ctd_type):
        self.filepath = filepath
        self.ctd_type = ctd_type
        self.column_mapping = COLUMN_MAPPING[ctd_type]
    
    def get_param_name(self, param_type):
        """Get parameter name based on CTD type"""
        for original, standard in self.column_mapping.items():
            if standard == get_standard_name(param_type, self.ctd_type):
                return original
        return None
        
    def standardize_columns(self, df):
        """Standardize column names using mapping"""
        mapping = COLUMN_MAPPING.get(self.ctd_type, {})
        return df.rename(columns=mapping)
        
    def get_column_name(self, param):
        """Get standardized column name"""
        return COLUMN_MAPPING[self.ctd_type].get(param)

class IdronautReader(CTDReader):
    """Reader for Idronaut CTD files (.txt)"""
    def read(self):
        df = pd.read_csv(self.filepath, delim_whitespace=True, skiprows=1)
        return self.standardize_columns(df)

class SeabirdReader(CTDReader):
    """Reader for Seabird CTD files (.cnv)"""
    def read(self):
        # Initialize column names dictionary
        column_names = {}
        skiprows = 0
        
        # Read header to get column names
        with open(self.filepath, 'r') as f:
            for i, line in enumerate(f):
                if line.startswith('# name'):
                    # Parse line like "# name 0 = scan: Scan Count"
                    parts = line.split('=')
                    col_index = int(parts[0].split()[2])
                    col_name = parts[1].split(':')[0].strip()
                    column_names[col_index] = col_name
                
                if '*END*' in line:
                    skiprows = i + 1
                    break
        
        # Read data with correct column names
        df = pd.read_csv(
            self.filepath, 
            skiprows=skiprows, 
            delim_whitespace=True,
            names=[column_names[i] for i in range(len(column_names))]
        )
        df = self.standardize_columns(df)
        return df

def clean_datetime_str(datetime_str):
    """Clean datetime string to standard format"""
    # Update pattern to handle various millisecond formats
    pattern = r'\.(\d{2})$'
    datetime_str = re.sub(pattern, r'\1', datetime_str)
    return datetime_str

def fix_seconds_format(datetime_str):
    """Fix 3-digit seconds format to standard 2-digit format"""
    parts = datetime_str.split(':')
    if len(parts) == 3:
        seconds_part = parts[2]
        # Check if seconds portion has 3 digits before decimal
        seconds_split = seconds_part.split('.')
        if len(seconds_split[0]) == 3:
            # Remove extra digit from seconds
            correct_seconds = seconds_split[0][1:]
            if len(seconds_split) > 1:
                # Reconstruct with milliseconds if present
                parts[2] = f"{correct_seconds}.{seconds_split[1]}"
            else:
                parts[2] = correct_seconds
    return ':'.join(parts)

def parse_datetime(date_str, time_str):
    """Parse datetime from various possible formats"""
    try:
        if isinstance(date_str, pd.Timestamp):
            return date_str
            
        datetime_str = f"{date_str}T{time_str}"
        datetime_str = fix_seconds_format(datetime_str)
        
        try:
            return pd.to_datetime(datetime_str, format='%d-%m-%YT%H:%M:%S.%f')
        except ValueError:
            return pd.to_datetime(datetime_str, format='%d-%m-%YT%H:%M:%S')
        
    except Exception as e:
        print(f"Error parsing datetime: {datetime_str}")
        raise e

def read_ctd_file(filepath, ctd_type):
    """Factory function to create appropriate reader"""
    readers = {
        'idronaut': IdronautReader,
        'seabird': SeabirdReader
    }
    
    reader_class = readers.get(ctd_type)
    if not reader_class:
        raise ValueError(f"Unsupported CTD type: {ctd_type}")
        
    reader = reader_class(filepath, ctd_type)
    return reader.read()

def create_profile_plot(df, filename):
    """Create interactive profile plot"""
    # Get parameters to plot (exclude Date, Time, Depth)
    params = [col for col in df.columns if col not in ['Date', 'Time', 'Depth']]
    
    # Calculate subplot layout
    n_params = len(params)
    n_cols = 3  # 3 columns
    n_rows = (n_params + n_cols - 1) // n_cols
    
    # Create subplots
    fig = sp.make_subplots(rows=n_rows, cols=n_cols, 
                          subplot_titles=params,
                          shared_yaxes=True)
    
    # Add traces
    for idx, param in enumerate(params):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        fig.add_trace(
            go.Scatter(x=df[param], y=df['Depth'], name=param),
            row=row, col=col
        )
        
        # Reverse y-axis (depth)
        fig.update_yaxes(autorange="reversed", title_text="Depth (m)")
    
    # Update layout
    fig.update_layout(height=300*n_rows, 
                     width=1000,
                     title_text=f"CTD Profile - {filename}",
                     showlegend=False)
    
    return fig
def calculate_ocean_params(df, ctd_type, latitude=46.4):
    """Calculate key oceanographic parameters with CTD-specific naming"""
    
    # Get standardized column names
    pressure_col = COLUMN_MAPPING[ctd_type]['Pres' if ctd_type == 'idronaut' else 'prdM']
    temp_col = COLUMN_MAPPING[ctd_type]['Temp' if ctd_type == 'idronaut' else 't090C']
    sal_col = COLUMN_MAPPING[ctd_type]['Sal' if ctd_type == 'idronaut' else 'sal00']
    o2_perc_col = COLUMN_MAPPING[ctd_type]['O2%' if ctd_type == 'idronaut' else 'sbeox0PS']
    depth_col = COLUMN_MAPPING[ctd_type]['depSM' if ctd_type == 'seabird' else 'Depth']
    # Convert pressure to absolute
    # Convert to numpy arrays and ensure correct types
    p = df[pressure_col].abs().to_numpy()
    sal = df[sal_col].to_numpy()
    temp = df[temp_col].to_numpy()
    # Calculate SA and CT first (needed for all calculations)
    SA = gsw.SA_from_SP(sal, p, LONGITUDE, LATITUDE)
    CT = gsw.CT_from_t(SA, temp, p)
    # Calculate oxygen parameters
    # 1. Get oxygen solubility in μmol/kg using GSW
    o2_sol_umol = gsw.O2sol(SA, CT, p, LONGITUDE, LATITUDE)
    
    # 2. Convert solubility to different units
    o2_sol_mll = o2_sol_umol * 0.022391  # μmol/kg to mL/L
    o2_sol_mgl = o2_sol_mll * 1.42905    # mL/L to mg/L
    
    # 3. Calculate concentrations from percent saturation
    percent_sat = df[o2_perc_col]
    df[f'o2_mll_{ctd_type}'] = (percent_sat / 100.0) * o2_sol_mll
    df[f'o2_mgl_{ctd_type}'] = (percent_sat / 100.0) * o2_sol_mgl
    
    # Store solubility values
    df[f'o2_sol_mll_{ctd_type}'] = o2_sol_mll
    df[f'o2_sol_mgl_{ctd_type}'] = o2_sol_mgl
    
    if depth_col not in df.columns:
        df[depth_col] = -1 * gsw.z_from_p(p, latitude, LONGITUDE, latitude)
        
    # Calculate potential temperature with suffix
    pot_temp_col = f'pot_temp_C'
    df[pot_temp_col] = gsw.pt_from_CT(SA, CT)
    
    density_col = f'density_kg_m3'
    # Absolute density
    df[density_col] = gsw.density.rho(SA, CT, p)
    # Calculate buoyancy frequency (N²) with suffix
    N2, pmid = gsw.Nsquared(
        df[sal_col],
        df[temp_col],
        p
    )
    
    # Pad N² array and add with suffix
    N2_padded = np.pad(N2, (0, 1), mode='constant', constant_values=np.nan)
    df[f'N2'] = N2_padded
    
    # Find mixed layer depth with suffix
    mld = find_mld(df[pot_temp_col], df[density_col], df[depth_col])
    df[f'mld_temp'] = mld['mld_temp']
    df[f'mld_dens'] = mld['mld_dens']
    
    return df
def process_ctd_file(filepath, ctd_type, ship):
    """Process CTD file based on type"""
    # Get standardized column names
    cond_col = COLUMN_MAPPING[ctd_type]['Cond' if ctd_type == 'idronaut' else 'c0S/m']
    pres_col = COLUMN_MAPPING[ctd_type]['Pres' if ctd_type == 'idronaut' else 'prdM']
    o2_col = COLUMN_MAPPING[ctd_type]['O2%' if ctd_type == 'idronaut' else 'sbeox0PS']
    depth_col = COLUMN_MAPPING[ctd_type]['depSM' if ctd_type == 'seabird' else 'Depth']
    
    df = read_ctd_file(filepath, ctd_type)
    
    # Air removal based on conductivity
    df_air = df[df[cond_col] < 1]
    ctd_pres_offset = df_air[pres_col].median()
    ctd_O2_offset = df_air[o2_col].median() - 100

    # Apply corrections
    df = df[df[cond_col] > 1]
    df[pres_col] = df[pres_col] - ctd_pres_offset
    df[o2_col] = df[o2_col] - ctd_O2_offset

    # Calculate all parameters including oxygen
    df = calculate_ocean_params(df, ctd_type, latitude=LATITUDE)
    
    # Find downcast using CTD-specific depth column
    max_depth_idx = df[depth_col].idxmax()
    df["is_downcast"] = df.index <= max_depth_idx
    
    # Quality check pH if present
    ph_col = COLUMN_MAPPING[ctd_type].get('pH' if ctd_type == 'idronaut' else 'ph')
    if ph_col and ph_col in df.columns:
        df.loc[(df[ph_col] < 6.5) | (df[ph_col] > 9), ph_col] = np.nan
    return df

def process_all_files(directory, ctd_type, ship):
    """Process all CTD files recursively in directory structure"""
    # File extension based on CTD type
    extension = 'txt' if ctd_type == 'idronaut' else 'cnv'
    
    # Recursive search pattern
    search_pattern = os.path.join(directory, f"**/*.{extension}")
    
    # Find all matching files recursively
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"No {extension} files found in {directory} or its subdirectories")
        return
        
    for file in files:
        print(f"Processing {file}")
        df = process_ctd_file(file, ctd_type, ship)
        
        # Save processed file with .csv extension
        if ctd_type == 'idronaut':
            file_name = file.split('\\')[-1].split('.')[0]
            output_path = file.replace('.txt', '.csv')
        else:
            base_path = os.path.splitext(file)[0].split(ship)[0]  # Remove original extension
            file_name = file.split('\\')[-1].split('.')[0]
            file_path = base_path + file_name
            output_path = file_path.replace('Level0', 'Level1') + '.csv'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

if __name__ == "__main__":
    ship = "Forel"
    CTD_type = "seabird"
    data_dir = fr"C:\Users\cruz\Documents\SENSE\CTD\data\Level0\{CTD_type}/{ship}"

    process_all_files(data_dir,CTD_type, ship)