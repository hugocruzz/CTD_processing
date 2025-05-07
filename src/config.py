"""Configuration and constants for CTD processing."""

# Site configuration
SITE_CONFIG = {
    'LATITUDE': 60.0,
    'LONGITUDE': 60.0
}

# Mapping from instrument-specific column names to standard names with units
# Format: 'raw_name': ('standard_name_with_unit', unit_conversion_factor or None)
# If conversion_factor is None, no conversion is needed
COLUMN_MAPPINGS = {
    'idronaut': {
        'Pres': ('pressure_dbar', None),
        'Press': ('pressure_dbar', None),
        'Temp': ('temperature_C', None),
        'Cond': ('conductivity_mS_per_m', None),  # Already in mS/m
        'Sal': ('salinity_psu', None),
        'O2%': ('oxygen_saturation_percent', None),
        'O2ppm': ('oxygen_concentration_ml_per_L', 0.7),  # Convert ppm to ml/L
        'Date': ('date', None),
        'Time': ('time', None),
        "PAR": ('PAR_umol_m2_s', None),
    },
    
    'seabird': {
        'prdM': ('pressure_dbar', None),
        'depSM': ('depth_m', None),
        't090C': ('temperature_C', None),
        'c0S/m': ('conductivity_mS_per_m', 1000),  # Convert S/m to mS/m
        'sal00': ('salinity_psu', None),
        'sbeox0PS': ('oxygen_saturation_percent', None),
        'ph': ('ph', None),
        'turbWETntu0': ('turbidity_NTU', None),
        'flECO-AFL': ('fluorescence_mg_m3', None),
        'oxsatML/L': ('oxygen_concentration_ml_per_L', None),
        'scan': ('scan', None),
        'flag': ('flag', None)
    },
    
    'rbr': {
        'Time': ('timestamp', None),
        'Conductivity': ('conductivity_mS_per_m', None),
        'Temperature': ('temperature_C', None),
        'Pressure': ('pressure_dbar', None),
        'Temperature.1': ('temperature_secondary_C', None),
        'Dissolved O2 concentration': ('oxygen_concentration_ml_per_L', None),
        'Sea pressure': ('pressure_sea_dbar', None),
        'Depth': ('depth_m', None),
        'Salinity': ('salinity_psu', None),
        'Speed of sound': ('sound_speed_m_per_s', None),
        'Specific conductivity': ('conductivity_specific_mS_per_m', 0.1),  # µS/cm to mS/m
        'Dissolved O2 saturation': ('oxygen_saturation_percent', None)
    },
    
    'exo': {
        'TIME (HH:MM:SS)': ('time', None),
        'DATE (MM/DD/YYYY)': ('date', None),
        'COND µS/CM': ('conductivity_mS_per_m', 0.1),  # µS/cm to mS/m
        'SPCOND µS/CM': ('conductivity_specific_mS_per_m', 0.1),  # µS/cm to mS/m
        'SAL PSU': ('salinity_psu', None),
        'DEPTH M': ('depth_m', None),
        'PRESSURE PSI A': ('pressure_dbar', 0.689476),  # PSI to dbar
        'ODO % SAT': ('oxygen_saturation_percent', None),
        'ODO MG/L': ('oxygen_concentration_ml_per_L', 0.7),  # mg/L to ml/L (approximate)
        'PH': ('ph', None),
        'TEMP °C': ('temperature_C', None),
        'TURBIDITY FNU': ('turbidity_NTU', None)
    },
    'gf23': {
        'Depth': ('depth_m', None),
        'Temperature': ('temperature_C', None),
        'Conductivity': ('conductivity_mS_per_m', None),
        'Oxygen %': ('oxygen_saturation_percent', None),
        'Oxygen mg/L': ('oxygen_concentration_ml_per_L', None),
        'pH': ('ph', None),
        'PAR': ('PAR_umol_m2_s', None),
        'Salinity': ('salinity_psu', None),
        'SigmaT': ('sigma_t', None),
        'Trx-chl(a)': ('chlorophyll_rfu', None),
        'Pressure': ('pressure_dbar', None),
        'time In': ('time', None),  # Added for datetime integration
        'date': ('date', None)      # Added for datetime integration
    }
}

def get_standard_column_name(raw_name, ctd_type):
    """
    Get standardized column name for a raw column name.
    
    Args:
        raw_name: Raw column name from instrument
        ctd_type: Type of CTD ('idronaut', 'seabird', etc.)
        
    Returns:
        str: Standardized column name with unit
    """
    ctd_type = ctd_type.lower()
    
    # First try exact match
    if raw_name in COLUMN_MAPPINGS[ctd_type]:
        std_name, _ = COLUMN_MAPPINGS[ctd_type][raw_name]
        return std_name
    
    # Try case-insensitive match
    raw_lower = raw_name.lower()
    for raw, mapping in COLUMN_MAPPINGS[ctd_type].items():
        if raw.lower() == raw_lower:
            std_name, _ = mapping
            return std_name
    
    # If not found, return original name
    return raw_name

def get_column_mapping(ctd_type):
    """
    Get a dictionary mapping raw column names to standardized names.
    
    Args:
        ctd_type: Type of CTD ('idronaut', 'seabird', etc.)
    
    Returns:
        dict: Mapping of raw column names to standardized names
    """
    return {
        raw_name: get_standard_column_name(raw_name, ctd_type)
        for raw_name in COLUMN_MAPPINGS[ctd_type.lower()]
    }

def get_unit_conversion_factor(raw_name, ctd_type):
    """
    Get unit conversion factor for a raw column name.
    
    Args:
        raw_name: Raw column name from instrument
        ctd_type: Type of CTD ('idronaut', 'seabird', etc.)
    
    Returns:
        float or None: Conversion factor if needed, None if no conversion needed
    """
    ctd_type = ctd_type.lower()
    
    # First try exact match
    if raw_name in COLUMN_MAPPINGS[ctd_type]:
        return COLUMN_MAPPINGS[ctd_type][raw_name][1]
    
    # Try case-insensitive match
    raw_lower = raw_name.lower()
    for raw, mapping in COLUMN_MAPPINGS[ctd_type].items():
        if raw.lower() == raw_lower:
            return mapping[1]
    
    return None