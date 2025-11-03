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
        'Cond': ('conductivity_mS_per_cm', None),  # Already in mS/m
        'Sal': ('salinity_psu', None),
        'O2%': ('oxygen_saturation_percent', None),
        'O2ppm': ('O2ppm',None),  # Convert ppm to ml/L
        'Date': ('date', None),
        'ph': ('ph', None),
        'Ph': ('ph', None),
        'pH': ('ph', None),
        'Time': ('time', None),
        "PAR": ('PAR_umol_m2_s', None),
        # Chlorophyll columns from Recover files
        'Trx-chl(a)': ('chlorophyll_rfu', None),
        'Phycocyanin': ('phycocyanin_rfu', None),
        'Phycoerythrin': ('phycoerythrin_rfu', None),
        'Trx-Chl-a': ('chlorophyll_rfu', None),  # Alternative capitalization
        "Pethr": ('phycoerythrin_rfu', None),  # Alternative capitalization
        "Phyc": ('phycocyanin_rfu', None),      # Alternative capitalization
        "Chl(a)": ('chlorophyll_rfu', None),    # Alternative capitalization
        "Phy-Ethrin": ('phycoerythrin_rfu', None),  # Alternative capitalization
        'TRX-Chl(a)': ('chlorophyll_rfu', None),
        "Phy-Cyanin": ('phycocyanin_rfu', None),      # Alternative capitalization
        'Pchan': ('phycocyanin_rfu', None),  # Alternative column names for Recover files
        'Pechan': ('phycoerythrin_rfu', None),  # Alternative
        # Alternative column names for Recover files
        'Pressure': ('pressure_dbar', None),
    },
    
    'seabird': {
        'prdM': ('pressure_dbar', None),
        'depSM': ('depth_m', None),
        't090C': ('temperature_C', None),
        'c0S/m': ('conductivity_mS_per_cm', 10),  # Convert S/m to mS/m
        'sal00': ('salinity_psu', None),
        'sbeox0PS': ('oxygen_saturation_percent', None),
        'ph': ('ph', None),
        'Ph': ('ph', None),
        'pH': ('ph', None),
        'turbWETntu0': ('turbidity_NTU', None),
        'flECO-AFL': ('fluorescence_mg_m3', None),
        'oxsatML/L': ('oxygen_saturated_ml_per_L', None),
        'scan': ('scan', None),
        'flag': ('flag', None),
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
        'Dissolved O2 saturation': ('oxygen_saturation_percent', None),
        
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
        'ODO MG/L': ('oxygen_concentration_mg_per_L',1),  # mg/L to ml/L (approximate)
        'PH': ('ph', None),
        'TEMP °C': ('temperature_C', None),
        'TURBIDITY FNU': ('turbidity_NTU', None)
    },
    'gf23': {
        'Depth': ('depth_m', None),
        'Temperature': ('temperature_C', None),
        'Conductivity': ('conductivity_mS_per_cm', None),
        'Oxygen %': ('oxygen_saturation_percent', None),
        'Oxygen mg/L': ('oxygen_concentration_mg_per_L', None),
        'pH': ('ph', None),
        'PAR': ('PAR_umol_m2_s', None),
        'Salinity': ('salinity_psu', None),
        'SigmaT': ('sigma_t', None),
        'Trx-chl(a)': ('chlorophyll_rfu', None),
        'Pressure': ('pressure_dbar', None),
        'time In': ('time', None),  # Added for datetime integration
        'date': ('date', None),      # Added for datetime integration
        # Chlorophyll columns from Recover files
        'Phycocyanin': ('phycocyanin_rfu', None),
        'Phycoerythrin': ('phycoerythrin_rfu', None),
    },
    
    'rbr_rsk': {
        # Raw column names from RSK files -> standardized names with conversion factors
        # Based on rsk.printchannels() output and actual data columns:
        # conductivity: mS/cm -> mS/m (multiply by 100)
        # temperature: °C -> °C (no conversion)
        # pressure: dbar -> dbar (no conversion)
        # sea_pressure: dbar -> dbar (no conversion)  
        # depth: m -> m (no conversion)
        # salinity: PSU -> PSU (no conversion)
        # speed_of_sound: m/s -> m/s (no conversion)
        # specific_conductivity: µS/cm -> mS/m (multiply by 0.1)
        
        # Time and datetime columns
        'Time': ('timestamp', None),
        'timestamp': ('timestamp', None),
        
        # Temperature columns
        'conductivity': ('conductivity_mS_per_m', 100),  # mS/cm to mS/m
        'conductivity_mS_per_m': ('conductivity_mS_per_m', None),  # Already standardized
        'temperature': ('temperature_C', None),  # Already in °C
        'temperature_C': ('temperature_C', None),  # Already standardized
        'temperature1': ('temperature1', None),  # Secondary temperature sensor
        
        # Pressure and depth columns
        'pressure': ('pressure_dbar', None),  # Already in dbar
        'pressure_dbar': ('pressure_dbar', None),  # Already standardized
        'sea_pressure': ('sea_pressure_dbar', None),  # Already in dbar
        'sea_pressure_dbar': ('sea_pressure_dbar', None),  # Already standardized
        'depth': ('depth_m', None),  # Already in m
        'depth_m': ('depth_m', None),  # Already standardized
        
        # Salinity columns
        'salinity': ('salinity_psu', None),  # Already in PSU
        'salinity_psu': ('salinity_psu', None),  # Already standardized
        
        # Sound speed columns
        'speed_of_sound': ('speed_of_sound_m_per_s', None),  # Already in m/s
        'speed_of_sound_m_per_s': ('speed_of_sound_m_per_s', None),  # Already standardized
        
        # Conductivity-related columns
        'specific_conductivity': ('specific_conductivity_mS_per_m', 0.1),  # µS/cm to mS/m
        'specific_conductivity_mS_per_m': ('specific_conductivity_mS_per_m', None),  # Already standardized
        
        # Dissolved oxygen columns
        'dissolved_o2_concentration': ('dissolved_o2_concentration', None),
        'dissolved_o2_saturation': ('oxygen_saturation_percent', None),  # Map to standard oxygen saturation
        'oxygen_saturation_percent': ('oxygen_saturation_percent', None),  # Already standardized
        
        # Instrument and metadata columns
        'instrument': ('instrument', None),
        'Instrument': ('Instrument', None),  # Keep original capitalization
        'Cast_name': ('Cast_name', None),
        
        # Alternative capitalized versions that might appear
        'Conductivity': ('conductivity_mS_per_m', 100),
        'Temperature': ('temperature_C', None),
        'Pressure': ('pressure_dbar', None),
        'Sea_pressure': ('sea_pressure_dbar', None),
        'Depth': ('depth_m', None),
        'Salinity': ('salinity_psu', None),
        'Speed_of_sound': ('speed_of_sound_m_per_s', None),
        'Specific_conductivity': ('specific_conductivity_mS_per_m', 0.1),
        
        # Additional dissolved oxygen variations
        'Dissolved_o2_concentration': ('dissolved_o2_concentration', None),
        'Dissolved_o2_saturation': ('oxygen_saturation_percent', None),
    },
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