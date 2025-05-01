"""Configuration and constants for CTD processing."""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Parameter:
    """Base parameter configuration."""
    unit: str
    standard_name: str
    
    def get_standard_format(self) -> str:
        """Return standardized parameter name with unit."""
        return f"{self.standard_name}_{self.unit}"

@dataclass
class CTDParameters:
    """CTD parameter mappings configuration."""
    raw_name: str
    parameter: Parameter = None
    standard_name: str = None
    
    def get_mapped_name(self) -> str:
        """Get the standardized parameter name."""
        if self.parameter:
            return self.parameter.get_standard_format()
        return self.standard_name

# Site configuration
SITE_CONFIG = {
    'LATITUDE': 60.0,
    'LONGITUDE': 60.0
}

# Base parameters definitions
BASE_PARAMS = {
    'pressure': Parameter(unit='dbar', standard_name='pressure'),
    'temperature': Parameter(unit='C', standard_name='temperature'),
    'conductivity': Parameter(unit='mS_per_m', standard_name='conductivity'), #Check if mS/m or S/m
    'salinity': Parameter(unit='psu', standard_name='salinity'),
    'oxygen_saturation': Parameter(unit='percent', standard_name='oxygen_saturation')
}

# CTD-specific parameter mappings
CTD_MAPPINGS = {
    'idronaut': {
        'Pres': CTDParameters(raw_name='Pres', parameter=BASE_PARAMS['pressure']),
        'Press': CTDParameters(raw_name='Press', parameter=BASE_PARAMS['pressure']),
        'Temp': CTDParameters(raw_name='Temp', parameter=BASE_PARAMS['temperature']),
        'Cond': CTDParameters(raw_name='Cond', parameter=BASE_PARAMS['conductivity']),
        'Sal': CTDParameters(raw_name='Sal', parameter=BASE_PARAMS['salinity']),
        'O2%': CTDParameters(raw_name='O2%', parameter=BASE_PARAMS['oxygen_saturation']),
        'O2ppm': CTDParameters(raw_name='O2ppm', standard_name='oxygen_concentration_ppm')
    },
    'seabird': {
        'prdM': CTDParameters(raw_name='prdM', parameter=BASE_PARAMS['pressure']),
        'depSM': CTDParameters(raw_name='depSM', standard_name='depth_m'),
        't090C': CTDParameters(raw_name='t090C', parameter=BASE_PARAMS['temperature']),
        'c0S/m': CTDParameters(raw_name='c0S/m', parameter=BASE_PARAMS['conductivity']),
        'sal00': CTDParameters(raw_name='sal00', parameter=BASE_PARAMS['salinity']),
        'sbeox0PS': CTDParameters(raw_name='sbeox0PS', parameter=BASE_PARAMS['oxygen_saturation']),
        'ph': CTDParameters(raw_name='ph', standard_name='ph'),
        'turbWETntu0': CTDParameters(raw_name='turbWETntu0', standard_name='turbidity_NTU'),
        'wetCDOM': CTDParameters(raw_name='wetCDOM', standard_name='cdom_mg_m3'),
        'flECO-AFL': CTDParameters(raw_name='flECO-AFL', standard_name='fluorescence_mg_m3'),
        'oxsatML/L': CTDParameters(raw_name='oxsatML/L', standard_name='oxygen_saturation_ml_L'),
        'scan': CTDParameters(raw_name='scan', standard_name='scan'),
        'flag': CTDParameters(raw_name='flag', standard_name='flag')
    },
    'rbr': {
        # Basic parameters with spaces as in the raw file
        'Time': CTDParameters(raw_name='Time', standard_name='timestamp'),
        'Conductivity': CTDParameters(raw_name='Conductivity', parameter=BASE_PARAMS['conductivity']),
        'Temperature': CTDParameters(raw_name='Temperature', parameter=BASE_PARAMS['temperature']),
        'Pressure': CTDParameters(raw_name='Pressure', parameter=BASE_PARAMS['pressure']),
        'Temperature.1': CTDParameters(raw_name='Temperature.1', standard_name='secondary_temperature_C'),
        'Dissolved O2 concentration': CTDParameters(raw_name='Dissolved O2 concentration', standard_name='oxygen_concentration_ml_L'),
        'Sea pressure': CTDParameters(raw_name='Sea pressure', standard_name='sea_pressure_dbar'),
        'Depth': CTDParameters(raw_name='Depth', standard_name='depth_m'),
        'Salinity': CTDParameters(raw_name='Salinity', parameter=BASE_PARAMS['salinity']),
        'Speed of sound': CTDParameters(raw_name='Speed of sound', standard_name='sound_speed_m_s'),
        'Specific conductivity': CTDParameters(raw_name='Specific conductivity', standard_name='specific_conductivity_uS_cm'),
        'Dissolved O2 saturation': CTDParameters(raw_name='Dissolved O2 saturation', parameter=BASE_PARAMS['oxygen_saturation']),
        'Density anomaly': CTDParameters(raw_name='Density anomaly', standard_name='density_anomaly_kg_m3'),
    },
    'exo': {
        'TIME (HH:MM:SS)': CTDParameters(raw_name='TIME (HH:MM:SS)', standard_name='time_hh_mm_ss'),
        'DATE (MM/DD/YYYY)': CTDParameters(raw_name='DATE (MM/DD/YYYY)', standard_name='date_mm_dd_yyyy'),
        'FILE NAME': CTDParameters(raw_name='FILE NAME', standard_name='file_name'),
        'SITE NAME': CTDParameters(raw_name='SITE NAME', standard_name='site_name'),
        'USER ID': CTDParameters(raw_name='USER ID', standard_name='user_id'),
        'FAULT CODE': CTDParameters(raw_name='FAULT CODE', standard_name='fault_code'),
        'TAL PC RFU': CTDParameters(raw_name='TAL PC RFU', standard_name='tal_pc_rfu'),
        'CHLOROPHYLL RFU': CTDParameters(raw_name='CHLOROPHYLL RFU', standard_name='chlorophyll_rfu'),        
        'COND µS/CM': CTDParameters(raw_name='COND µS/CM', parameter=BASE_PARAMS['conductivity']),
        'SPCOND µS/CM': CTDParameters(raw_name='SPCOND µS/CM', standard_name='specific_conductivity_uS_per_cm'),
        'TDS MG/L': CTDParameters(raw_name='TDS MG/L', standard_name='total_dissolved_solids_mg_per_L'),
        'SAL PSU': CTDParameters(raw_name='SAL PSU', parameter=BASE_PARAMS['salinity']),
        'NLF COND µS/CM': CTDParameters(raw_name='NLF COND µS/CM', standard_name='nlf_conductivity_uS_per_cm'),
        'DEPTH M': CTDParameters(raw_name='DEPTH M', standard_name='depth_m'),
        'VERTICAL POSITION M': CTDParameters(raw_name='VERTICAL POSITION M', standard_name='vertical_position_m'),
        'PRESSURE PSI A': CTDParameters(raw_name='PRESSURE PSI A', parameter=BASE_PARAMS['pressure']),
        'ODO % SAT': CTDParameters(raw_name='ODO % SAT', parameter=BASE_PARAMS['oxygen_saturation']),
        'ODO MG/L': CTDParameters(raw_name='ODO MG/L', standard_name='oxygen_concentration_mg_per_L'),
        'ODO % CB': CTDParameters(raw_name='ODO % CB', standard_name='oxygen_saturation_corrected_percent'),
        'NH4+ -N MG/L': CTDParameters(raw_name='NH4+ -N MG/L', standard_name='ammonium_nitrogen_mg_per_L'),
        'NH4+ -N MV': CTDParameters(raw_name='NH4+ -N MV', standard_name='ammonium_nitrogen_mv'),
        'NH3 MG/L': CTDParameters(raw_name='NH3 MG/L', standard_name='ammonia_mg_per_L'),
        'NO3 -N MG/L': CTDParameters(raw_name='NO3 -N MG/L', standard_name='nitrate_nitrogen_mg_per_L'),
        'NO3 -N MV': CTDParameters(raw_name='NO3 -N MV', standard_name='nitrate_nitrogen_mv'),
        'PH': CTDParameters(raw_name='PH', standard_name='ph'),
        'PH MV': CTDParameters(raw_name='PH MV', standard_name='ph_mv'),
        'CABLE PWR V': CTDParameters(raw_name='CABLE PWR V', standard_name='cable_power_voltage'),
        'BATTERY V': CTDParameters(raw_name='BATTERY V', standard_name='battery_voltage'),
        'TEMP °C': CTDParameters(raw_name='TEMP °C', parameter=BASE_PARAMS['temperature']),
        'TURBIDITY FNU': CTDParameters(raw_name='TURBIDITY FNU', standard_name='turbidity_fnu'),
        'TSS MG/L': CTDParameters(raw_name='TSS MG/L', standard_name='total_suspended_solids_mg_per_L'),
    }
}

def get_column_mapping(ctd_type: str) -> Dict[str, str]:
    """
    Generate column mapping dictionary for a specific CTD type.
    
    Args:
        ctd_type: Type of CTD ('idronaut' or 'seabird')
        
    Returns:
        Dictionary mapping raw column names to standardized names
    """
    return {
        param.raw_name: param.get_mapped_name() 
        for param in CTD_MAPPINGS[ctd_type.lower()].values()
    }

# Generate the column mappings (backwards compatible with existing code)
COLUMN_MAPPING = {
    ctd_type: get_column_mapping(ctd_type)
    for ctd_type in CTD_MAPPINGS.keys()
}