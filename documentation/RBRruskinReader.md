# RBRruskinReader Documentation

## Overview

The `RBRruskinReader` class is a new addition to the CTD processing toolkit that enables reading RBR's native `.rsk` files using the official RBR Python package. This reader provides direct access to RBR instrument data without the need for format conversion.

## Prerequisites

### RBR Python Package Installation

The `RBRruskinReader` requires the official RBR Python package. Install it using:

```bash
pip install rbr
```

If the RBR package is not installed, the reader will raise an `ImportError` with installation instructions.

## Features

### Supported Data Types
- Temperature (°C)
- Conductivity (mS/m)
- Pressure (dbar)
- Depth (m)
- Salinity (PSU)
- Dissolved Oxygen (concentration and saturation)
- pH
- Turbidity
- Chlorophyll/Fluorescence
- PAR (Photosynthetically Available Radiation)
- CDOM
- Backscatter
- Sound Speed

### Automatic Unit Conversions
The reader automatically handles common unit conversions:
- Conductivity: mS/cm → mS/m (×100)
- Dissolved Oxygen: μmol/L → ml/L (÷44.661)
- Other parameters maintain their original units when appropriate

### Column Standardization
Raw RBR column names are mapped to standardized names consistent with other CTD readers:
- `Temperature` → `temperature_C`
- `Conductivity` → `conductivity_mS_per_m`
- `Pressure` → `pressure_dbar`
- `Depth` → `depth_m`
- etc.

## Usage

### Basic Usage

```python
from readers import RBRruskinReader

# Create reader instance
reader = RBRruskinReader('path/to/data.rsk', 'rbr_rsk')

# Read the data
df = reader.read()

# Display basic information
print(f"Data shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())
```

### Integration with Main Processing Pipeline

The reader is automatically integrated with the main processing pipeline. RSK files are detected by their `.rsk` extension:

```python
from main import process_ctd_file

# Process an RSK file
process_ctd_file(
    filepath='data/raw/instrument_data.rsk',
    ctd_type='rbr_rsk',  # Automatically detected
    data_dir='data/raw',
    Level1_output='data/Level1',
    Level2_output='data/Level2',
    Level2B_output='data/Level2B'
)
```

### Batch Processing

RSK files are included in batch processing operations:

```python
from main import process_all_files

process_all_files(
    directory='data/raw',
    Level1_output='data/Level1',
    Level2_output='data/Level2', 
    Level2B_output='data/Level2B'
)
```

## File Structure

### Input Files
- **Extension**: `.rsk`
- **Format**: RBR's native binary format
- **Content**: Multi-channel CTD data with metadata

### Output Files
- **Level 1**: Raw data with standardized column names
- **Level 2**: Processed data with oceanographic calculations
- **Level 2B**: Split upward/downward profiles (if enabled)

## Configuration

### Column Mappings

The reader uses the `rbr_rsk` configuration in `config.py`:

```python
'rbr_rsk': {
    'Time': ('timestamp', None),
    'Conductivity': ('conductivity_mS_per_m', None),
    'Temperature': ('temperature_C', None),
    'Pressure': ('pressure_dbar', None),
    # ... additional mappings
}
```

### Adding New Parameters

To add support for new RBR parameters:

1. Add the mapping to `config.py`:
```python
'rbr_rsk': {
    # Existing mappings...
    'New Parameter': ('new_parameter_unit', conversion_factor),
}
```

2. Add unit conversion if needed in `RBRruskinReader._apply_rbr_unit_conversions()`:
```python
conversions = {
    # Existing conversions...
    'new_parameter_unit': {'from_unit': 'original_unit', 'factor': conversion_factor},
}
```

## Error Handling

### Common Issues

1. **RBR Package Not Installed**
   ```
   ImportError: RBR package is required to read .rsk files. Install with: pip install rbr
   ```
   **Solution**: Install the RBR package using `pip install rbr`

2. **Corrupted RSK File**
   ```
   Error reading RSK file: [specific error message]
   ```
   **Solution**: Verify file integrity and try re-downloading from instrument

3. **No Data in RSK File**
   ```
   ValueError: No data found in RSK file: filename.rsk
   ```
   **Solution**: Ensure the RSK file contains actual measurement data

### Debugging

Enable verbose output by checking the console messages:
- Channel information: Lists detected channels and their units
- Unit conversions: Shows applied conversion factors
- File structure: Displays timestamp and data ranges

## Testing

Use the provided test script to verify functionality:

```bash
python test_rbr_rsk_reader.py
```

The test script will:
- Check RBR package availability
- Test class instantiation
- Process sample RSK files (if available)
- Verify error handling

## Comparison with RBRReader

| Feature | RBRReader | RBRruskinReader |
|---------|-----------|-----------------|
| File Format | Text files (.txt) | Binary files (.rsk) |
| Dependencies | Standard Python | RBR Python package |
| Metadata | External JSON files | Embedded in RSK |
| Data Integrity | Manual verification | Built-in validation |
| Performance | Good | Excellent |
| Feature Support | Limited | Full RBR feature set |

## Limitations

1. **Single Cast Support**: Currently processes the first cast from multi-cast RSK files
2. **RBR Package Dependency**: Requires additional package installation
3. **Binary Format**: Cannot be manually inspected like text files

## Future Enhancements

- Support for multi-cast RSK files
- Advanced metadata extraction
- Real-time processing capabilities
- Integration with RBR's quality control algorithms

## Support

For issues specific to:
- **RSK file format**: Contact RBR Support
- **Python package**: Check RBR's GitHub repository
- **This reader implementation**: Check the CTD processing repository