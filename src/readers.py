"""CTD data readers for different file formats."""
import re
import os
import json
import glob

import pandas as pd
from config import CTD_MAPPINGS, get_column_mapping
# Function to get standardized column name
class BaseReader:
    """Base class for CTD readers."""
    
    def __init__(self, filepath: str, reader_type: str):
        self.filepath = filepath
        self.reader_type = reader_type
    
    def read(self) -> pd.DataFrame:
        """Read CTD data."""
        raise NotImplementedError("Subclass must implement abstract method")
    

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names based on CTD type.
        
        Args:
            df: DataFrame with raw column names
            
        Returns:
            DataFrame with standardized column names
        """
        try:
            # Get column mapping for this reader type
            column_mapping = get_column_mapping(self.reader_type)
            
            # Create a dictionary to map raw column names to standardized names
            rename_dict = {}
            
            for col in df.columns:
                # First try exact match
                if col in column_mapping:
                    rename_dict[col] = column_mapping[col]
                else:
                    # Try case-insensitive match
                    col_lower = col.lower()
                    for raw_name, std_name in column_mapping.items():
                        if raw_name.lower() == col_lower:
                            rename_dict[col] = std_name
                            break
            
            # Apply renaming if any mappings were found
            if rename_dict:
                df = df.rename(columns=rename_dict)
            else:
                print(f"No column mappings found for {self.reader_type}")
            
            return df
        
        except Exception as e:
            print(f"Error standardizing columns: {e}")
            # Return original dataframe if standardization fails
            return df

    
class IdronautReader(BaseReader):
    """Reader for Idronaut CTD files (.txt)"""
    def read(self) -> pd.DataFrame:
        try:
            # Read first few lines to check format
            with open(self.filepath, 'r') as f:
                first_lines = [next(f) for _ in range(3)]
            
            # Get headers from first line
            headers = first_lines[0].strip().split()
            
            # Check if second line contains units
            has_units = any(unit_marker in first_lines[1] 
                          for unit_marker in ['[', 'dbar', '°C', 'PSU'])
            
            if has_units:
                # Skip unit row but use headers from first row
                df = pd.read_csv(
                    self.filepath, 
                    delim_whitespace=True, 
                    skiprows=1,  # Skip only the units row
                    names=headers,  # Use headers from first row
                    index_col=False
                )
                # Remove the row containing units (it will be the first row)
                df = df.iloc[1:].reset_index(drop=True)
                
                print(f"Reading file with units header: {self.filepath}")
                
                # Store units information
                units = {}
                unit_line = first_lines[1].strip().split()
                for header, unit in zip(headers, unit_line):
                    if '[' in unit and ']' in unit:
                        units[header] = unit.strip('[]')
                self.units = units
                print(f"Detected units: {units}")
            else:
                # Regular format, read normally
                df = pd.read_csv(
                    self.filepath, 
                    delim_whitespace=True,
                    index_col=False
                )
                print(f"Reading file with standard header: {self.filepath}")
            #Convert all numerical values to float
            df = df.apply(pd.to_numeric, errors='ignore')
            #Find column where "Date" is present
            #If Date is not a column
            if "Date" not in df.columns:
                #Find column where "Date" is present
                date_col = [col for col in df.columns if "Date" in col]
                #Rename the column to "Date"
                df.rename(columns={date_col[0]: "Date"}, inplace=True)
            df["datetime"] = pd.to_datetime(df["Date"]+ " " + df["Time"])
            return self.standardize_columns(df)
            
        except Exception as e:
            print(f"Error reading file {self.filepath}: {str(e)}")
            raise
        
class SeabirdReader(BaseReader):
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
class ExoReader(BaseReader):
    """Reader for exo probe"""
    def __init__(self, filepath: str, reader_type: str):
        super().__init__(filepath, reader_type)
        self.units = {}
    def read(self) -> pd.DataFrame:
        """Read RBR CTD data from text file.
        
        Returns:
            pd.DataFrame: DataFrame with processed RBR CTD data
        """
        self.df = pd.read_csv(self.filepath, encoding="utf-16", delimiter=",", skiprows=9)
                
        # Convert conductivity from µS/CM to mS_per_m
        if 'COND µS/CM' in self.df.columns:
            self.df['COND µS/CM'] = self.df['COND µS/CM'] * 0.1  # µS/CM to mS_per_m
            self.df.rename(columns={'COND µS/CM': 'conductivity_mS_per_m'}, inplace=True)
            print("Converted 'COND µS/CM' to 'conductivity_mS_per_m'")
        
        #Standardize column names
        self.df = self.standardize_columns(self.df)
        return self.df
        
class RBRReader(BaseReader):
    """Reader for RBR CTD files."""
    
    def __init__(self, filepath: str, reader_type: str):
        super().__init__(filepath, reader_type)
        self.units = {}
        
    def read(self) -> pd.DataFrame:
        """Read RBR CTD data from text file.
        
        Returns:
            pd.DataFrame: DataFrame with processed RBR CTD data
        """
        # First, load the metadata file for units information
        self._find_and_load_metadata()
        
        # Read the file to determine structure
        with open(self.filepath, 'r') as file:
            lines = file.readlines()
        
        # Find the header line and data start
        header_line = None
        data_start = 0
        
        for i, line in enumerate(lines):
            # RBR files typically have headers that contain these fields
            if any(header in line for header in ['Temperature', 'Conductivity', 'Pressure']):
                header_line = line
                data_start = i + 1
                break
        
        # If we couldn't find standard headers, look for any header-like pattern
        if header_line is None:
            for i, line in enumerate(lines):
                # Look for tab or multi-space separated values that could be headers
                if re.match(r'^[\w\s\-\(\)]+(\t|[ ]{2,})[\w\s\-\(\)]+', line.strip()):
                    header_line = line
                    data_start = i + 1
                    break
        
        # If we still couldn't determine the header, default to line 8
        if header_line is None:
            header_line = lines[8] if len(lines) > 8 else ""
            data_start = 9
        
        # Process the header: split by tabs or multiple spaces
        headers = re.split(r'\t|[ ]{2,}', header_line.strip())
        headers = [h.strip() for h in headers if h.strip()]
        
        # Read data
        # Since RBR files can have varied formats, try different approaches
        try:
            # First try with auto-detection of separator
            df = pd.read_csv(self.filepath, sep=",")
            print(f"Successfully read {self.filepath} with auto-detection")
        except Exception as e:
            print(f"Failed to read with auto-detection: {e}")
            try:
                # Try with tab separator
                df = pd.read_csv(self.filepath, skiprows=data_start, names=headers, sep='\t')
                print(f"Successfully read {self.filepath} with tab separator")
            except Exception as e:
                print(f"Failed to read with tab separator: {e}")
                try:
                    # Last resort: try a fixed-width format
                    df = pd.read_fwf(self.filepath, skiprows=data_start, names=headers)
                    print(f"Successfully read {self.filepath} with fixed-width format")
                except Exception as e:
                    print(f"Failed to read {self.filepath} with all methods: {e}")
                    raise
        
        df = self.standardize_columns(df)
        
        # Convert all columns that should be numeric
        for col in df.columns:
            if col.lower() not in ['date', 'time', 'timestamp']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Process datetime columns
        self._process_datetime_columns(df)
        
        # Apply unit conversions based on metadata
        df = self._apply_unit_conversions(df)
        
        # Clean column names - replace spaces with underscores
        df.columns = [col.replace(' ', '_') for col in df.columns]
        
        return df
    
    def _process_datetime_columns(self, df):
        """Process and combine date/time columns if present"""
        # Check if there's a date/time column
        date_time_cols = [col for col in df.columns if any(t in col.lower() for t in ['time', 'date'])]
        
        if len(date_time_cols) >= 2:
            # Try to combine date and time columns
            date_col = next((col for col in date_time_cols if 'date' in col.lower()), None)
            time_col = next((col for col in date_time_cols if 'time' in col.lower() and 'date' not in col.lower()), None)
            
            if date_col and time_col:
                try:
                    df['timestamp'] = pd.to_datetime(df[date_col] + ' ' + df[time_col])
                    df.drop([date_col, time_col], axis=1, inplace=True)
                    print(f"Created timestamp from {date_col} and {time_col}")
                except Exception as e:
                    print(f"Failed to create timestamp: {e}")
        # If there's a single Time column in ISO format
        elif any('time' in col.lower() for col in df.columns) and not any('date' in col.lower() for col in df.columns):
            time_col = next((col for col in df.columns if 'time' in col.lower()), None)
            if time_col:
                try:
                    df['timestamp'] = pd.to_datetime(df[time_col])
                    if time_col != 'timestamp':  # Don't drop if it's already named timestamp
                        df.drop([time_col], axis=1, inplace=True)
                    print(f"Created timestamp from {time_col}")
                except Exception as e:
                    print(f"Failed to parse time column: {e}")
    
    def _find_and_load_metadata(self):
        """Find and load the associated metadata file."""
        try:
            # Get the base name of the data file (removing "_data.txt")
            data_file_name = os.path.basename(self.filepath)
            base_name = data_file_name.rsplit('_data', 1)[0]  # Split on last occurrence of "_data"
            
            # Build the metadata file path - specifically look for _metadata.txt file
            metadata_path = os.path.join(os.path.dirname(self.filepath), f"{base_name}_metadata.txt")
            
            # If .txt metadata file doesn't exist, try with .json extension
            if not os.path.exists(metadata_path):
                metadata_path = os.path.join(os.path.dirname(self.filepath), f"{base_name}_metadata.json")
            
            # If still not found, try just .json extension with base name
            if not os.path.exists(metadata_path):
                metadata_path = os.path.join(os.path.dirname(self.filepath), f"{base_name}.json")
            
            # If still not found, look for any JSON file in the directory
            if not os.path.exists(metadata_path):
                json_files = glob.glob(os.path.join(os.path.dirname(self.filepath), "*.json"))
                if json_files:
                    metadata_path = json_files[0]
                else:
                    print(f"No metadata file found for {self.filepath}")
                    return
            
            print(f"Found metadata file: {metadata_path}")
            
            # Load and parse the metadata file
            with open(metadata_path, 'r') as f:
                content = f.read()
                
                # Check if it's a text file that might contain JSON
                if metadata_path.endswith('.txt'):
                    try:
                        metadata = json.loads(content)
                    except json.JSONDecodeError:
                        print(f"Metadata file {metadata_path} is not in JSON format")
                        return
                else:
                    metadata = json.loads(content)
            
            # Extract units information
            if 'dataheader' in metadata:
                for header in metadata['dataheader']:
                    if 'name' in header and 'units' in header:
                        self.units[header['name']] = header['units']
                
                print(f"Loaded units from metadata: {self.units}")
            else:
                print(f"No dataheader found in metadata file {metadata_path}")
                
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    def _apply_unit_conversions(self, df):
        """Apply unit conversions based on metadata"""
        if not self.units:
            print("No units information available, skipping conversions")
            return df
        
        # Copy dataframe to avoid modifying during iteration
        df_converted = df.copy()
        
        # Apply conversions
        for col in df.columns:
            if col in self.units:
                unit = self.units[col]
                
                # Handle conductivity conversions
                if col == 'Conductivity' and unit == 'mS/cm':
                    # Convert from mS/cm to mS/m (CONFIG expects mS/m)
                    df_converted[col] = df[col] * 100
                    print(f"Converted {col} from {unit} to mS/m")
                
                # Handle dissolved oxygen conversions
                if col == 'Dissolved O2 concentration' and unit == 'umol/L':
                    # Convert from umol/L to ml/L (standard oceanographic unit)
                    # 1 ml/L = 44.661 μmol/L
                    df_converted[col] = df[col] / 44.661
                    print(f"Converted {col} from {unit} to ml/L")
                
                # Add more conversions as needed
                
        return df_converted