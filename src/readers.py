"""CTD data readers for different file formats."""
import re
import os
import json
import glob
import numpy as np

import pandas as pd
from config import COLUMN_MAPPINGS, get_column_mapping, get_unit_conversion_factor

import pyrsktools as rbr
# Function to get standardized column name
class BaseReader:
    """Base class for CTD readers."""
    
    def __init__(self, filepath: str, reader_type: str, campaign_name: str = None):
        self.filepath = filepath
        self.reader_type = reader_type
        self.campaign_name = campaign_name
    
    def read(self) -> pd.DataFrame:
        """Read CTD data."""
        raise NotImplementedError("Subclass must implement abstract method")
    

    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names based on CTD type and apply unit conversions.
        
        Args:
            df: DataFrame with raw column names
            
        Returns:
            DataFrame with standardized column names and units
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
                    # Apply unit conversion if needed
                    conversion_factor = get_unit_conversion_factor(col, self.reader_type)
                    if conversion_factor is not None:
                        df[col] = df[col] * conversion_factor
                        print(f"Applied conversion factor {conversion_factor} to column {col} to get {rename_dict[col]}")
                else:
                    # Try case-insensitive match
                    col_lower = col.lower()
                    for raw_name, std_name in column_mapping.items():
                        if raw_name.lower() == col_lower:
                            rename_dict[col] = std_name
                            # Apply unit conversion if needed
                            conversion_factor = get_unit_conversion_factor(raw_name, self.reader_type)
                            if conversion_factor is not None:
                                df[col] = df[col] * conversion_factor
                                print(f"Applied conversion factor {conversion_factor} to column {col} to get {rename_dict[col]}")
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


    def _convert_gps_to_decimal(self, coord: str) -> float:
        """Convert GPS coordinate string to decimal degrees."""
        if not isinstance(coord, str):
            return coord  # Return as is if not a string (e.g., already a float)
        try:
            # Clean the string: remove apostrophes and replace comma with dot
            coord_clean = re.sub(r"[’'‘]", "", coord).replace(",", ".")
            
            # Split into degrees and minutes
            if "°" in coord_clean:
                parts = coord_clean.split("°")
                degrees = float(parts[0])
                minutes = float(parts[1]) / 60
                return degrees + minutes
            else:
                return float(coord_clean) # Already in decimal
        except (ValueError, IndexError) as e:
            print(f"Could not convert coordinate '{coord}': {e}")
            return np.nan

    def _load_and_merge_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load and merge metadata from Excel file for specific campaigns."""
        if self.campaign_name == "GF24":
            metadata_path = r"C:\Users\cruz\Documents\SENSE\SubOcean\data\raw\GF24/20240821_Metadata_CTD_greenfjord_2024.xlsx"
            if not os.path.exists(metadata_path):
                print(f"Metadata file not found at {metadata_path}")
                return df

            try:
                metadata_df = pd.read_excel(metadata_path)
                
                # Get the base name of the current CTD file without extension
                base_filename = os.path.splitext(os.path.basename(self.filepath))[0]

                # Find the matching row in the metadata
                # The 'cast name' column in the Excel file might have different names
                cast_name_col = None
                for col in metadata_df.columns:
                    if 'cast name' in col.lower():
                        cast_name_col = col
                        break
                
                if not cast_name_col:
                    print("Could not find 'cast name' column in metadata.")
                    return df

                # Ensure 'cast name' column is string type for comparison
                metadata_df[cast_name_col] = metadata_df[cast_name_col].astype(str)

                # Find the row where 'cast name' matches the base filename
                matched_row = metadata_df[metadata_df[cast_name_col] == base_filename]

                if not matched_row.empty:
                    # Get the first matched row as a series
                    metadata_series = matched_row.iloc[0]
                    print(f"Found metadata for {base_filename}")
                    # Add each piece of metadata as a new column in the dataframe
                    for key, value in metadata_series.items():
                        if 'lat' in key.lower() or 'lon' in key.lower():
                            if 'lat' in key.lower():
                                df[key] = self._convert_gps_to_decimal(value)
                            else:
                                df[key] = -self._convert_gps_to_decimal(value)
                        else:
                            df[key] = value
                    
                    # Create standardized Latitude and Longitude columns
                    lat_col_in = next((col for col in metadata_series.index if 'lat in' in col.lower()), None)
                    lon_col_in = next((col for col in metadata_series.index if 'long in' in col.lower()), None)

                    if lat_col_in:
                        df['Latitude'] = self._convert_gps_to_decimal(metadata_series[lat_col_in])
                    if lon_col_in:
                        df['Longitude'] = -self._convert_gps_to_decimal(metadata_series[lon_col_in])
                else:
                    print(f"No metadata found for {base_filename}")

            except Exception as e:
                print(f"Error loading or merging metadata: {e}")

        return df

class IdronautReader(BaseReader):
    """Reader for Idronaut CTD files (.txt)"""
    
    def find_recover_file(self) -> str:
        """
        Find a Recover file in the same directory as the Idronaut file.
        
        Returns:
            str: Path to the Recover file if found, None otherwise
        """
        directory = os.path.dirname(self.filepath)
        filename = os.path.basename(self.filepath)
        
        # Look for files containing "Recover" in the same directory
        recover_patterns = [
            "*Recover*.txt",
            "*recover*.txt", 
            "*RECOVER*.txt"
        ]
        
        for pattern in recover_patterns:
            recover_files = glob.glob(os.path.join(directory, pattern))
            if recover_files:
                # If multiple files found, try to match by timestamp/date
                base_timestamp = self._extract_timestamp_from_filename(filename)
                if base_timestamp:
                    for recover_file in recover_files:
                        recover_timestamp = self._extract_timestamp_from_filename(os.path.basename(recover_file))
                        if recover_timestamp and recover_timestamp == base_timestamp:
                            return recover_file
                # If no timestamp match, return the first one
                return recover_files[0]
        
        return None
    
    def _extract_timestamp_from_filename(self, filename: str) -> str:
        """Extract timestamp from filename (format: YYYYMMDD_HHMM)"""
        import re
        pattern = r'(\d{8}_\d{4})'
        match = re.search(pattern, filename)
        return match.group(1) if match else None
    
    def read_recover_file(self, recover_filepath: str) -> pd.DataFrame:
        """
        Read a Recover file and return DataFrame with pressure and chlorophyll data.
        Handles the special format with ** delimiters.
        
        Args:
            recover_filepath: Path to the Recover file
            
        Returns:
            pd.DataFrame: DataFrame containing Pressure and chlorophyll columns
        """
        try:
            print(f"Reading Recover file: {recover_filepath}")
            
            # Read the header from line 2 (index 1)
            with open(recover_filepath, 'r') as f:
                lines = f.readlines()
            
            if len(lines) < 3:
                print("Recover file has insufficient lines")
                return None
            
            # Get headers from second line, removing 'Parameter List' prefix
            header_line = lines[1].strip()
            if 'Parameter List' in header_line:
                # Find where 'List' ends and extract everything after
                list_end = header_line.find('List') + 4  # 'List' is 4 characters
                header_line = header_line[list_end:].strip()
            
            # Split headers by spaces
            headers = header_line.split()
            print(f"Found headers: {headers}")
            
            # Process data lines manually since ** format is only in data, not headers
            # Skip first row (cast name), use headers from line 2, skip line 3 (None)
            # Process data lines starting from line 4 (index 3)
            processed_data = []
            for i in range(3, len(lines)):  # Start from line 4 (index 3)
                line = lines[i].strip()
                if line and '**' in line:
                    # Split by ** and clean each part
                    parts = line.split('**')
                    cleaned_parts = []
                    for part in parts:
                        cleaned = part.strip()
                        if cleaned:  # Keep non-empty values, including '0'
                            cleaned_parts.append(cleaned)
                    
                    # Only keep rows that have the expected number of columns
                    if len(cleaned_parts) >= len(headers):
                        processed_data.append(cleaned_parts[:len(headers)])
            
            if not processed_data:
                print("No valid data found in Recover file after processing")
                return None
            
            # Create DataFrame with proper headers
            df = pd.DataFrame(processed_data, columns=headers)
            print(f"Created DataFrame with shape: {df.shape}")
            print(f"Headers: {list(df.columns)}")
            
            # Remove any completely empty columns or rows
            df = df.dropna(how='all', axis=1).dropna(how='all', axis=0)
            
            if df.empty:
                print("No valid data found in Recover file after processing")
                return None
            
            # Convert numeric columns to float
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                except:
                    pass
            
            print(f"Successfully parsed {len(df)} rows from Recover file")
            print(f"Columns: {list(df.columns)}")
            
            # Keep only relevant columns for merging
            # Looking for pressure column and chlorophyll-related columns
            pressure_cols = [col for col in df.columns if 'pressure' in col.lower() or col.lower() == 'depth']
            chlorophyll_cols = [col for col in df.columns if any(chl in col.lower() for chl in ['trx-chl', 'phycocyanin', 'phycoerythrin', 'chl(a)'])]
            
            keep_cols = pressure_cols + chlorophyll_cols
            
            if pressure_cols and chlorophyll_cols:
                print(f"Found pressure columns: {pressure_cols}")
                print(f"Found chlorophyll columns: {chlorophyll_cols}")
                return df[keep_cols].copy()
            else:
                print("No suitable columns found in Recover file")
                return None
                
        except Exception as e:
            print(f"Error reading Recover file {recover_filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def merge_recover_data(self, main_df: pd.DataFrame, recover_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge Recover file data with main Idronaut data using pressure/depth interpolation.
        
        Args:
            main_df: Main Idronaut DataFrame
            recover_df: Recover file DataFrame
            
        Returns:
            pd.DataFrame: Merged DataFrame with additional chlorophyll columns
        """
        try:
            # Find pressure columns
            main_pressure_col = None
            for col in ['Pres', 'pressure_dbar', 'Pressure']:
                if col in main_df.columns:
                    main_pressure_col = col
                    break
            
            recover_pressure_col = None
            for col in ['Pressure', 'Depth', 'Pres', 'pressure_dbar']:
                if col in recover_df.columns:
                    recover_pressure_col = col
                    break
            
            if main_pressure_col is None or recover_pressure_col is None:
                print("Could not find pressure/depth columns for interpolation")
                print(f"Main columns: {list(main_df.columns)}")
                print(f"Recover columns: {list(recover_df.columns)}")
                return main_df
            
            # Get chlorophyll columns from recover data
            chlorophyll_cols = [col for col in recover_df.columns if col != recover_pressure_col]
            
            if not chlorophyll_cols:
                print("No chlorophyll columns found in Recover data")
                return main_df
            
            print(f"Interpolating {chlorophyll_cols} from Recover data")
            print(f"Using pressure/depth columns: {main_pressure_col} (main) -> {recover_pressure_col} (recover)")
            
            # If using depth from recover file, we might need to convert to pressure or vice versa
            # For now, assume they are in similar units (both pressure or both depth)
            
            # Perform interpolation for each chlorophyll column
            merged_df = main_df.copy()
            
            for chl_col in chlorophyll_cols:
                # Remove NaN values and filter out invalid values for interpolation
                valid_mask = (~(np.isnan(recover_df[recover_pressure_col]) | np.isnan(recover_df[chl_col])) & 
                             (recover_df[recover_pressure_col] >= 0) & 
                             (recover_df[chl_col] >= 0))
                
                recover_pressure_clean = recover_df[recover_pressure_col][valid_mask]
                recover_chl_clean = recover_df[chl_col][valid_mask]
                
                if len(recover_pressure_clean) > 1:
                    # Sort by pressure/depth for proper interpolation
                    sort_idx = np.argsort(recover_pressure_clean)
                    recover_pressure_sorted = recover_pressure_clean.iloc[sort_idx]
                    recover_chl_sorted = recover_chl_clean.iloc[sort_idx]
                    
                    # Interpolate chlorophyll values at main pressure points
                    interpolated_values = np.interp(
                        main_df[main_pressure_col],
                        recover_pressure_sorted,
                        recover_chl_sorted
                    )
                    
                    # Add interpolated column to main dataframe
                    merged_df[chl_col] = interpolated_values
                    print(f"Added interpolated column: {chl_col}")
                    print(f"  Interpolated range: {interpolated_values.min():.3f} to {interpolated_values.max():.3f}")
                else:
                    print(f"Not enough data points for interpolation of {chl_col}")
            
            return merged_df
            
        except Exception as e:
            print(f"Error merging Recover data: {e}")
            import traceback
            traceback.print_exc()
            return main_df
    
    def read(self) -> pd.DataFrame:
        try:
            # Check for Recover file first
            recover_filepath = self.find_recover_file()
            recover_df = None
            
            if recover_filepath:
                print(f"Found Recover file: {recover_filepath}")
                recover_df = self.read_recover_file(recover_filepath)
            
            # Read main Idronaut file
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
            
            # Convert all numerical values to float
            df = df.apply(pd.to_numeric, errors='ignore')
            
            # Find column where "Date" is present
            # If Date is not a column
            if "Date" not in df.columns:
                # Find column where "Date" is present
                date_col = [col for col in df.columns if "Date" in col]
                # Rename the column to "Date"
                if date_col:
                    df.rename(columns={date_col[0]: "Date"}, inplace=True)
            
            df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
            
            # Merge with Recover data if available
            if recover_df is not None:
                df = self.merge_recover_data(df, recover_df)
            
            df = self._load_and_merge_metadata(df)
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
        df = self._load_and_merge_metadata(df)
        df = self.standardize_columns(df)
        return df

class GF23Reader(BaseReader):
    """Reader for GF23 CTD files (.txt)"""
    
    def find_recover_file(self) -> str:
        """
        Find a Recover file in the same directory as the GF23 file.
        
        Returns:
            str: Path to the Recover file if found, None otherwise
        """
        directory = os.path.dirname(self.filepath)
        filename = os.path.basename(self.filepath)
        
        # Look for files containing "Recover" in the same directory
        recover_patterns = [
            "*Recover*.txt",
            "*recover*.txt", 
            "*RECOVER*.txt",
            "*Recover*.TXT",
            "*recover*.TXT", 
            "*RECOVER*.TXT"
        ]
        
        for pattern in recover_patterns:
            recover_files = glob.glob(os.path.join(directory, pattern))
            if recover_files:
                # If multiple files found, try to match by station name or timestamp
                base_name = os.path.splitext(filename)[0]
                for recover_file in recover_files:
                    recover_name = os.path.basename(recover_file)
                    # Check if they might be related (same station or similar naming)
                    if any(part in recover_name.lower() for part in base_name.lower().split('_')):
                        return recover_file
                # If no specific match, return the first one
                return recover_files[0]
        
        return None
    
    def read_recover_file(self, recover_filepath: str) -> pd.DataFrame:
        """
        Read a Recover file and return DataFrame with pressure and chlorophyll data.
        Handles the special format with ** delimiters.
        
        Args:
            recover_filepath: Path to the Recover file
            
        Returns:
            pd.DataFrame: DataFrame containing Pressure and chlorophyll columns
        """
        try:
            print(f"Reading Recover file: {recover_filepath}")
            

            
            # Read with ** as delimiter, skip metadata, and fix header
            df = pd.read_csv(
            recover_filepath,
                sep=r"\*\*",        # delimiter is2**
                engine="python",
                header=2,           # take second line as header
                skiprows=[2]        # skip the "None" line
            )

            #Select only the third column and the last four columns
            df = df.iloc[:, [2] + list(range(-4, -1))]
            #Rename them: "Depth", "TRX-Chl(a)", "Phycocyanin", "Phycoerythrin"
            df.columns = ["Pressure", "TRX-Chl(a)", "Phycocyanin", "Phycoerythrin"]

            return df
            
        except Exception as e:
            print(f"Error reading Recover file {recover_filepath}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def merge_recover_data(self, main_df: pd.DataFrame, recover_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge Recover file data with main GF23 data using pressure/depth interpolation.
        
        Args:
            main_df: Main GF23 DataFrame
            recover_df: Recover file DataFrame
            
        Returns:
            pd.DataFrame: Merged DataFrame with additional chlorophyll columns
        """
        try:
            # Find pressure/depth columns
            main_pressure_col = None
            for col in ['Depth', 'Pressure', 'pressure_dbar', 'depth_m']:
                if col in main_df.columns:
                    main_pressure_col = col
                    break
            
            recover_pressure_col = None
            for col in ['Pressure', 'Depth', 'pressure_dbar', 'depth_m']:
                if col in recover_df.columns:
                    recover_pressure_col = col
                    break
            
            if main_pressure_col is None or recover_pressure_col is None:
                print("Could not find pressure/depth columns for interpolation")
                print(f"Main columns: {list(main_df.columns)}")
                print(f"Recover columns: {list(recover_df.columns)}")
                return main_df
            
            # Get chlorophyll columns from recover data
            chlorophyll_cols = [col for col in recover_df.columns if col != recover_pressure_col]
            
            if not chlorophyll_cols:
                print("No chlorophyll columns found in Recover data")
                return main_df
            
            print(f"Interpolating {chlorophyll_cols} from Recover data")
            print(f"Using pressure/depth columns: {main_pressure_col} (main) -> {recover_pressure_col} (recover)")
            
            # Perform interpolation for each chlorophyll column
            merged_df = main_df.copy()
            
            for chl_col in chlorophyll_cols:
                # Remove NaN values and filter out invalid values for interpolation
                valid_mask = (~(np.isnan(recover_df[recover_pressure_col]) | np.isnan(recover_df[chl_col])) & 
                             (recover_df[recover_pressure_col] >= 0) & 
                             (recover_df[chl_col] >= 0))
                
                recover_pressure_clean = recover_df[recover_pressure_col][valid_mask]
                recover_chl_clean = recover_df[chl_col][valid_mask]
                
                if len(recover_pressure_clean) > 1:
                    # Sort by pressure/depth for proper interpolation
                    sort_idx = np.argsort(recover_pressure_clean)
                    recover_pressure_sorted = recover_pressure_clean.iloc[sort_idx]
                    recover_chl_sorted = recover_chl_clean.iloc[sort_idx]
                    
                    # Interpolate chlorophyll values at main pressure points
                    interpolated_values = np.interp(
                        main_df[main_pressure_col],
                        recover_pressure_sorted,
                        recover_chl_sorted
                    )
                    
                    # Add interpolated column to main dataframe
                    merged_df[chl_col] = interpolated_values
                    print(f"Added interpolated column: {chl_col}")
                    print(f"  Interpolated range: {interpolated_values.min():.3f} to {interpolated_values.max():.3f}")
                else:
                    print(f"Not enough data points for interpolation of {chl_col}")
            
            return merged_df
            
        except Exception as e:
            print(f"Error merging Recover data: {e}")
            import traceback
            traceback.print_exc()
            return main_df

    def read(self) -> pd.DataFrame:
        try:
            # Check for Recover file first
            recover_filepath = self.find_recover_file()
            recover_df = None
            
            if recover_filepath:
                print(f"Found Recover file: {recover_filepath}")
                recover_df = self.read_recover_file(recover_filepath)
            
            # Read the main GF23 TXT file
            df = pd.read_csv(self.filepath, delim_whitespace=True, skiprows=2, header=None)

            # Force the columns to match the expected structure
            df.columns = [
                "Depth", "Temperature", "Conductivity", "Oxygen %", "Oxygen mg/L",
                "pH", "PAR", "Salinity", "SigmaT", "Trx-chl(a)", "Pressure"
            ]

            # Remove "**" from the dataset
            df = df.replace(r'\*\*', '', regex=True)

            # Convert all numerical values to float
            df = df.apply(pd.to_numeric, errors='ignore')

            # Load the activity log
            # Get parent dir to             os.path.dirname(self.filepath)
            parent_dir = os.path.dirname(os.path.dirname(self.filepath))
            logpath = os.path.join(parent_dir, "activity_log_ODV.csv")
            activity_log = pd.read_csv(logpath)
            activity_log = activity_log[activity_log["Activity"] == "CTD"]

            # Fill missing latitudes with lat.OUT
            activity_log["lat IN"] = activity_log.apply(
                lambda row: row["lat OUT"] if row["lat IN"] == "" else row["lat IN"], axis=1
            )

            # Convert GPS coordinates
            activity_log["Latitude"] = activity_log["lat IN"].apply(self._convert_gps)
            activity_log["Longitude"] = activity_log["long IN"].apply(self._convert_gps) * -1

            # Clean and format the activity log
            activity_log_clean = activity_log[["station_ID", "station_ODV", "date", "Latitude", "Longitude", "time In"]].copy()
            activity_log_clean.rename(columns={"station_ID": "Station"}, inplace=True)

            # Extract the station name from the file path
            station_name = os.path.splitext(os.path.basename(self.filepath))[0]
            df["Station"] = station_name

            # Merge the activity log with the data
            df = df.merge(activity_log_clean, on="Station", how="left")
            if df["Latitude"].isnull().all() or df["Longitude"].isnull().all():
                print(f"Warning: No matching activity log entry found for station {station_name}")
            # Add the date as a new variable
            try:
                # Handle both two-digit and four-digit year formats
                df["date"] = pd.to_datetime(df["date"], format="%d.%m.%y", errors="coerce")
                if df["date"].isnull().any():
                    # Retry with four-digit year format if parsing fails
                    df["date"] = pd.to_datetime(df["date"], format="%d.%m.%Y", errors="coerce")
            except Exception as e:
                print(f"Error parsing date: {e}")
                raise

            # Add a datetime column by combining date and time
            try:
                df["datetime"] = pd.to_datetime(df["date"].dt.strftime("%Y-%m-%d") + " " + df["time In"])
            except Exception as e:
                print(f"Error creating datetime column: {e}")
                raise

            # Standardize column names using the config.py mappings
            column_mapping = get_column_mapping(self.reader_type)
            df.rename(columns=column_mapping, inplace=True)

            # Standardize column names and apply unit conversions
            df = self.standardize_columns(df)
            
            # Merge with Recover data if available
            if recover_df is not None:
                df = self.merge_recover_data(df, recover_df)

            df = self._load_and_merge_metadata(df)

            return df

        except Exception as e:
            print(f"Error reading GF23 file {self.filepath}: {e}")
            raise

    @staticmethod
    def _convert_gps(string):
        """Convert GPS coordinates from DMD format to decimal degrees."""
        import re
        string_clean = re.sub(r"[’'‘]", "", string).replace(",", ".")
        degrees = float(string_clean.split("°")[0])
        minutes = float(string_clean.split("°")[1]) / 60
        return degrees + minutes

class ExoReader(BaseReader):
    """Reader for exo probe"""
    def __init__(self, filepath: str, reader_type: str, campaign_name: str = None):
        super().__init__(filepath, reader_type, campaign_name)
        self.units = {}
    def read(self) -> pd.DataFrame:
        """Read RBR CTD data from text file.
        
        Returns:
            pd.DataFrame: DataFrame with processed RBR CTD data
        """
        self.df = pd.read_csv(self.filepath, encoding="utf-16", delimiter=",", skiprows=9)
        # Convert all numeric columns to float, handling any non-numeric entries
        for column in self.df.columns:
            # Skip columns that are likely dates or text
            if any(keyword in column.lower() for keyword in ['date', 'time', 'site', 'station', 'id', 'name', 'comment']):
                continue
                
            # Check if the column appears to be numeric (most values can be converted)
            sample_values = self.df[column].dropna().head(10)  # Take a sample of the first values
            if len(sample_values) == 0:
                continue  # Skip empty columns
                
            # Try to determine if column is numeric by checking if most values can be converted
            numeric_count = 0
            for val in sample_values:
                try:
                    float(val)
                    numeric_count += 1
                except (ValueError, TypeError):
                    pass
                    
            # If at least 70% of sample values are numeric, convert the whole column
            if numeric_count / len(sample_values) >= 0.7:
                try:
                    self.df[column] = pd.to_numeric(self.df[column], errors='coerce')
                    print(f"Converted column '{column}' to numeric")
                except Exception as e:
                    print(f"Could not convert column '{column}' to numeric: {e}")
                
        #Standardize column names
        self.df = self.standardize_columns(self.df)
        
        # Convert conductivity from µS/CM to mS_per_m
        if 'COND µS/CM' in self.df.columns:
            self.df['COND µS/CM'] = self.df['COND µS/CM'] * 0.1  # µS/CM to mS_per_m
            self.df.rename(columns={'COND µS/CM': 'conductivity_mS_per_m'}, inplace=True)
            print("Converted 'COND µS/CM' to 'conductivity_mS_per_m'")
        
        return self.df
        
class RBRReader(BaseReader):
    """Reader for RBR CTD files."""
    
    def __init__(self, filepath: str, reader_type: str, campaign_name: str = None):
        super().__init__(filepath, reader_type, campaign_name)
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
                unit = self.units
                
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


class RBRruskinReader(BaseReader):
    """Reader for RBR RSK files using the RBR Python package."""
    
    def __init__(self, filepath: str, reader_type: str, campaign_name: str = None):
        super().__init__(filepath, reader_type, campaign_name)
        if rbr is None:
            raise ImportError("RBR package is required to read .rsk files. Install with: pip install rbr")
    
    def read(self) -> pd.DataFrame:
        """Read RBR RSK data using the RBR Python package.
        
        Returns:
            pd.DataFrame: DataFrame with processed RBR RSK data
        """
        try:
            # Open the RSK file
            rsk = rbr.RSK(self.filepath)
            rsk.open()
            # Read the data
            rsk.readdata()
            
            # Convert to pandas DataFrame
            # The RBR package provides data in rsk.data which is a list of casts
            if rsk.data is None or len(rsk.data) == 0:
                raise ValueError(f"No data found in RSK file: {self.filepath}")
            
            # Get the first cast (most common case)
            df = pd.DataFrame(rsk.data)
            # Close the RSK file
            rsk.close()
            
            # Convert all numeric columns
            for col in df.columns:
                if col.lower() not in ['date', 'time', 'timestamp']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Use the standardize_columns method which handles both renaming AND unit conversion
            # This will apply the conversions defined in config.py
            df = self.standardize_columns(df)
            
            return df
            
        except Exception as e:
            print(f"Error reading RSK file {self.filepath}: {e}")
            raise