# CTD Processing Pipeline

This repository contains a Python-based pipeline for processing CTD (Conductivity, Temperature, Depth) data. The pipeline supports multiple CTD types, applies quality control, calculates oceanographic parameters, and segments profiles for further analysis.

---

## **Features**
1. **CTD Data Processing**:
   - Supports multiple CTD types: Idronaut, Seabird, RBR, and Exo.
   - Applies corrections to raw data, including:
     - Removing air measurements.
     - Correcting oxygen saturation (dO2) using air data.
     - Filtering invalid pH values.
   - Calculates oceanographic parameters such as salinity, density, and mixed layer depth (MLD).

2. **Profile Segmentation**:
   - Segments CTD profiles based on pressure data using local minima as delimiters.

3. **PAR (Photosynthetically Active Radiation)**:
   - Calculates the average PAR in air data during the air/water separation step.

4. **Output Management**:
   - Processes multiple files while preserving the directory structure.
   - Saves segmented profiles as individual CSV files.

5. **Visualization**:
   - Generates profile plots for visual inspection (via `visualize` module).

---

## **File Structure**
```
CTD_processing/
├── data/                     # Directory for raw and processed data
│   ├── Level0/               # Raw data
│   └── Level1/               # Processed data
├── src/                      # Source code
│   ├── main.py               # Main script for processing CTD files
│   ├── processors.py         # Functions for data processing and calculations
│   ├── readers.py            # CTD-specific file readers
│   ├── visualize.py          # Visualization utilities
│   └── config.py             # Configuration for site-specific parameters
└── README.md                 # Documentation
```

---

## **Setup**

### **Requirements**
- Python 3.8+
- Required Python libraries:
  - `numpy`
  - `pandas`
  - `scipy`
  - `gsw` (Gibbs SeaWater Oceanographic Toolbox)
  - `matplotlib` (for visualization)

### **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/CTD_processing.git
   cd CTD_processing
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **Usage**

### **1. Process All Files**
Run the main.py script to process all CTD files in a directory:
```bash
python src/main.py
```

### **2. Directory Structure**
- Place raw data in the `data/Level0/<campaign>` directory.
- Processed data will be saved in `data/Level1/<campaign>`.

### **3. Configuration**
Update `config.py` with site-specific parameters such as latitude, longitude, and CTD mappings.

---

## **Key Modules**

### **1. main.py**
- Entry point for processing CTD files.
- Functions:
  - `process_all_files(directory, ship, outputfolder)`: Processes all files in a directory.
  - `process_ctd_file(filepath, ctd_type, data_dir, outputfolder)`: Processes a single CTD file and segments profiles.

### **2. `processors.py`**
- Core processing functions:
  - `clean_air_data`: Removes air measurements and applies corrections.
  - `calculate_ocean_params`: Calculates oceanographic parameters (e.g., salinity, density, MLD).
  - `identify_downcast`: Identifies the downcast portion of the profile.
  - `quality_check_ph`: Filters invalid pH values.
  - `find_mld`: Calculates mixed layer depth (MLD).

### **3. `readers.py`**
- CTD-specific file readers:
  - `IdronautReader`
  - `SeabirdReader`
  - `RBRReader`
  - `ExoReader`

### **4. `visualize.py`**
- Visualization utilities:
  - `create_profile_plot`: Generates plots for CTD profiles.

---

## **Processing Workflow**

1. **Read Raw Data**:
   - Use CTD-specific readers to load raw data into a DataFrame.

2. **Clean Air Data**:
   - Separate air and water data.
   - Apply corrections (e.g., pressure and oxygen offsets).
   - Calculate average PAR in air.

3. **Process Oceanographic Parameters**:
   - Calculate salinity, density, and other derived parameters.
   - Compute mixed layer depth (MLD).

4. **Segment Profiles**:
   - Use pressure data to segment profiles into individual casts.

5. **Save Results**:
   - Save processed profiles as CSV files in the output directory.

---

## **Example**

### **Input Directory Structure**
```
data/
└── Level0/
    └── Campaign1/
        ├── file1.cnv
        ├── file2.txt
        └── subfolder/
            └── file3.csv
```

### **Command**
```bash
python src/main.py
```

### **Output Directory Structure**
```
data/
└── Level1/
    └── Campaign1/
        ├── file1_profile_1.csv
        ├── file1_profile_2.csv
        ├── file2_profile_1.csv
        └── subfolder/
            └── file3_profile_1.csv
```

---

## **Notes**
- Ensure the `config.py` file is correctly configured for your site and CTD types.
- The script assumes standardized column names for processing. Update `CTD_MAPPINGS` in `config.py` if needed.

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Acknowledgments**
- This pipeline uses the Gibbs SeaWater (GSW) Oceanographic Toolbox for Python for oceanographic calculations.
- Special thanks to the contributors and maintainers of the libraries used in this project.