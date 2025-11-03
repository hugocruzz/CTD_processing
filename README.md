# CTD Data Processing Pipeline

This repository contains a comprehensive Python-based pipeline for processing CTD (Conductivity, Temperature, Depth) data from various campaigns and instruments. The pipeline is designed to read raw data, apply essential corrections and quality control, calculate key oceanographic parameters, and structure the output for further analysis.

---

## **Features**

- **Multi-Instrument Support**: Readers for various CTD types, including Idronaut, Seabird, RBR, and EXO.
- **Data Standardization**: Standardizes column names and units across different instrument formats using a centralized configuration.
- **Automated Processing**: Applies a series of corrections and calculations to clean and enrich the data.
- **Profile Segmentation**: Automatically segments a single data file into multiple profiles based on pressure changes.
- **Flexible Output**: Generates processed data in CSV format, organized by campaign and processing level.
- **Metadata Integration**: Merges data with logbook or metadata files where available.

---

## **File Structure**

```
CTD_processing/
├── data/
│   ├── Level0/         # Raw, unprocessed instrument data
│   ├── Level1/         # Profiles after basic processing and segmentation
│   ├── Level2/         # Fully processed and quality-controlled data
│   └── combined/       # Merged datasets from campaigns
├── notebooks/          # Jupyter notebooks for merging and plotting
├── src/                # Python source code
│   ├── main.py         # Main script to run the processing pipeline
│   ├── processors.py   # Core data correction and calculation functions
│   ├── readers.py      # Instrument-specific data readers
│   ├── config.py       # Mappings for columns and unit conversions
│   ├── utils.py        # Helper functions for file handling
│   └── visualize.py    # Plotting and visualization tools
└── README.md           # This documentation file
```

---

## **Setup**

### **Requirements**

- Python 3.8+
- `pandas`, `numpy`, `scipy`, `gsw`

### **Installation**

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd CTD_processing
    ```

2.  **Install dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install pandas numpy scipy gsw
    ```

---

## **Data Processing Workflow**

The core of the pipeline is in the `src/processors.py` script, which applies a series of steps to the raw data read from the instruments.

### **1. Reading and Standardization**

- **Reader Selection**: Based on the file type and name (`get_ctd_type` in `main.py`), the appropriate reader from `readers.py` is chosen.
- **Column Mapping**: The `standardize_columns` method in the `BaseReader` class uses `config.py` to map raw instrument column names (e.g., `t090C`, `c0S/m`) to standardized names (e.g., `temperature_C`, `conductivity_mS_per_m`).
- **Unit Conversion**: During standardization, unit conversions are applied. For example, conductivity from Seabird files in S/m is converted to mS/m by multiplying by 1000.

### **2. Air Data Removal (`clean_air_data`)**

This is a critical step to remove measurements taken while the CTD was out of the water.

- **Identifying Air Data**: The function identifies "in-air" measurements by finding data points where conductivity is below a certain threshold (`threshold_cond`). For most instruments, this is `1 mS/m`.
- **Pressure Offset Correction**: The median pressure from the "in-air" data is calculated and treated as the atmospheric pressure offset. This offset is then subtracted from the entire pressure column to set the water surface as zero pressure.
- **Oxygen Offset Correction**: If oxygen data is available, the median oxygen saturation from the "in-air" data is compared to 100% saturation. The difference is used as an offset to correct the oxygen measurements.
- **Filtering**: After corrections, all data points with a negative (post-correction) pressure are removed.

### **3. Oceanographic Parameter Calculation (`calculate_ocean_params`)**

After cleaning, the pipeline calculates several key oceanographic parameters using the **Gibbs SeaWater (GSW) Oceanographic Toolbox**.

- **Salinity (Absolute and Practical)**: If not provided by the instrument, Practical Salinity is calculated from conductivity, temperature, and pressure using the **PSS-78 equation** (`gsw.SP_from_C`).
- **Density**: Conservative Temperature, Absolute Salinity, and density (`sigma0`, `sigma1`, etc.) are calculated using the TEOS-10 standard.
- **Oxygen Conversion**: Oxygen concentration is converted between different units (e.g., mg/L to ml/L) as needed.

### **4. Downcast Identification (`identify_downcast`)**

For many analyses, only the "downcast" (when the CTD is descending) is used.

- **Finding Maximum Depth**: The function identifies the index of the maximum pressure or depth value in the profile.
- **Flagging the Downcast**: All data points from the start of the profile up to this maximum depth index are flagged as `is_downcast = True`.

### **5. Quality Control (`quality_check_ph`)**

- **pH Filtering**: The `quality_check_ph` function flags or removes pH values that are outside a plausible range (typically 6 to 9).

### **6. Profile Segmentation (`segment_profiles`)**

A single file may contain multiple casts. The `segment_profiles` function (in `main.py`) splits them.

- **Finding Minima**: It uses `scipy.signal.find_peaks` to find local minima in a smoothed pressure series. These minima represent the points where the CTD was brought to the surface between casts.
- **Creating Segments**: The data between these minima are extracted as separate profiles.

---

## **Usage**

To run the processing pipeline, configure the `if __name__ == "__main__":` block in `src/main.py`.

1.  **Set the `campaign`**: Specify the name of the campaign folder inside `data/Level0/`.
    ```python
    campaign = "Greenfjord2023"
    ```
2.  **Define Paths**: The script automatically sets the input (`data_dir`) and output paths (`Level1_output`, `Level2_output`).
3.  **Run the script**:
    ```bash
    python src/main.py
    ```

The processed files will be saved in the corresponding `data/Level1` and `data/Level2` directories, preserving the subfolder structure of the raw data.