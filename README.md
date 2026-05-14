# CamCAN_MEGLongRest_SpecParamLME_2026

This repository contains analysis code for the following manuscript: 

Title: **Longitudinal versus Cross-sectional effects of Age on MEG Power Spectra Parameters: Implications for Normative Models and Brain Ageing**

Authors: Maite Crespo-Garcia, Dace Apšvalka, Ina Demetriou, Adam Attaheri, Tina Emery, Máté Aller, Cam-CAN & Richard Henson.



## Code Description

### Preprocessing

The configuration files for the **MNE-BIDS automatic pipeline** are located in `code/prep/mnebids`:

- `config_rest.py`:  Configuration for automatic preprocessing of **rest MEG** data.

- `config_emptyroom.py`: Configuration for automatic preprocessing of **empty room** MEG data associated to resting-state MEG data.
  
  

The preprocessing of the **ECG channel** is performed within the script that computes the spectral power (see below). 



The **custom** Python files for other **preprocessing steps** are located in `code/prep`:

- `custom_bad_epochs.py`: Detects bad epochs in **rest MEG** data using muscle artifact z-scoring. Saves the indexes of the good epochs to be used later in the pipeline.

- `custom_icastep.py`: Executes a customized ICA fitting procedure on filtered **rest MEG** data, after removing bad epochs detected with the `custom_bad_epocs.py` script.

- `custom_icaartifacts_schmidt.py`: This script identifies ICA components corresponding to EOG (ocular) and ECG (cardiac) artifacts by applying the correlation-based thresholding method described in Schmidt et al. (2025 eLife).

- `custom_remove_icaartifacts.py`: Applies the ICA rejection identified in `custom_icaartifacts_schmidt.py` to the filtered **rest MEG** data. It serves as the final cleaning step before downstream analysis. Alternatively, it can also create another **control rest MEG data only with ocular artifacts (but not cardiac) removed**.

- `custom_remove_allbutecg04.py`: Performs a selective ICA-based cleaning to create a **control dataset only with cardiac ICs (most of brain activity removed)**. It removes all independent components EXCEPT those identified as cardiac (ECG-related) artifacts in `custom_icaartifacts_schmidt.py`.
  
  

### Spectral analyses

- `psd_rest.py`: Computes the **power spectrum** for **rest MEG** data that has been cleaned of ocular and cardiac ICA artifacts (via `custom_remove_icaartifacts.py`), and other **control MEG datasets** with different ICA removal treatments.

- `psd_emptyroom.py`: Computes the **power spectrum** for **empty room** recordings associated with specific rest MEG recordings.

- `psd_ecg.py`: Preprocesses and computes the **power spectrum** of the **ECG channel**.
  
  

### Statistical analyses



### Figures and tables

Give an overview of the program files and their purposes. Remove redundant or obsolete files from the replication archive. For example, main.do sets file paths, installs necessary ADO packages, and executes all other dofiles. Meanwhile, cleaning.do loads data, handles missing values, and analysis.do performs basic statistical analysis and generate visualizations.

Make sure to also include any crucial information that replicators should be aware of to facilitate a one-click run of the code.

## Dependencies (incomplete list):

- Python

- MNE-Python, MNE-BIDS pipeline

- pandas, numpy, matplotlib

- specparam 2.0

- pymer4

## Overview

Python scripts to run preprocessing and analyses: `code`

Source data to obtain the figures and tables: `data`

Figures and Tables: `results/figures` and `results/tables` 

## Data Availability

The raw MEG data from Phase 2 are already available on the CamCAN repository([Cam-CAN Data Repository](https://cam-can.mrc-cbu.cam.ac.uk/dataset)).

Data from Phase 5 will be added soon (or available on request until then).

### Data Sources

Provide detailed information about the data sources, whether obtained from public repositories, institutional databases, or other sources. Include instructions on how others can access the data, including where it can be downloaded and the names under which it is cataloged. This is particularly important for replicators to ensure consistent results by using the same datasets. For example, if the package uses the World Bank’s World Development Indicators, ensure to add the URL, variable names, and file names exactly as they appear on the portal, and the year the data was accessed.

You can use the following as a template. Make sure to fill out this information for each of the data files used:

- **Filename 1:** Exact file name as shown on the source website

- **Source:** Name of the source website

- **URL:** Exact downloadable URL of the data used

- **Access year:**  Date when the data was accessed. This is especially important as data can be updated, and replicators should know the exact time when the data was downloaded.

- **Variable names (optional):** This is useful when only data for certain variables is downloaded, such as when using the World Bank’s World Development Indicators data.

- **License (optional):** While this is not mandatory, it is great to know under which license the data is available to understand if it is public or private, or publication limitations.

### Statement about Rights

- [ ] I certify that the author(s) of the manuscript have legitimate access to and permission to use the data used in this manuscript.
- [ ] I certify that the author(s) of the manuscript have documented permission to redistribute/publish the data contained within this replication package. Appropriate permission are documented in the LICENSE.txt file.

## Instructions for Replicators

New users should follow these steps to run the package successfully:

- Users must first have access to all data files if they are not included in the reproducibility package. They should go to the mentioned links, download the listed files, and place them in the data folder.

- Update the following files with your directory paths
  
  - `main_dofile.do`

- Ensure all required software and dependencies are installed as listed in the [Requirements](#requirements) section.

- Run the `main_dofile.do` file.

## List of Exhibits

Clearly identify and document the tables and figures as they appear in the manuscript by their corresponding numbers. If file names do not correspond to exhibit numbers, provide detailed explanations.

If not all data is provided in the reproducibility package, as described in the data section, then the list of tables should clearly indicate which tables, figures, and in-text numbers can be reproduced with the public material provided.

Example template for exhibit identification:

The provided code reproduces:

- [ ] All numbers provided in text in the paper
- [ ] All tables and figures in the paper
- [ ] Selected tables and figures in the paper, as explained and justified below

| Exhibit name | Output filename  | Script                   | Note                                                                                                                                                                                                       |
| ------------ | ---------------- | ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Table 1      | Balancetable.xls | 02_analysis.do (line 23) | Found in Outputs/tables/main                                                                                                                                                                               |
| Figure 1     | Regresults.png   | 02_analysis.do (line 40) | Found in Outputs/figures/annex, Image Format: Portable Network Graphic (PNG), Bits Per Pixel: 32, Color: Truecolour with alpha, Dimensions: 970 x 544, Interlaced: Yes, XResolution: 144, YResolution: 144 |

## Requirements

### Computational Requirements

We used Linux as main operative system for running all the Python scripts.

In this section, specify operating system requirements, software dependencies, environment setup instructions, and any other relevant information essential for replicating the results. Each of these factors plays an important role in ensuring successful replication.

### Software Requirements

List all software requirements, including versions, dependencies, libraries, environment setup, and packages installed. Using different versions of the same software could lead to variations in results. If multiple software are used, include details for all.

Example:

- **Stata version 15**
  
  - estout
  
  - rdrobust

- **Python 3.6.4**
  
  - pandas 0.24.2
  
  - numpy 1.16.4

### Memory and Runtime and Storage Requirements

Provide consistent information about memory resources for reliable computation. Include runtime information for replicators to assess processing times and detect potential issues with the code. It would be best to describe how much storage is required in addition to the space visible in the typical repository, for instance, because data will be unzipped, data downloaded, or temporary files written.

## 

## Folder Structure



```
code
  ├── Main_dofile.do
  ├── 01_cleaning.do
  └── 02_analysis.do
data
  ├── Figure01_source_data.tsv
  ├── Figure02_source_data.tsv
  ├── Figure03_source_data.tsv
  ├── Figure04_source_data.tsv
  ├── FigureS01_source_data.tsv
  ├── FigureS02_source_data.tsv
  ├── FigureS03_source_data.tsv
  ├── FigureS04_source_data.tsv
  ├── FigureS05_source_data.tsv
  ├── FigureS06_source_data.tsv
  ├── FigureS07_source_data.tsv
  ├── FigureS08_source_data.tsv
  ├── Table01_source_data.tsv
  ├── Table02_source_data.tsv
  ├── Table03_source_data.tsv
  ├── Table04_source_data.tsv
  ├── Table05_source_data.tsv
  ├── TableS01_source_data.tsv
  ├── TableS02_source_data.tsv
  ├── TableS03_source_data.tsv
  ├── TableS04_source_data.tsv
  ├── TableS05_source_data.tsv
  ├── TableS06_source_data.tsv
  ├── TableS07_source_data.tsv
  ├── TableS08_source_data.tsv
  ├── TableS09_source_data.tsv
  ├── TableS10_source_data.tsv
  └── TableS11_source_data.tsv
results
  ├── Tables
  └── Figures
```
