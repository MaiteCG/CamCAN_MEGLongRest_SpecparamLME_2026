"""
CamCAN Longitudinal Resting-State MEG Preprocessing Configuration
================================================================

This script serves as the configuration file for the MNE-BIDS-Pipeline, 
specifically tailored for the CamCAN (Cambridge Centre for Ageing and 
Neuroscience) Phases 2 and 5 from the longitudinal resting-state MEG data.

Key Functionalities:
--------------------
* **BIDS Integration**: Dynamically constructs 'rawdata' and 'derivatives' 
  paths based on pipeline version and filter parameters.
* **Signal Processing**: Sets frequency filters (0.1 to 145 Hz) 
  and downsampling to 300 Hz.
* **Spatial Normalization**: Implements MaxFilter (tSSS) with movement 
  compensation and head position transformation to a fixed Z-coordinate 
  (44 mm) for cross-subject alignment.

Usage:
------
This file is intended to be passed as the `--config` argument to the 
MNE-BIDS-Pipeline execution command, e.g.: 

mne_bids_pipeline --config=/path/to/your/config/file.py

However, it is also possible to specify more configuration parameters, which we
did, for convenience:

For running the entire preprocessing pipeline for a subject:
mne_bids_pipeline --config=/path/to/your/config/file.py --subject=subject --steps=preprocessing

For running specific preprocessing steps for a subject:
mne_bids_pipeline --config=/path/to/your/config/file.py --subject=subject --steps=preprocessing/_01_data_quality

mne_bids_pipeline --config=/path/to/your/config/file.py --subject=subject --steps=preprocessing/_02_head_pos

mne_bids_pipeline --config=/path/to/your/config/file.py --subject=subject --steps=preprocessing/_03_maxfilter

mne_bids_pipeline --config=/path/to/your/config/file.py --subject=subject --steps=preprocessing/_04_frequency_filter

mne_bids_pipeline --config=/path/to/your/config/file.py --subject=subject --steps='preprocessing/_07_make_epochs'

For more details on how to run the pipeline and specify parameters, refer to the MNE-BIDS-Pipeline documentation: https://mne.tools/mne-bids-pipeline/stable/getting_started/basic_usage.html#run-the-pipeline

Parameters:
-----------
For detailed information on each pipeline parameter, refer to the MNE-BIDS-Pipeline documentation: https://mne.tools/mne-bids-pipeline/stable/settings/general.html

"""

import mne
import numpy as np
import os
import pandas as pd
from mne.transforms import translation

# =============================================================================
# --- Project-specific Settings ---
# =============================================================================
maindir = '' # path where the BIDS project folder is stored, e.g. '/home/CamCAN/data/'
bids_project_folder = '' # Name of the BIDS project folder, e.g. 'BIDS_long_P2_rest_arm1'

# --- Pipeline-specific variables ---
pipver = '' # any string to identify the version of the pipeline, e.g. 'v01'.
lfreq = 0.1 # Hz, high-pass filter cutoff frequency. 
hfreq = 145.0 # Hz, low-pass filter cutoff frequency. 
fsample = 300.0 # Hz, resampling frequency.
frange = f"{round(lfreq, 1)}-{int(hfreq)}Hz"

trans = True # whether to apply head position transformation to a fixed Z-coordinate for cross-subject alignment
zmm = 44 # destination z coordinate head position in mm

runicastep = False # whether to run the ICA step for artifact removal. We set it to False because we inserted a custom bad epochs detection step before ICA, and ran ICA steps with custom scripts to select ocular and cardiac components based on correlation with EOG and ECG channels using specific r thresholds.

# --- Directories and files ---
bids_root = os.path.join(maindir, bids_project_folder, 'rawdata')

# The name of the derivatives folder is up to the user, but we recommend including the pipeline version and the filter parameters in the name for clarity and reproducibility. If you apply the head position transformation, it's also good to include that in the name.
if trans:
    derivfolder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    derivfolder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz'

deriv_root = os.path.join(maindir, bids_project_folder, 'derivatives', derivfolder)


# --- Functions ---
def makenewdir(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

# Create the derivatives directory if it doesn't exist
makenewdir(deriv_root)

# Print the paths for verification (optional)
print(f'BIDS root: {bids_root}')
print(f'Derivatives root: {deriv_root}')

# =============================================================================
# --- Pipeline's Configuration options ---
###############################################################################
# General settings
# ----------------
bids_root = bids_root # Specify the BIDS root directory. 
deriv_root = deriv_root # The root of the derivatives directory in which the pipeline will store the processing results.
subjects =  'all' # or list of subject IDs, e.g. ['CC110033', 'CC110047']
# sessions = 'all' # ['P2'] # activate if the version of the bids data has the session tag.
runs = ["01"] # deactivate if the version of the bids data doesn't have the run tag.
ch_types = ["meg"] # The channel types to consider.
data_type = "meg" # The BIDS data type.
crop_runs = None # We did not crop the data at this point
process_empty_room = False

# ---- MNE-BIDS-Pipeline Preprocessing steps ----
###############################################################################
# Bad channel detection
# ----------------
find_flat_channels_meg = True
find_noisy_channels_meg = True

###############################################################################
# Maxwell Filter
# ----------------
use_maxwell_filter = True
mf_st_duration = 10 # seconds
mf_st_correlation = 0.98
mf_head_origin = "auto" # If 'auto', it will be estimated from headshape points.
mf_int_order = 8

if trans:
    mf_destination = translation(z=zmm/1000)  # z in mm, convert to m

# --- Movement correction ---
mf_mc = True
mf_mc_t_step_min = 0.01
mf_mc_t_window = "auto"
mf_mc_gof_limit = 0.98
mf_mc_dist_limit = 0.005
mf_mc_rotation_velocity_limit = None
mf_mc_translation_velocity_limit = None
mf_filter_chpi = True

###############################################################################
# Filtering and resampling
# ----------------
l_freq = lfreq
h_freq = hfreq
notch_freq = [50, 100, 150]
raw_resample_sfreq = fsample

###############################################################################
# Epoching
# ----------------
conditions = None  # for a resting-state analysis
task_is_rest = True
epochs_tmin = 0.0 # must be 0 for rest
epochs_tmax = 10.0 # must be 10 for this pipeline
rest_epochs_duration = 10
rest_epochs_overlap = 0
baseline = None

###############################################################################
# Artifact removal
# ----------------
# This step was not used for the analyses in the paper
if runicastep == True:
    spatial_filter = 'ica'
    ica_reject = None #"autoreject_local" #None # #{'grad': 400e-12, 'mag': 10e-12}
    ica_algorithm = "picard"
    ica_l_freq = 1.0
    ica_max_iterations = 3000
    ica_n_components = None #0.8
    ica_ecg_threshold = 0.15
    ica_eog_threshold = 2.5
