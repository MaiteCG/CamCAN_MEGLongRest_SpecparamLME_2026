import mne
import numpy as np
import os
import pandas as pd
import sys
from mne.transforms import translation

if os.name == 'nt':
    cfgdir = r"U:\Documents\CamCAN\code\maipy"
    print('nt way')
else:
    cfgdir = "/imaging/camcan/sandbox/mc06/code/maipy"
    print('linux way')  

sys.path.insert(1, cfgdir)
import mcgdirs as dirs

# Functions
def makenewdir(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)

# Main variables
task = 'rest'
phase = 'p2'
arm = 1
pipver = 'stier' #'v02' #'v01'
lfreq = 0.1 #Hz
hfreq = 145.0 #Hz
fsample = 300.0 #Hz
frange = f"{round(lfreq, 1)}-{int(hfreq)}Hz"

trans = True
zmm = 44 # destination z coordinate head position in mm

runicastep = False

# Directories and files
bids_project_folder = f'BIDS_long_{phase}_{task}_arm{arm}' # _last4
#bids_project_folder = f'BIDS_long_{task}' 
bids_root = os.path.join(dirs.mysandboxdatadir, bids_project_folder, 'rawdata')
#bids_root = os.path.join(dirs.camcandir, 'ccrescan','BIDS', 'data')

if trans:
    derivfolder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    derivfolder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz'

deriv_root = os.path.join(dirs.mysandboxdatadir, bids_project_folder, 'derivatives', derivfolder)

makenewdir(deriv_root)

print(f'BIDS root: {bids_root}')
print(f'Derivatives root: {deriv_root}')

# Pipeline variables
subjects =  'all' #['CC420173'] #['CC120065'] #, 'CC121428', 'CC210088', 'CC220107'] # , 'CC220335'
#sessions = ['P2']
runs = ["01"] #activate for my version of the bids data
ch_types = ["meg"]
data_type = "meg"

crop_runs = None #(300.0, 600.0)  # 5 minutes from the middle of the recording for speed
process_empty_room = False

# ---- MNE-BIDS-Pipeline Preprocessing steps ----
# ---- Bad channel detection ----
# Note: Find the way to include here the noisy channels that were visually identified
# Probably add to the raw data, because the documentation says that: "The list of bad channels 
# detected through this procedure will be merged with the list of bad channels already present 
# in the dataset, if any."
find_flat_channels_meg = True
find_noisy_channels_meg = True

# ---- Maxwell Filter ----
use_maxwell_filter = True
mf_st_duration = 10 #MaxFilter™'s default is 10.0 seconds in v2.2.
mf_st_correlation = 0.98
mf_head_origin = "auto" # in meters, array-like, shape (3,) | If 'auto', it will be estimated from headshape points.
mf_int_order = 8 # MaxFilter default is 8
if trans:
    mf_destination = translation(z=zmm/1000)  # z in mm, convert to m



# mf_destination = "reference_run"
# mf_reference_run: Optional[str] = '01'
# mf_cal_fname: Optional[str] = None #only be used for BIDS datasets that don't store
     # the fine-calibration file
# mf_ctc_fname: Optional[str] = None # If `None`, the recommended location is used.
#mf_st_fixed =  True # If True (default), do tSSS using the median head position during the st_duration window. This is the default behavior of MaxFilter and has been most extensively tested. 
#mf_st_only = False #If True, only tSSS (temporal) projection of MEG data will be performed on the output data. (...) Noise reduction from SSS basis multiplication, cross-talk cancellation, movement compensation, and so forth will not be applied to the data.

# movement correction
mf_mc = True
mf_mc_t_step_min = 0.01
mf_mc_t_window = "auto"
mf_mc_gof_limit = 0.98
mf_mc_dist_limit = 0.005
mf_mc_rotation_velocity_limit = None
mf_mc_translation_velocity_limit = None
mf_filter_chpi = True

'''if pipver == 'v01':
    mf_filter_chpi = None #gave an error: No appropriate cHPI information found in info["hpi_meas"] and info["hpi_subsystem"]
elif pipver == 'v02':
    mf_filter_chpi = True
elif pipver == 'v03':
    mf_filter_chpi = False'''

# ---- Filtering and resampling ----
l_freq = lfreq
h_freq = hfreq
notch_freq = [50, 100, 150]
raw_resample_sfreq = fsample

# ---- Epoching ----
if task == 'rest':
    conditions = None  # for a resting-state analysis
    task_is_rest = True
    epochs_tmin = 0.0 # must be 0 for rest
    epochs_tmax = 10.0 # must be 10 for this pipeline
    rest_epochs_duration = 10
    rest_epochs_overlap = 0
    baseline = None

###############################################################################
# ARTIFACT REMOVAL
# ----------------
#
# You can choose between ICA and SSP to remove eye and heart artifacts.
# SSP: https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_ssp.html?highlight=ssp # noqa
# ICA: https://mne-tools.github.io/stable/auto_tutorials/plot_artifacts_correction_ica.html?highlight=ica # noqa
# if you choose ICA, run steps 5a and 6a
# if you choose SSP, run steps 5b and 6b
#
# Currently you cannot use both.

if runicastep == True:
    spatial_filter = 'ica' #'ssp'
    ica_reject = None #"autoreject_local" #None # #{'grad': 400e-12, 'mag': 10e-12}
    ica_algorithm = "picard"
    ica_l_freq = 1.0
    ica_max_iterations = 3000
    ica_n_components = None #0.8
    ica_ecg_threshold = 0.15
    ica_eog_threshold = 2.5


###############################################################################
# GROUP AVERAGE SENSORS
# ---------------------

    interpolate_bads_grand_average = True


# SSP and peak-to-peak rejection
'''spatial_filter = "ssp"
n_proj_eog = dict(n_mag=0, n_grad=0)
n_proj_ecg = dict(n_mag=2, n_grad=2)
#ssp_ecg_channel = "MEG0113"  # ECG channel is not hooked up in this dataset
reject = ssp_reject_ecg = {"grad": 2000e-13, "mag": 5000e-15}'''


#conditions = ["scene_initial", "scene_repeat"]

# Noise estimation
noise_cov = 'ad-hoc' #"emptyroom"