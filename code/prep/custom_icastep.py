"""
Independent Component Analysis (ICA) for Longitudinal MEG Data.

This script executes a customized ICA fitting procedure on filtered MEG data, after removing bad epochs detected with the 'custom_bad_epocs.py' script. It bypasses the standard MNE-BIDS pipeline
ICA to ensure that the ICA fit is applied to data without big movement/muscle artifacts.

Processing Steps:
1. Loads preprocessed, MaxFiltered, and filtered raw data obtained with the automatic pipeline.
2. Applies a 1 Hz high-pass filter (recommended for stable ICA decomposition).
3. Segments data into 10-second fixed-length epochs.
4. Removes epochs previously identified as "bad" using external .npy file (created with custom_bad_epocs.py script).
5. Fits ICA using the 'picard' algorithm (optimized for speed and stability).
6. Saves the ICA solution and cleans up redundant pipeline files.

Author: Maité Crespo García
Affiliation: MRC Cognition and Brain Sciences Unit, Cambridge, UK
Date: 20-Oct-2025
"""

# Imports
import argparse
import gc
import logging
logger = logging.getLogger(__name__)
import mne
import numpy as np
import os
import pandas as pd
import time
from picard import picard

# =============================================================================
# --- Project-specific Settings ---
# =============================================================================
maindir = '' # path where the BIDS project folder is stored, e.g. '/home/CamCAN/data/'
bids_project_folder = '' # Name of the BIDS project folder, e.g. 'BIDS_long_P2_rest_arm1'

task = 'rest'
phases = ['p2', 'p5']
arms = [1, 2]

# --- Pipeline-specific variables ---
pipver = '' # any string to identify the version of the pipeline, e.g. 'v01'.
lfreq = 0.1 # Hz, high-pass filter cutoff frequency. 
hfreq = 145.0 # Hz, low-pass filter cutoff frequency. 
fsample = 300.0 # Hz, resampling frequency.
frange = f"{round(lfreq, 1)}-{int(hfreq)}Hz"

trans = True # True
zmm = 44 # destination z coordinate head position in mm

overwrite = False

# ---- specific parameters of bad epochs detection ----
proc =  'sss' 
cropdata = 532

# ---- Directories and files ----
load_deriv_folder = 'mne-bids-pipeline_stier'
if trans:
    save_deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    save_deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz'

usenew = True
if usenew:
    load_deriv_folder = save_deriv_folder

# ---- Logging ----
# Directory where the log file will be saved
taskref = 'rest'
phaseref = 'p5' 
armref = 1

save_deriv_root = os.path.join(maindir, bids_project_folder,
                        'derivatives', save_deriv_folder)
if not os.path.exists(save_deriv_root): os.makedirs(save_deriv_root)

logdir = os.path.join(save_deriv_root,'logfiles')
if not os.path.exists(logdir): os.makedirs(logdir)

# Set up log file
logfile = os.path.join(logdir, f'custom_icastep.log')
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)

# ---- File with subjects and arms ----
subjlistfile = os.path.join(maindir,f'meglong_{task}_subjects.tsv')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, default=None, dest='subject', action='store')
    parser.add_argument('--phase', type=str, default=None, dest='phase', action='store')
    args = parser.parse_args()
    print(args)
    print(args.subject)
    print(args.phase)

    # Read file with subjects and arms
    subjectsdf = pd.read_csv(subjlistfile, sep='\t').set_index('subject')

    if args.subject is not None:
        subjects = [args.subject]
    else:
        subjects = subjectsdf.index.tolist()

    if args.phase is not None:
        phases_to_process = [args.phase]
    else:
        phases_to_process = phases

    for id in subjects:
        armx = subjectsdf.loc[id,'arm']

        for phase in phases_to_process:
            t0 = time.time()

            # ---- Define the ICA file to be saved ----
            bids_project_folder = f'BIDS_long_{phase}_{task}_arm{armx}'
            load_deriv_dir = os.path.join(maindir, bids_project_folder,
                    'derivatives', save_deriv_folder)            
            loaddir = os.path.join(load_deriv_dir, 'sub-'+id, 'meg')

            icafilename = f'sub-{id}_task-{task}_proc-icafit_ica.fif'
            icafile = os.path.join(loaddir, icafilename)

            if os.path.exists(icafile) and not overwrite:
                msg = f'Subject {id} in phase {phase} already has an ICA file. Skipping.'
                print(msg)
                logger.info(msg)
                continue

            # ---- Define file with good epochs information ----
            #
            goodepochs_derivdir = os.path.join(maindir, bids_project_folder,
                    'derivatives', load_deriv_folder)
            megdir = os.path.join(goodepochs_derivdir, 'sub-'+id, 'meg')

            if usenew:
                badepochsfilename = f'sub-{id}_task-{task}_proc-{proc}_desc-dur{cropdata}sepo10s_badepochs.npy'
            else:
                badepochsfilename = f'sub-{id}_task-{task}_proc-{proc}_desc-epo{cropdata}_badepochs.npy'
            badepochsfile = os.path.join(megdir, badepochsfilename)

            if not os.path.exists(badepochsfile):
                msg = f'Subject {id} in phase {phase} does not have a bad epochs file.'
                print(msg)
                logger.info(msg)
                continue

            # Load good epochs information
            vardict = np.load(badepochsfile, allow_pickle=True).item()
            good_epochs = vardict['good_epochs']

            # ---- Define the filtered raw file ----

            filt_filename = f'sub-{id}_task-{task}_run-01_proc-filt_raw.fif'
            filt_file = os.path.join(loaddir, filt_filename)

            if not os.path.exists(filt_file):
                msg = f'Subject {id} in phase {phase} does not have a filtered raw file.'
                print(msg)
                logger.info(msg)
                continue

            # ---- Load the filtered raw data ----
            raw = mne.io.read_raw_fif(filt_file, preload=True)

            # ---- High-pass filter at 1 Hz for ICA ----
            filt_raw = raw.copy().filter(l_freq=1.0, h_freq=None)
            del raw
            gc.collect() #to force the Python garbage collector to free up the MEG data from RAM immediately.

            # ---- Create epochs of 10s ----
            epochs = mne.make_fixed_length_epochs(filt_raw, duration=10)

            # ---- Drop bad epochs ----
            indices = [i for i in epochs.selection if i not in good_epochs]
            epochs.drop(indices, reason='Bad epochs added from custom_bad_epochs.py')

            # ---- Fit ICA ----
            ica = mne.preprocessing.ICA(n_components=None, method='picard',
                                        max_iter=3000, random_state=42, verbose=False)
            ica.fit(epochs, verbose=False)

            # ---- Save ICA fit ----
            ica.save(icafile, overwrite=True, verbose=False)

            # Delete not useful files from automatic pipeline like cleaned epochs file
            cleaned_epochs_filename = f'sub-{id}_task-{task}_proc-clean_epo.fif'
            cleaned_epochs_file = os.path.join(loaddir, cleaned_epochs_filename)

            if os.path.exists(cleaned_epochs_file):
                os.remove(cleaned_epochs_file)

            t1 = time.time()
            deltat = t1 - t0                    
            msg = f'Subject {id} in phase {phase}: ica fit file computed in {deltat:.2f} seconds.'
            print(msg)
            logger.info(msg)

if __name__ == "__main__":
    main()