"""
Selective Artifact Removal: Retaining Cardiac Components Only.

This script performs a selective ICA-based cleaning to create a control 
dataset only with cardiac ICs (most of brain activity removed). It removes all independent components EXCEPT those identified as cardiac (ECG-related) artifacts in 'custom_icaartifacts_schmidt.py'.

Purpose:
- To be used for control analyses (e.g., assessing the impact of cardiac 
  artifacts on the estimation of the aperiodic exponent).
- Effectively isolates the cardiac signature within the MEG sensor space 
  by filtering out both neural activity and non-cardiac noise.

Processing Steps:
1. Loads the ICA classification results (.tsv).
2. Identifies components whose 'status_description' contains 'ECG'.
3. Defines all other components (neural, ocular, etc.) as 'badics'.
4. Applies the ICA solution to the raw data to remove everything but 
   the isolated cardiac signal.
5. Saves the result with the 'allbutecg04' processing tag.

Author: Maité Crespo García
Affiliation: MRC Cognition and Brain Sciences Unit, Cambridge, UK
Date: 21-Oct-2025
"""

# Imports
import argparse
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

trans = True
zmm = 44 # destination z coordinate head position in mm

overwrite = False

icselection = 'ecg04eog08'
proc =  'filt'
newicselection = 'allbutecg04' # name of the processing stage that will be used

# ---- Directories and files ----
load_deriv_folder = 'mne-bids-pipeline_stier'
if trans:
    save_deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    save_deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz'

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
logfile = os.path.join(logdir, f'custom_remove_icaartifacts.log')
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
        #print(f'Processing subject {id}...')
        armx = subjectsdf.loc[id,'arm']

        for phase in phases_to_process:
            #print(f'Processing phase {phase}...')
            t0 = time.time()

            # ---- Define the outcome files ----
            bids_project_folder = f'BIDS_long_{phase}_{task}_arm{armx}'
            load_deriv_dir = os.path.join(maindir, bids_project_folder,
                    'derivatives', save_deriv_folder)            
            loaddir = os.path.join(load_deriv_dir, 'sub-'+id, 'meg')

            # ---- Outcome file: ICA cleaned epochs ----
            cleanrawfilename = f'sub-{id}_task-{task}_run-01_proc-{proc+newicselection}_raw.fif'
            cleanrawfile = os.path.join(loaddir, cleanrawfilename)
            if os.path.exists(cleanrawfile) and not overwrite:
                msg = f'Subject {id} in phase {phase} already has cleaned raw file. Skipping.'
                print(msg)
                logger.info(msg)
                continue

            # ---- Define tsv file with the components and artifact labels ----
            tsvfilename = f'sub-{id}_task-{task}_proc-ica_desc-{icselection}_components.tsv'
            tsvfile = os.path.join(loaddir, tsvfilename)

            if not os.path.exists(tsvfile):
                msg = f'Subject {id} in phase {phase} does not have an ICA components selection tsv file. Skipping.'
                print(msg)
                logger.warning(msg)
                continue

            # ---- Define ica file with the fitted ica ----
            icafitfilename = f'sub-{id}_task-{task}_proc-icafit_ica.fif'
            icafitfile = os.path.join(loaddir, icafitfilename) 

            if not os.path.exists(icafitfile):
                msg = f'Subject {id} in phase {phase} does not have an ICA fit file. Skipping.'
                print(msg)
                logger.warning(msg)
                continue

            # ---- Define the raw file to be cleaned ----
            rawfilename = f'sub-{id}_task-{task}_run-01_proc-{proc}_raw.fif'
            rawfile = os.path.join(loaddir, rawfilename)
            if not os.path.exists(rawfile):
                msg = f'Subject {id} in phase {phase} does not have a raw file to be cleaned. Skipping.'
                print(msg)
                logger.warning(msg)
                continue    

            # ---- Read the tsv file with the ica labels generated by the correlation-threshold method ----
            compdf = pd.read_csv(tsvfile, sep='\t').set_index('component')

            try:
                ecgics = compdf[compdf.status_description.str.contains('ECG', na=False)].index.to_list()
            except:
                print('No ECG components identified.')  
                pd.DataFrame().to_csv(os.path.join(loaddir,'error.txt'))
                continue

            if len(ecgics) == 0:
                msg = f'Subject {id} in phase {phase} does not have ECG components identified. Skipping.'
                print(msg)
                logger.warning(msg)
                continue

            badics = [item for item in compdf.index.to_list() if item not in ecgics]


            # read the ica fit output generated by the pipeline
            ica = mne.preprocessing.read_ica(icafitfile)
            ica.exclude = [] # just to be sure
                    
            raw = mne.io.read_raw_fif(rawfile, preload=True)      

            # remove IC artifacts identified with the correlation-threshold method (outside mne-bids pipeline)
            cleanraw = ica.apply(
                raw, 
                include=None, 
                exclude=badics, 
                n_pca_components=None, 
                start=None, 
                stop=None, 
                on_baseline='warn', 
                verbose=False # 'WARNING'
            )

            # Save the cleaned data (this will be used to compute the power spectra)
            cleanraw.save(cleanrawfile, overwrite=True)

            t1 = time.time()
            deltat = t1 - t0                    
            msg = f'Subject {id} s PSD computed in {deltat:.2f} seconds.'
            print(msg)
            logger.info(msg)

if __name__ == "__main__":
    main()