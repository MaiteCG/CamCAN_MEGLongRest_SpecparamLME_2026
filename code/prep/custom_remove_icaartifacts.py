
"""
custom_remove_icaartifacts.py

Description: This script is used to remove the ica components that were identified as cardiac or ocular artifacts in a previous step (e.g., custom_icaartifacts_schmidt.py). This step runs outside the automatic MNE-BIDS pipeline, but is applied to the longitudinal MEG data preprocessed initially with the automatic MNE-BIDS pipeline and then with custom scripts.

This script loads the _components.tsv file with the ica components and labels (see also code/preprocessing/stier/README.md).

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
import sys
import time
from picard import picard

if os.name == 'nt':
    cfgdir = r"U:\Documents\CamCAN\code\maipy"
else:
    cfgdir = "/imaging/camcan/sandbox/mc06/code/maipy"

sys.path.insert(1, cfgdir)
import mcgdirs as dirs

# ---- Main variables ----
task = 'rest'
phases = ['p2', 'p5']
pipver = 'stier'
arms = [1, 2]

lfreq = 0.1 #Hz
hfreq = 145.0 #Hz
fsample = 300.0 #Hz
frange = f"{round(lfreq, 1)}-{int(hfreq)}Hz"

trans = True # whether to use head-transformed data or not
zmm = 44 # destination z coordinate head position in mm

overwrite = False

icselection = 'eog08'# 'ecg04eog08' #'ecg04' #
proc =  'filt'

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
bids_project_folder = f'BIDS_long_{phaseref}_{taskref}_arm{armref}'

save_deriv_root = os.path.join(dirs.mysandboxdatadir, bids_project_folder,
                        'derivatives', save_deriv_folder)
if not os.path.exists(save_deriv_root): os.makedirs(save_deriv_root)

logdir = os.path.join(save_deriv_root,'logfiles')
if not os.path.exists(logdir): os.makedirs(logdir)

# Set up log file
logfile = os.path.join(logdir, f'custom_remove_icaartifacts.log')
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)

# ---- File with subjects and arms ----
subjlistfile = os.path.join(dirs.mysandboxdatadir,f'meglong_{task}_subjects.tsv')

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

            # ---- Define the outcome files ----
            bids_project_folder = f'BIDS_long_{phase}_{task}_arm{armx}'
            load_deriv_dir = os.path.join(dirs.mysandboxdatadir, bids_project_folder,
                    'derivatives', save_deriv_folder)            
            loaddir = os.path.join(load_deriv_dir, 'sub-'+id, 'meg')

            # ---- Outcome file: ICA cleaned epochs ----
            cleanrawfilename = f'sub-{id}_task-{task}_run-01_proc-{proc+icselection}_raw.fif'
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

            # find the components that were identified as eog or ecg artifacts
            badics = compdf[compdf.status == 'bad'].index

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
            msg = f'Subject {id} in phase {phase}: ICA artifacts removed in {deltat:.2f} seconds.'
            print(msg)
            logger.info(msg)

if __name__ == "__main__":
    main()