# psd_long_emptyroom_2s.py
"""
psd_long_emptyroom_2s.py

Description: This script is used to compute the power spectral density (PSD) on empty room recordings assigned to MEG longitudinal data. On 2 s epochs.

Author: Maité Crespo García
Affiliation: MRC Cognition and Brain Sciences Unit, Cambridge, UK
Date: 12-Nov-2025 (last modified)
"""

# Imports
import argparse
import mne
import os
import pandas as pd
import sys
import time
import logging
logger = logging.getLogger(__name__)

if os.name == 'nt':
    cfgdir = r"U:\Documents\CamCAN\code\maipy"
else:
    cfgdir = "/imaging/camcan/sandbox/mc06/code/maipy"

sys.path.insert(1, cfgdir)
import mcgdirs as dirs

# ---- Main variables ----
phases = ['p2', 'p5']
task = 'noise' #'emptyroom'
pipver = 'erm' #'stier'
arms = [1, 2]

lfreq = 0.1 #Hz
hfreq = 145.0 #Hz
fsample = 300.0 #Hz
frange = f"{round(lfreq, 1)}-{int(hfreq)}Hz"

epochdur = 2  # seconds

fres = 0.1 if epochdur==10 else (0.5 if epochdur==2 else None) # Frequency resolution in Hz
#sfres = f'fres{str(fres).replace('.','p')}Hz' if fres != 0.1 else ''
# fres is redundant with epochdur, so not included in the filename

trans = False # Whether to use head transformation or not
zmm = 44 # destination z coordinate head position in mm

proc = 'filt' #'sss' #  
overwrite = True

cropdata = 50

powmethod = 'welch' #'multitaper'  # 
powabbr = 'WL' if powmethod=='welch' else ('MT' if powmethod=='multitaper' else '')

exclude_subjects = ['CC420094', 'CC520552', 'CC520562', 'CC520597'] #

# ---- Directories and files ----
#load_deriv_folder = f'mne-bids-pipeline_{pipver}'
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

logdir = os.path.join(save_deriv_root, 'logfiles')
if not os.path.exists(logdir): os.makedirs(logdir)

# Set up log file
logfile = os.path.join(logdir, f'psd_long_emptyroom_2s_{proc}.log')
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)

# ---- File with subjects and arms ----
subjlistfile = os.path.join(dirs.mysandboxdatadir,f'meglong_{taskref}_subjects.tsv')

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

            # ---- Define the output file (psd) ----
            bids_project_folder = f'BIDS_long_{phase}_{taskref}_arm{armx}'
            derivdir = os.path.join(dirs.mysandboxdatadir, bids_project_folder,
                    'derivatives', save_deriv_folder)
            megdir = os.path.join(derivdir, 'sub-'+id, 'meg')

            
            savefilename = f'sub-{id}_task-{task}_proc-{proc}_desc-dur{cropdata}sepo{epochdur}s{powabbr}_psd.hdf5'
            psdfile = os.path.join(megdir, savefilename)

            # ---- Check if PSD already computed ----
            if os.path.exists(psdfile) and not overwrite:
                msg = f'Subject {id} in phase {phase}: PSD already computed, skipping.'
                print(msg)
                logger.info(msg)
                continue
            

            # ---- Define file with maxfiltered empty room data ----
            load_derivdir = os.path.join(dirs.mysandboxdatadir, bids_project_folder,
                    'derivatives', save_deriv_folder)
            load_megdir = os.path.join(load_derivdir, 'sub-'+id, 'meg')

            cleanfilename = f'sub-{id}_task-{task}_proc-{proc}_raw.fif'
            cleanfile = os.path.join(load_megdir, cleanfilename)
            
            # ---- Check if clean file exists ----
            if not os.path.exists(cleanfile):
                msg = f'Subject {id} in phase {phase}: clean file does not exist, skipping.'
                print(msg)
                logger.warning(msg)
                continue

            # ---- Compute PSD ----
            raw = mne.io.read_raw_fif(cleanfile, preload=True)
            
            # ---- Check data duration ----
            tmax = cropdata - 0.004  # 10 seconds minus 4 ms for the last sample
            if raw.duration < tmax:
                msg = f'Empty room data for subject {id} in phase {phase} is shorter than expected ({raw.duration:.2f} seconds). Skipping.'
                print(msg)
                logger.error(msg)
                continue

            # ---- Crop the data to the desired duration ----
            '''if raw.info['sfreq'] != fsample:
                raw.resample(fsample)
                raw = raw.resample(
                    sfreq=250, #raw.info["sfreq"] / 4.0,
                    method="polyphase",
                    verbose=True,
                )'''
            raw.crop(tmin=0, tmax=tmax)
            
            # ---- Create epochs ----
            epochs = mne.make_fixed_length_epochs(raw, duration=epochdur, preload=False)
            
            # ---- Compute PSD for meg data ----
            if powmethod == 'multitaper':
                psd = epochs.compute_psd(
                    method=powmethod, fmin=0.5, fmax=145, picks='meg', exclude='bads', 
                    bandwidth=1, output='power', tmin=0, tmax=epochdur, adaptive=True, 
                    normalization='full'
                    )
            elif powmethod == 'welch':
                psd = epochs.compute_psd(
                    method=powmethod, fmin=0.5, fmax=145, picks='meg', exclude='bads', 
                    output='power', tmin=0, tmax=epochdur,
                    window='hamming', n_fft=int(fsample/fres)
                )

                '''psd = epochs.compute_psd(
                    method=powmethod, fmin=1, fmax=145, picks='meg', exclude='bads', 
                    output='power', tmin=0, tmax=epochdur,
                    remove_dc=True,
                    n_per_seg=int(raw.info['sfreq']),
                    n_overlap=0, #int(raw.info['sfreq']/2),
                    window='hamming', n_fft=int(2*raw.info['sfreq']/1) #
                )'''

            '''rawpsd = raw.compute_psd(
                    method=powmethod, fmin=1, fmax=145, picks='meg', exclude='bads', 
                    output='power', tmin=0, tmax=None,
                    remove_dc=True,
                    n_per_seg=int(raw.info['sfreq']),
                    n_overlap=0,
                    window='hamming', n_fft=int(raw.info['sfreq']/fres)#, method_kw={'average':None}
                )'''

            # Save the PSD to an HDF5 file      
            psd.save(psdfile, overwrite=True)   

            t1 = time.time()
            deltat = t1 - t0
            msg = f'Subject {id} in phase {phase}: PSD of empty room was computed in {deltat:.2f} seconds.'
            print(msg)
            logger.info(msg)

if __name__ == "__main__":
    main()