# psd_ecg_long_2s.py
"""
psd_ecg_long_2s_likeMEG.py

Description: This script is used to compute the power spectral density (PSD) of the ECG channel for MEG longitudinal data. On 2 s epochs.

Author: Maité Crespo García
Affiliation: MRC Cognition and Brain Sciences Unit, Cambridge, UK
Date: 23-Dec-2025
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
task = 'rest'
pipver = 'stier'
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

channelselection = 'ECG'  # 'meg' #
proc =  'sss'  # 'raw' #

overwrite = False

cropdata = 532

powmethod = 'welch' #'multitaper'  # 
powabbr = 'WL' if powmethod=='welch' else ('MT' if powmethod=='multitaper' else '')

exclude_subjects = ['CC520552', 'CC520597'] #'CC420094', 'CC520562', 

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

logdir = os.path.join(save_deriv_root, 'logfiles')
if not os.path.exists(logdir): os.makedirs(logdir)

# Set up log file
logfile = os.path.join(logdir, f'psd_ecg_long_{proc}likemeg.log')
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

            # ---- Define the output file (psd) ----
            bids_project_folder = f'BIDS_long_{phase}_{task}_arm{armx}'
            derivdir = os.path.join(dirs.mysandboxdatadir, bids_project_folder,
                    'derivatives', save_deriv_folder)
            megdir = os.path.join(derivdir, 'sub-'+id, 'meg')

            savefilename = f'sub-{id}_task-{task}_proc-{proc}{channelselection}likemeg_desc-dur{cropdata}sepo{epochdur}s{powabbr}_psd.hdf5'
            psdfile = os.path.join(megdir, savefilename)

            # ---- Check if PSD already computed ----
            if os.path.exists(psdfile) and not overwrite:
                msg = f'Subject {id} in phase {phase}: PSD already computed, skipping.'
                print(msg)
                logger.info(msg)
                continue

            # ---- Define file with clean data (after removing ICA artifacts) ----
            cleanfilename = f'sub-{id}_task-{task}_run-01_proc-{proc}_raw.fif'
            cleanfile = os.path.join(megdir, cleanfilename)
            
            # ---- Check if clean file exists ----
            if not os.path.exists(cleanfile):
                msg = f'Subject {id} in phase {phase}: clean file does not exist, skipping.'
                print(msg)
                logger.warning(msg)
                pd.DataFrame().to_csv(os.path.join(megdir, 'error.txt'))
                continue

            # ---- Compute PSD ----
            raw = mne.io.read_raw_fif(cleanfile, preload=True).pick('ecg')
            
            # ---- Check data duration ----
            tmax = cropdata - 0.004  # 10 seconds minus 4 ms for the last sample
            if raw.duration < tmax:
                msg = f'Raw data for subject {id} in phase {phase} is shorter than expected ({raw.duration:.2f} seconds). Skipping.'
                print(msg)
                logger.error(msg)
                continue

            # ---- Preprocessing: filter the data ----
            
            # -------------------------------
            # 2. Baseline wandering correction
            # -------------------------------
            # High-pass filter at 0.5 Hz (sometimes 0.3 Hz if needed)
            raw = raw.filter(l_freq=0.1, h_freq=145.0, picks="ecg")

            # -------------------------------
            # 3. Line noise removal
            # -------------------------------
            # Notch filter at 50 Hz (Europe) or 60 Hz (US)
            # Add harmonics if necessary, e.g., [50, 100] or [60, 120]
            raw = raw.notch_filter(freqs=[50, 100, 150], picks="ecg")

            # -------------------------------
            # 4. Resample the data if necessary
            # -------------------------------
            if raw.info['sfreq'] != fsample:
                raw.resample(sfreq=fsample, npad='auto')
                msg = f'Subject {id} in phase {phase}: data resampled to {fsample} Hz.'
                print(msg)
                logger.info(msg)

            # ---- Crop the data to the desired duration ----
            raw.crop(tmin=0, tmax=tmax)

            # ---- Create epochs ----
            epochs = mne.make_fixed_length_epochs(raw, duration=epochdur, preload=False)
            
            # ---- Compute PSD for meg data ----
            if powmethod == 'multitaper':
                psd = epochs.compute_psd(
                    method=powmethod, fmin=0.5, fmax=145, picks='ecg', exclude='bads', 
                    bandwidth=1, output='power', tmin=0, tmax=epochdur, adaptive=True, 
                    normalization='full'
                    )
            elif powmethod == 'welch':
                psd = epochs.compute_psd(
                    method=powmethod, fmin=0.5, fmax=145, picks='ecg', exclude='bads', 
                    output='power', tmin=0, tmax=epochdur,
                    window='hamming', n_fft=int(fsample/fres)
                )
                '''psd = epochs.compute_psd(
                    method=powmethod, fmin=1, fmax=145, picks='ecg', exclude='bads', 
                    output='power', tmin=0, tmax=epochdur,
                    remove_dc=True,
                    n_per_seg=int(raw.info['sfreq']),
                    n_overlap=0, #int(raw.info['sfreq']/2),
                    window='hamming', n_fft=int(2*raw.info['sfreq']/1) #
                )'''

            # Save the PSD to an HDF5 file      
            psd.save(psdfile, overwrite=True)
            if os.path.exists(os.path.join(megdir, 'error.txt')):
                os.remove(os.path.join(megdir, 'error.txt')) 

            t1 = time.time()
            deltat = t1 - t0
            msg = f'Subject {id} in phase {phase}: PSD computed in {deltat:.2f} seconds.'
            print(msg)
            logger.info(msg)

if __name__ == "__main__":
    main()