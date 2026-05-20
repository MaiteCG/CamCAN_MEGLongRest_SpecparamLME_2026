"""
Relative Cardiac Band Power Quantification mapped to Neural Peaks.

This script extracts the mean linear power of the ECG channel within the precise 
frequency boundaries (center frequency ± 2*bandwidth) where periodic peaks 
were detected in the subject's MEG data. It normalizes this cardiac band power 
by the total ECG power within the fitting range (2-40 Hz) and applies a 
log10 transformation.

Purpose:
- To create a 'Relative Cardiac Power' metric custom-tailored to each sensor's 
  individual neural peak profile.
- This serves as a high-precision control for statistical modeling (e.g., LME), 
  confirming that age-related shifts or longitudinal changes in neural peak 
  power are completely independent of physiological cardiac dynamics or 
  incomplete ICA cleanup.

Processing Steps:
1. Loads the peak configurations (center frequencies and bandwidths) identified 
   in the neural data ('specparam_rest.py').
2. Loads the full ECG Power Spectral Density ('psd_ecg.py').
3. Interpolates line noise frequencies (50 Hz and harmonics) in the ECG data.
4. For every peak detected in the neural data:
   - Sets boundaries around the peak (f_low = cf - 2 * bw; f_high = cf + 2 * bw).
   - Computes the mean linear power of the ECG channel within this band.
   - Computes the total linear power of the ECG channel across the 2-40 Hz range.
   - Calculates the log relative power: log10(Band Power / Total Power).
5. Saves a sensor-by-peak master array to a .tsv file inside the derivatives folder.

Author: Maité Crespo García
Affiliation: MRC Cognition and Brain Sciences Unit, Cambridge, UK
Date: 19-May-2026 (last modified)
"""

# Imports
import argparse
from specparam.utils.spectral import interpolate_spectra
import json
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
import time

# =============================================================================
# --- Project-specific Settings ---
# =============================================================================
maindir = '' # path where the BIDS project folder is stored, e.g. '/home/CamCAN/data/'
bids_project_folder = '' # Name of the BIDS project folder, e.g. 'BIDS_long_P2_rest_arm1'

# --- Pipeline-specific variables ---
pipver = '' # any string to identify the version of the pipeline, e.g. 'v01'.
task = 'rest'
megtypes = ['mag', 'grad'] # list of MEG sensor types to process. Can be any combination of 'mag', 'grad', and 'eeg'.
#tasks = ['rest', 'emptyroom']
phases = ['p2', 'p5']
arms = [1, 2]

lfreq = 0.1 #Hz
hfreq = 145.0 #Hz
fsample = 300.0 #Hz
frange = f"{round(lfreq, 1)}-{int(hfreq)}Hz"

trans = True # Whether to use head transformation or not for rest data
ecg_trans = False # Whether to use head transformation or not for empty room data
zmm = 44 # destination z coordinate head position in mm

# Processed data to use for extraction of aperiodic parameters
icselection = 'ecg04eog08' #'allbutecg04' # 'eog08' # 
proc = 'filt' + icselection #'sss' #'clean'
chanselection = 'ECG'
ecg_proc = 'sss' + chanselection
likemeg = True # whether to process ECG data in the same way as MEG data (filters)
slikemeg = 'likemeg' if likemeg else ''

overwrite = False # whether to overwrite existing files

cropdata = 532 # seconds for rest data

#component = 'aperiodic'  # 'peak' #
ecg_component = 'total' #'peak' #'aperiodic'  #

# --- Package used for aperiodic fitting ---
package = 'specparam' #'fooof' # irasa

fitting_param = 'finley' #'schmidt' #'oursv2' 
jsonfile = os.path.join(dirs.homecamcancodedir, 'sens', f'aperiodic_fitting_params_{fitting_param}.json')

if os.path.exists(jsonfile):
    with open(jsonfile) as json_file:
        json_dict = json.load(json_file)
    
    epoch_duration = json_dict['epoch_duration']
    powmethod = json_dict['powmethod']
    fres = json_dict['fres']
    freq_range = json_dict['freq_range']
    peak_width_limits = json_dict['peak_width_limits']
    min_peak_height = json_dict['min_peak_height']
    peak_threshold = json_dict['peak_threshold']
    max_n_peaks = json_dict['max_n_peaks']
    frangestr = f'{freq_range[0]}-{freq_range[1]}Hz'
else:

    if fitting_param == 'oursv1':
        # ---- Variables from PSD computation ----
        epoch_duration = 10 # seconds
        powmethod = 'MT' # 'MT' for multitaper, 'W' for welch #
        
        fres = 0.1 # frequency resolution in Hz

        # ---- Variables for the aperiodic fitting ----
        # Define the frequency range to fit
        freq_range = [3, 35] # following suggestion in documentation #0.5-140 Hz in the paper, but we use 1-48 Hz to avoid 50 Hz noise
        frangestr = f'{freq_range[0]}-{freq_range[1]}Hz'
        peak_width_limits=[1, 6]
        min_peak_height=0.05
        peak_threshold=1.5
        max_n_peaks=6
    else:
        raise ValueError(f'Fitting parameter set "{fitting_param}" not recognized.') 


    fitting_param_dict = {
        'epoch_duration': epoch_duration,
        'powmethod': powmethod,
        'fres': fres,
        'freq_range': freq_range,
        'peak_width_limits': peak_width_limits,
        'min_peak_height': min_peak_height,
        'peak_threshold': peak_threshold,
        'max_n_peaks': max_n_peaks
    }

    json.dump(fitting_param_dict, open(f'aperiodic_fitting_params_{fitting_param}.json', 'w'), indent=4)

psddesc = f'dur{cropdata}sepo{epoch_duration}s{powmethod}'

# --- Directories and files ---

if trans:
    rest_deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    rest_deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz'

if trans:
    psd_deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    psd_deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz'

if False:
    goodepochs_deriv_folder = 'mne-bids-pipeline_stier' # for bad epochs only (for 10s epochs)
else:
    goodepochs_deriv_folder = psd_deriv_folder

if ecg_trans:
    ecg_deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    ecg_deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz'

if ecg_trans:
    raise ValueError('Are you sure you want to use head transformation for empty room data? This is not recommended.')
else:
    ecg_psd_deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz'

# ---- Logging ----
# Directory where the log file will be saved
taskref = 'rest'
phaseref = 'p5' 
armref = 1

save_deriv_root = os.path.join(maindir, bids_project_folder,
                        'derivatives', ecg_deriv_folder)

logdir = os.path.join(save_deriv_root, 'logfiles')
if not os.path.exists(logdir): os.makedirs(logdir)

# Set up log file
logfile = os.path.join(logdir, f'aperiodic_long_ecg_comp_logrelpow_totsv_{ecg_component}_{package}_{proc}{slikemeg}_{frangestr}.log')
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)

# ---- File with subjects and arms ----
subjlistfile = os.path.join(maindir,f'meglong_{taskref}_subjects.tsv')

# Main code
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

    # loop over subjects
    for id in subjects:
        armx = subjectsdf.loc[id,'arm']
        
        for phase in phases_to_process:
            t0 = time.time()

            # Loop over MEG sensor types
            for megtype in megtypes:

                # ---- Define the input/output directory for rest peak power tsv files ----
                bids_project_folder = f'BIDS_long_{phase}_{taskref}_arm{armx}'
                ecg_derivdir = os.path.join(maindir, bids_project_folder,
                        'derivatives', ecg_deriv_folder)
                ecg_megdir = os.path.join(ecg_derivdir, 'sub-'+id, 'meg')

                # --- Define the output tsv file with empty room power parameters ---
                outpeaksfilename = f'sub-{id}_task-{task}_proc-{ecg_proc}{slikemeg}_desc-{psddesc}{megtype}{fitting_param}_specparam_{ecg_component}relpow2-40Hz.tsv'
                outpeaksfile = os.path.join(ecg_megdir, outpeaksfilename)               

                # ---- Check if the output file already exists ----
                if os.path.isfile(outpeaksfile) and not overwrite:
                    msg = f'Subject {id} in phase {phase}, MEG type {megtype}: {task} power parameters relative to aperiodic {task} file already exists at {outpeaksfile}, skipping.'
                    print(msg)
                    logger.info(msg)
                    continue

                if ecg_component == 'total':
                    # ---- Define the input directory for empty room psd files files ----
                    ecg_derivdir = os.path.join(maindir, bids_project_folder,
                            'derivatives', ecg_psd_deriv_folder)
                    ecg_megdir = os.path.join(ecg_derivdir, 'sub-'+id, 'meg')

                    # --- Define the input file for empty room total PSD data ---
                    ecgpsdfilename = f'sub-{id}_task-{task}_proc-{ecg_proc}{slikemeg}_desc-{psddesc}_psd.hdf5'
                    ecgpsdfile = os.path.join(ecg_megdir, ecgpsdfilename)

                elif ecg_component == 'peak':
                    raise ValueError('This should not go this way now.')
                    # ---- Define the input directory for empty room numpy files ----
                    ecg_derivdir = os.path.join(maindir, bids_project_folder,
                            'derivatives', ecg_deriv_folder)
                    ecg_megdir = os.path.join(ecg_derivdir, 'sub-'+id, 'meg')

                    # --- Define the input file for empty room periodic component data ---
                    emptyroomfilename = f'sub-{id}_task-{ecg_task}_proc-{ecg_proc}_desc-{ecg_psddesc}{megtype}{fitting_param}_{package}_{ecg_component}.npy'
                    emptyroomfile = os.path.join(ecg_megdir, emptyroomfilename)
                
                
                rest_derivdir = os.path.join(maindir, bids_project_folder,
                        'derivatives', rest_deriv_folder)
                rest_megdir = os.path.join(rest_derivdir, 'sub-'+id, 'meg')

                # --- Check if the ECG input file exists ---
                if not os.path.isfile(ecgpsdfile):
                    msg = f'Subject {id} in phase {phase}, MEG type {megtype}: {ecg_component} ECG psd not found, skipping. Please run psd_ecg_long_2s.py first!'
                    print(msg)
                    logger.warning(msg)
                    continue

                '''
                if component == 'aperiodic':
                    # --- Define the input file for rest numpy files with periodic component ---
                    compfilename = f'sub-{id}_task-{task}_proc-{proc}_desc-{psddesc}{megtype}{fitting_param}_{package}_{component}.npy'
                    compfile = os.path.join(rest_megdir, compfilename)
                else:
                    raise ValueError(f'Component "{component}" not recognized for this script.')
                
                # --- Check if the subject's rest aperiodic component file exists ---
                if not os.path.isfile(compfile):
                    msg = f'Subject {id} in phase {phase}, MEG type {megtype}: rest {component} component file not found at {compfile}, skipping. Please run aperiodic_long_components.py first!'
                    print(msg)
                    logger.warning(msg)
                    continue
                '''

                # ---- Define the input file for subject's rest peak parameters ---
                restpeaksfilename = f'sub-{id}_task-{task}_proc-{proc}_desc-{psddesc}{megtype}{fitting_param}_specparam.tsv'
                restpeaksfile = os.path.join(rest_megdir, restpeaksfilename)

                # --- Check if the subject's rest peak parameters file exists ---
                if not os.path.isfile(restpeaksfile):
                    msg = f'Subject {id} in phase {phase}, MEG type {megtype}: rest peak parameters file not found at {restpeaksfile}, skipping. Please run aperiodic_long_sp.py first!'
                    print(msg)
                    logger.warning(msg)
                    continue

                # --- Load the rest aperiodic component data ---
                '''
                print(f'Loading rest {component} data from {compfile}...')
                comp_data = np.load(compfile, allow_pickle=True).item()
                rest_spectra = comp_data['spectra']  # shape (n_channels, n_frequencies)
                rest_freqs = comp_data['freqs']      # shape (n_frequencies,)
                rest_ch_names = comp_data['channels'] # list of channel names
                space = comp_data['space']      # sensor space info
                print(f'Rest {component} data loaded.')
                del comp_data # free memory
                '''

                # --- Define the bad epochs file ---
                goodepochs_derivdir = os.path.join(maindir, bids_project_folder,
                        'derivatives', goodepochs_deriv_folder)
                goodepochs_megdir = os.path.join(goodepochs_derivdir, 'sub-'+id, 'meg')

                if False:
                    badepochsfilename = f'sub-{id}_task-{task}_proc-sss_desc-epo{cropdata}_badepochs.npy'
                else:
                    badepochsfilename = f'sub-{id}_task-{task}_proc-sss_desc-dur{cropdata}sepo{epoch_duration}s_badepochs.npy'
                badepochsfile = os.path.join(goodepochs_megdir, badepochsfilename)

                # --- Check if the bad epochs file exists ---
                if not os.path.exists(badepochsfile):
                    msg = f'Subject {id} in phase {phase} does not have a bad epochs file.'
                    print(msg)
                    logger.info(msg)
                    continue


                # --- Load the empty room data ---
                if ecg_component == 'total':
                    # needs to interpolate 23.4 Hz noise
                    # --- Read the power spectrum data ---
                    psd = mne.time_frequency.read_spectrum(ecgpsdfile)

                    # --- Get the psds for each epoch and channel ---
                    ecg_spectra, ecg_freqs = psd.get_data(picks=['ecg'], return_freqs=True)
                    
                    # --- Save the channels names ---
                    ecg_ch_names = psd.ch_names
                    del psd # free memory

                    # --- Read the list of good epochs ---
                    vardict = np.load(badepochsfile, allow_pickle=True).item()
                    good_epochs = vardict['good_epochs']

                    ### Remove the first 30-s of the psd, to wait until the participant has settled in. This is equivalent to removing the first 3 epochs of 10-s, or the first 15 epochs of 2-s.
                    n_epochs_to_remove = int(30/epoch_duration)
                    good_epochs = good_epochs[good_epochs >= n_epochs_to_remove]               

                    # --- Select the good epochs and average PSD across epochs ---
                    ecg_spectra = ecg_spectra[good_epochs,:,:]                    

                    # --- Average PSD across epochs ---
                    ecg_spectra_avg = np.average(ecg_spectra, axis=0)
                    del ecg_spectra # free memory

                    # ---- Interpolate 23.4 Hz (Golan's) noise ----
                    if package == 'specparam':
                        _, ecg_spectra_int = interpolate_spectra(ecg_freqs, ecg_spectra_avg, [21.9, 23.9]) # channels x frequencies
                        del ecg_spectra_avg # free memory

                elif ecg_component == 'peak':
                    raise ValueError('This should not go this way now.')
                    # --- Load the empty room periodic component data ---
                    print(f'Loading empty room {ecg_component} data from {emptyroomfile}...')
                    ecg_comp_data = np.load(emptyroomfile, allow_pickle=True).item()
                    ecg_spectra_int = ecg_comp_data['spectra']  # shape (n_channels, n_frequencies)
                    ecg_freqs = ecg_comp_data['freqs']      # shape (n_frequencies,)
                    ecg_ch_names = ecg_comp_data['channels'] # list of channel names
                    space = ecg_comp_data['space']      # sensor space info
                    print(f'Empty room {ecg_component} data loaded.')
                    del ecg_comp_data # free memory


                # --- Load the subject's rest peak parameters to get the peak bands ---            
                df_rest = pd.read_csv(restpeaksfile, sep='\t', index_col='channel')

                # --- Create a copy of the rest dataframe to fill with empty room power ---
                df_rest_out = df_rest.copy()

                # Fill in with nans the columns that are not peak frequencies, band widths or channel info
                remove_cols = [col for col in df_rest_out.columns if col not in ['channel', 'channel_name'] and not col.startswith('cf_') and not col.startswith('bw_')]
                print(f'Removing columns: {remove_cols}')
                df_rest_out[remove_cols] = np.nan

                del df_rest # free memory

                # --- Compute ECG power within the subject's rest peak bands ---
                # loop over possible peaks
                for p in range(0,6): 
                    cf_col = f'cf_{p}'
                    bw_col = f'bw_{p}'
                    pw_col = f'pw_{p}'

                    if cf_col in df_rest_out.columns and bw_col in df_rest_out.columns:
                        for ch in df_rest_out.index:
                            cf = df_rest_out.loc[ch, cf_col]
                            bw = df_rest_out.loc[ch, bw_col]

                            if not np.isnan(cf) and not np.isnan(bw):
                                f_lower = cf - bw / 2
                                f_upper = cf + bw / 2

                                # Find frequency indices within the band
                                '''rest_freq_indices = np.where((rest_freqs >= f_lower) & (rest_freqs <= f_upper))[0]'''

                                total_freq_indices = np.where((ecg_freqs >= 2) & (ecg_freqs <= 40))[0]

                                ecg_freq_indices = np.where((ecg_freqs >= f_lower) & (ecg_freqs <= f_upper))[0]

                                if (len(total_freq_indices) > 0) & (len(ecg_freq_indices) > 0):
                                    # Get channel index
                                    '''rest_ch_index = rest_ch_names.index(df_rest_out.loc[ch, 'channel_name'])

                                    # Compute mean power within the band
                                    rest_band_power = np.mean(np.mean(rest_spectra[rest_ch_index, rest_freq_indices]))'''

                                    ecg_ch_index = 0 # only one ECG channel

                                    ecg_band_power = np.mean(ecg_spectra_int[ecg_ch_index,ecg_freq_indices])

                                    total_power = np.sum(ecg_spectra_int[ecg_ch_index,total_freq_indices])
                                    
                                    # Store in dataframe
                                    if ecg_band_power > 0:                                    
                                        df_rest_out.loc[ch, pw_col] = np.log10(ecg_band_power/total_power)
                                                                           
                                    del total_power, ecg_band_power # free memory
                                    
                                else:
                                    df_rest_out.loc[ch, pw_col] = np.nan # unnecessary, but for clarity
                            else:
                                df_rest_out.loc[ch, pw_col] = np.nan # unnecessary, but for clarity

                # --- Save the empty room power parameters to a TSV file ---
                df_rest_out.to_csv(outpeaksfile, sep='\t', index_label='channel')
                del ecg_spectra_int # free memory
                #del rest_spectra # free memory
                del df_rest_out # free memory
                
                t1 = time.time()   
                deltat = t1 - t0
                msg = f'Subject {id} in phase {phase} ECG power parameters computed in {deltat:.2f} seconds.'
                print(msg)
                logger.info(msg)
            

if __name__ == "__main__":
    main()