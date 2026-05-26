"""
Normalized Empty Room Power Calculation for Spectral Peaks.

This script quantifies empty room power specifically within the 
frequency ranges where neural peaks were detected in the associated rest MEG data. It 
calculates a log-difference between empty room power and the subject's 
aperiodic power. Then saves these values in a TSV file for each subject, phase and MEG sensor type.

Purpose:
- To create a 'Noise-to-Aperiodic-Signal' ratio for every detected peak.
- This allows for a high-precision control analysis: determining if 
  longitudinal changes in oscillatory power are contaminated by 
  frequency-specific environmental fluctuations.
- Normalizes empty room power to the subject's aperiodic baseline to 
  ensure the noise estimate is in the same mathematical 'space' as the 
  subject's periodic parameters.

Processing Steps:
1. Loads the subject's peak frequencies/widths (from 'specparam_rest.py').
2. Loads the subject's aperiodic-only PSD (from 'specparam_components.py').
3. Loads the full Empty Room PSD (from 'psd_emptyroom.py').
4. For every peak detected in the subject:
   - Identifies the frequency range (center frequency ± 2*bandwidth).
   - Calculates mean linear power in that range for both Empty Room and 
     Aperiodic Rest data.
   - Computes: log10(Empty Room Power) - log10(Aperiodic Rest Power).
5. Exports a TSV of 'relative noise power' for each subject peak.

Author: Maité Crespo García
Affiliation: MRC Cognition and Brain Sciences Unit, Cambridge, UK
Date: 12-Jan-2026 (last modified)
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
phases = ['p2', 'p5']
arms = [1, 2]
lfreq = 0.1 # Hz, high-pass filter cutoff frequency. 
hfreq = 145.0 # Hz, low-pass filter cutoff frequency. 
fsample = 300.0 # Hz, resampling frequency.
frange = f"{round(lfreq, 1)}-{int(hfreq)}Hz"
megtypes = ['mag', 'grad'] # list of MEG sensor types to process. Can be any combination of 'mag', 'grad', and 'eeg'.

er_task = 'noise' # task label for empty room data in BIDS filenames

trans = True # Whether to use head transformation or not for rest data
er_trans = False # Whether to use head transformation or not for empty room data
zmm = 44 # destination z coordinate head position in mm

# Processed data to use for extraction of aperiodic parameters
icselection = 'ecg04eog08' #'allbutecg04' # 'eog08' # 
proc = 'filt' + icselection #'sss' #'clean'
er_proc = 'filt'  # 'sss'

overwrite = True # whether to overwrite existing files

cropdata = 532 # seconds for rest data
er_cropdata = 50 # seconds for empty room data

component = 'aperiodic'  # 'peak' #
er_component = 'total' #'peak' #'aperiodic'  #

# --- Package used for aperiodic fitting ---
package = 'specparam' #'fooof' # irasa

fitting_param = 'finley' #'schmidt' #'oursv2' 
jsonfile = os.path.join(f'aperiodic_fitting_params_{fitting_param}.json')

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
er_psddesc = f'dur{er_cropdata}sepo{epoch_duration}s{powmethod}'


# --- Directories and files ---

if trans:
    rest_deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    rest_deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz'

if trans:
    psd_deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    psd_deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz'

if er_trans:
    er_deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    er_deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz'

if er_trans:
    raise ValueError('Are you sure you want to use head transformation for empty room data? This is not recommended.')
    er_psd_deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    er_psd_deriv_folder = f'mne-bids-pipeline_erm_filt{frange}_fs{int(fsample)}Hz'

# ---- Logging ----
# Directory where the log file will be saved
taskref = 'rest'
phaseref = 'p5' 
armref = 1

save_deriv_root = os.path.join(maindir, bids_project_folder,
                        'derivatives', er_deriv_folder)

logdir = os.path.join(save_deriv_root, 'logfiles')
if not os.path.exists(logdir): os.makedirs(logdir)

# Set up log file
logfile = os.path.join(logdir, f'aperiodic_long_comp_totsv_{component}_{package}_{proc}_{frangestr}.log')
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
                er_derivdir = os.path.join(maindir, bids_project_folder,
                        'derivatives', er_deriv_folder)
                er_megdir = os.path.join(er_derivdir, 'sub-'+id, 'meg')

                # --- Define the output tsv file with empty room power parameters ---
                outpeaksfilename = f'sub-{id}_task-{er_task}_proc-{er_proc}_desc-{er_psddesc}{megtype}{fitting_param}_specparam_{er_component}minus{component}{task}.tsv'
                outpeaksfile = os.path.join(er_megdir, outpeaksfilename)               

                # ---- Check if the output file already exists ----
                if os.path.isfile(outpeaksfile) and not overwrite:
                    msg = f'Subject {id} in phase {phase}, MEG type {megtype}: {er_task} power parameters relative to aperiodic {task} file already exists at {outpeaksfile}, skipping.'
                    print(msg)
                    logger.info(msg)
                    continue

                if er_component == 'total':
                    # ---- Define the input directory for empty room psd files files ----
                    er_derivdir = os.path.join(maindir, bids_project_folder,
                            'derivatives', er_psd_deriv_folder)
                    er_megdir = os.path.join(er_derivdir, 'sub-'+id, 'meg')

                    # --- Define the input file for empty room total PSD data ---
                    emptyroomfilename = f'sub-{id}_task-{er_task}_proc-{er_proc}_desc-{er_psddesc}_psd.hdf5'
                    emptyroomfile = os.path.join(er_megdir, emptyroomfilename)

                rest_derivdir = os.path.join(maindir, bids_project_folder,
                        'derivatives', rest_deriv_folder)
                rest_megdir = os.path.join(rest_derivdir, 'sub-'+id, 'meg')

                # --- Check if the empty room input file exists ---
                if not os.path.isfile(emptyroomfile):
                    msg = f'Subject {id} in phase {phase}, MEG type {megtype}: {er_component} psd not found, skipping. Please run aperiodic_long_emptyroom_components.py or psd_long_emptyroom_2s.py first!'
                    print(msg)
                    logger.warning(msg)
                    continue

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
                print(f'Loading rest {component} data from {compfile}...')
                comp_data = np.load(compfile, allow_pickle=True).item()
                rest_spectra = comp_data['spectra']  # shape (n_channels, n_frequencies)
                rest_freqs = comp_data['freqs']      # shape (n_frequencies,)
                rest_ch_names = comp_data['channels'] # list of channel names
                space = comp_data['space']      # sensor space info
                print(f'Rest {component} data loaded.')
                del comp_data # free memory


                # --- Load the empty room data ---
                if er_component == 'total':
                    # needs to interpolate 23.4 Hz noise
                    # --- Read the power spectrum data ---
                    psd = mne.time_frequency.read_spectrum(emptyroomfile)

                    # --- Get the psds for each epoch and channel ---
                    er_spectra, er_freqs = psd.pick(megtype).get_data(return_freqs=True)
                    
                    # --- Save the channels names ---
                    er_ch_names = psd.ch_names
                    del psd # free memory

                    # --- Average PSD across epochs ---
                    er_spectra_avg = np.average(er_spectra, axis=0)
                    del er_spectra # free memory

                    # ---- Interpolate 23.4 Hz (Golan's) noise ----
                    if package == 'specparam':
                        _, er_spectra_int = interpolate_spectra(er_freqs, er_spectra_avg, [21.9, 23.9]) # channels x frequencies
                        del er_spectra_avg # free memory
                elif er_component == 'peak':
                    # --- Load the empty room periodic component data ---
                    print(f'Loading empty room {er_component} data from {emptyroomfile}...')
                    er_comp_data = np.load(emptyroomfile, allow_pickle=True).item()
                    er_spectra_int = er_comp_data['spectra']  # shape (n_channels, n_frequencies)
                    er_freqs = er_comp_data['freqs']      # shape (n_frequencies,)
                    er_ch_names = er_comp_data['channels'] # list of channel names
                    space = er_comp_data['space']      # sensor space info
                    print(f'Empty room {er_component} data loaded.')
                    del er_comp_data # free memory


                # --- Load the subject's rest peak parameters to get the peak bands ---            
                df_rest = pd.read_csv(restpeaksfile, sep='\t', index_col='channel')

                # --- Create a copy of the rest dataframe to fill with empty room power ---
                df_rest_out = df_rest.copy()

                # Fill in with nans the columns that are not peak frequencies, band widths or channel info
                remove_cols = [col for col in df_rest_out.columns if col not in ['channel', 'channel_name'] and not col.startswith('cf_') and not col.startswith('bw_')]
                print(f'Removing columns: {remove_cols}')
                df_rest_out[remove_cols] = np.nan

                del df_rest # free memory

                # --- Compute empty room power within the subject's rest peak bands ---
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
                                rest_freq_indices = np.where((rest_freqs >= f_lower) & (rest_freqs <= f_upper))[0]

                                er_freq_indices = np.where((er_freqs >= f_lower) & (er_freqs <= f_upper))[0]

                                if (len(rest_freq_indices) > 0) & (len(er_freq_indices) > 0):
                                    # Get channel index
                                    rest_ch_index = rest_ch_names.index(df_rest_out.loc[ch, 'channel_name'])

                                    # Compute mean power within the band
                                    rest_band_power = np.mean(np.mean(rest_spectra[rest_ch_index, rest_freq_indices]))

                                    er_ch_index = er_ch_names.index(df_rest_out.loc[ch, 'channel_name'])

                                    er_band_power = np.mean(np.mean(er_spectra_int[er_ch_index,er_freq_indices]))
                                    
                                    # Store in dataframe
                                    if er_band_power > 0:                                    
                                        df_rest_out.loc[ch, pw_col] = np.log10(er_band_power) - np.log10(rest_band_power)
                                                                           
                                    del rest_band_power, er_band_power # free memory
                                    
                                else:
                                    df_rest_out.loc[ch, pw_col] = np.nan # unnecessary, but for clarity
                            else:
                                df_rest_out.loc[ch, pw_col] = np.nan # unnecessary, but for clarity

                # --- Save the empty room power parameters to a TSV file ---
                df_rest_out.to_csv(outpeaksfile, sep='\t', index_label='channel')
                del er_spectra_int # free memory
                del rest_spectra # free memory
                del df_rest_out # free memory
                
                t1 = time.time()   
                deltat = t1 - t0
                msg = f'Subject {id} in phase {phase} empty room power parameters computed in {deltat:.2f} seconds.'
                print(msg)
                logger.info(msg)
            

if __name__ == "__main__":
    main()