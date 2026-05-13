# aperiodic_long_sp.py
"""
aperiodic_long_sp.py

Description: This script is used to fit the power spectral density (PSD) from MEG longitudinal data with the FOOOF algorithm (as implemented in specparam package). The result of this fitting is the extraction of aperiodic parameters (exponent and offset) as well as periodic parameters (peaks in specific frequency bands). NOTE: First 30-s of the psd are removed.

Author: Maité Crespo García
Affiliation: MRC Cognition and Brain Sciences Unit, Cambridge, UK
Date: 18-Feb-2026 (last modified)
"""

# Imports
import argparse
from fooof import FOOOFGroup
#from fooof.analysis import get_band_peak_fg, get_band_peak_fm
#from fooof.analysis.periodic import get_band_peak
#from fooof.utils import interpolate_spectrum
from specparam.utils.spectral import interpolate_spectra
import json
import logging
logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt
import mne
import numpy as np
import os
import pandas as pd
from specparam import SpectralModel, SpectralGroupModel
import sys
import time

if os.name == 'nt':
    cfgdir = r"U:\Documents\CamCAN\code\maipy"
else:
    cfgdir = "/imaging/camcan/sandbox/mc06/code/maipy"

sys.path.insert(1, cfgdir)
import mcgdirs as dirs

# --- Main global variables ---
pipver = 'stier'
task = 'rest'
#tasks = ['rest', 'emptyroom']
phases = ['p2', 'p5']
arms = [1, 2]
megtypes = ['grad', 'mag']

lfreq = 0.1 #Hz
hfreq = 145.0 #Hz
fsample = 300.0 #Hz
frange = f"{round(lfreq, 1)}-{int(hfreq)}Hz"

trans = True # Whether to use head transformation or not
zmm = 44 # destination z coordinate head position in mm

# Processed data to use for extraction of aperiodic parameters
icselection = 'eog08' #'ecg04eog08' # 'allbutecg04' #
proc = 'filt' + icselection #'sss' #'clean'

overwrite = False # whether to overwrite existing files

dointerpolation = True # False # whether to do interpolation of 23.4 Hz noise
sinterp = '' if dointerpolation else '_nointerp'

knee=False # whether to fit the aperiodic component with a knee (instead of a simple power law, or fixed mode). 
withknee = 'knee' if knee else ''
aperiodic_mode = 'knee' if knee else 'fixed'

cropdata = 532 

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
#sfres = '' if fres == 0.1 else f'fres{fres}Hz'
# fres seems redundant with epoch_duration, so not included in the filename


# packages in Schmidt et al., 2024:
# Power spectra were parameterized across frequency ranges of 0.5–145 Hz. 
# FOOOF models were fit using the following settings: 
# peak width limits: [1 – 6]; max number of peaks: 2; 
# minimum peak height: 0.0; peak threshold: 2.0; aperiodic mode: ‘fixed’.

# Donoghue et al., 2020
# peak_width_limits=[1,6], max_n_peaks=6, min_peak_height=0.05, 
# peak_threshold=1.5, aperiodic_mode=‘fixed’

# --- Directories and files ---

if trans:
    psd_deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    psd_deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz'

if False:
    goodepochs_deriv_folder = 'mne-bids-pipeline_stier' # for bad epochs only (for 10s epochs)
else:
    goodepochs_deriv_folder = psd_deriv_folder

if trans:
    save_deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    save_deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz'

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
logfile = os.path.join(logdir, f'aperiodic_long_{package}_{proc}_{frangestr}{sinterp}.log')
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)

# ---- File with subjects and arms ----
subjlistfile = os.path.join(dirs.mysandboxdatadir,f'meglong_{task}_subjects.tsv')

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

            # ---- Define the output directory ----
            bids_project_folder = f'BIDS_long_{phase}_{task}_arm{armx}'
            save_derivdir = os.path.join(dirs.mysandboxdatadir, bids_project_folder,
                    'derivatives', save_deriv_folder)
            save_megdir = os.path.join(save_derivdir, 'sub-'+id, 'meg')
            if not os.path.exists(save_megdir):
                os.makedirs(save_megdir)

            # --- Define the psd directory and file ---
            psd_derivdir = os.path.join(dirs.mysandboxdatadir, bids_project_folder,
                    'derivatives', psd_deriv_folder)
            psd_megdir = os.path.join(psd_derivdir, 'sub-'+id, 'meg')

            psdfilename = f'sub-{id}_task-{task}_proc-{proc}_desc-{psddesc}_psd.hdf5'
            psdfile = os.path.join(psd_megdir, psdfilename)

            # --- Check if the psd file exists ---
            if not os.path.isfile(psdfile):
                msg = f'Subject {id} in phase {phase}: psd file not found, skipping.'
                print(msg)
                logger.warning(msg)
                pd.DataFrame().to_csv(os.path.join(save_megdir, f'error.txt'))       
                continue

            # --- Define the bad epochs file ---
            goodepochs_derivdir = os.path.join(dirs.mysandboxdatadir, bids_project_folder,
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

            #txtfile = os.path.join(save_megdir, 'done.txt')
            
            # Loop over MEG sensor types
            for megtype in megtypes:

                # --- Define the output file for aperiodic parameters ---
                savefilename = f'sub-{id}_task-{task}_proc-{proc}_desc-{psddesc}{megtype}{fitting_param}{withknee}_{package}{sinterp}.tsv'
                savefile = os.path.join(save_megdir, savefilename)

                # --- Check if the output file already exists ---
                if os.path.isfile(savefile) and not overwrite:
                    msg = f'Subject {id} in phase {phase}, MEG type {megtype}: aperiodic parameters already computed, skipping.'
                    print(msg)
                    logger.info(msg)
                    continue
                
                # --- Read the power spectrum data ---
                psd = mne.time_frequency.read_spectrum(psdfile)

                # --- Get the psds for each epoch and channel ---
                spectra, freqs = psd.pick(megtype).get_data(return_freqs=True)
                
                # --- Save the channels names ---
                ch_names = psd.ch_names
                del psd # free memory

                # --- Read the list of good epochs ---
                vardict = np.load(badepochsfile, allow_pickle=True).item()
                good_epochs = vardict['good_epochs']

                ### Remove the first 30-s of the psd, to wait until the participant has settled in. This is equivalent to removing the first 3 epochs of 10-s, or the first 15 epochs of 2-s.
                n_epochs_to_remove = int(30/epoch_duration)
                good_epochs = good_epochs[good_epochs >= n_epochs_to_remove]

                # --- Select the good epochs and average PSD across epochs ---
                spectra = spectra[good_epochs,:,:]
                spectra = np.average(spectra, axis=0)

                # ---- Interpolate 23.4 Hz (Golan's) noise ----
                if package == 'specparam' and dointerpolation:
                    freqs, spectra = interpolate_spectra(freqs, spectra, [21.9, 23.9]) # channels x frequencies

                '''
                for i in range(spectra.shape[0]):
                powers = spectra[i, :]  # Get the power spectrum for each channel
                freqs, powers = interpolate_spectrum(freqs, powers, [21.9, 23.9])
                spectra[i, :] = powers  # Update the spectra with interpolated values
                '''
                if fres > 0.1 and epoch_duration == 10:                
                    spectra = mne.filter.resample(spectra, down=int(fres/0.1), pad='line', npad='auto', axis=1, method='polyphase')
                    freqs = np.linspace(freqs[0], freqs[-1], num=int((freqs[-1]-freqs[0])/fres)+1)
                    freqs = freqs[:len(spectra[0,:])]  # Adjust the frequencies to match the new data shape

                if package == 'specparam':
                    # Initialize a SpectralGroupModel
                    fg = SpectralGroupModel(peak_width_limits=peak_width_limits, min_peak_height=min_peak_height,
                    peak_threshold=peak_threshold, max_n_peaks=max_n_peaks, 
                    aperiodic_mode = aperiodic_mode, verbose=False)
                    
                elif package == 'fooof':
                    raise NotImplementedError('package "fooof" not implemented in this script. The code below is just as a reference for what we did in other scripts with the FOOOF package. It is now recommended to use the specparam package instead.')
                    # Initialize a FOOOFGroup object, with desired settings
                    '''
                    fg = FOOOFGroup(peak_width_limits=peak_width_limits, min_peak_height=min_peak_height,
                    peak_threshold=peak_threshold, max_n_peaks=max_n_peaks, 
                    aperiodic_mode = 'fixed', verbose=False)
                    '''
                
                # This line was because some values were negative, perhaps after the resampling step. This was causing an error in the fitting, when transforming the PSD to log scale. I substitute the negative values with the minimum found in positive values.
                for i in range(spectra.shape[0]):
                    if any(spectra[i,:]<0):
                        posidx = np.asarray(spectra[i,:]>0).nonzero()
                        negidx = np.asarray(spectra[i,:]<0).nonzero()
                        spectra[i,negidx] = spectra[i,posidx].min()
                        #print(i, min(spectra[i,:]), max(spectra[i,:]))

                # --- Fit the power spectrum model across all channels ---                    
                fg.fit(freqs, spectra, freq_range)

                # --- Save the group results (group means all the channels) ---
                # Note: this saves the offset, exponent, and goodness of fit (R2, RMSE)
                # for each channel.
                df = fg.to_df(peak_org=max_n_peaks)

                df['channel_name'] = ch_names  
                
                # Save the results to a TSV file
                df.to_csv(savefile, sep='\t', index_label='channel')
                
                t1 = time.time()   
                deltat = t1 - t0
                msg = f'Subject {id} in phase {phase}: aperiodic parameters computed in {deltat:.2f} seconds.'
                print(msg)
                logger.info(msg)
            
            '''
            with open(txtfile, "w") as f:
                f.write("")
            '''

if __name__ == "__main__":
    main()