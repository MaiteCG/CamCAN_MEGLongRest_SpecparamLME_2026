'''
Script to generate the source data for Supplementary Figure 2, which shows the counts of peak detections across all the datasets, per frequency bin (0.5 Hz resolution), for a given megtype (gradiometers).


Description: Extracts the number of peaks detected per frequency bin across all the datasets. The input data are the tsv files created with the script specparam_rest.py, which contain the aperiodic/periodic parameters for each subject, phase, and sensor type separately. The peak parameters in these input files are NOT yet classified into a classic frequency bands (theta, alpha, beta, etc.). The output of this script will be used to create a figure like in Donoghue et al., 2020, Figure 7B, showing the counts of peak detections across all the datasets, per frequency bin (0.5 Hz resolution).

Author: Maité Crespo García
Date: 15-04-2026 (created)
'''

# Imports
import json
import numpy as np
import os
import pandas as pd
import sys
import logging
logger = logging.getLogger(__name__)

print(os.getcwd())
datadir = os.path.abspath(os.path.join(os.path.dirname( os.getcwd() ), '.', 'data'))
print(f'Data directory: {datadir}')

specparamdir = os.path.abspath(os.path.join(os.path.dirname( os.getcwd() ), '.', 'code/specparam'))
print(f'Specparam directory: {specparamdir}')

maindir = '' # define the main directory where the BIDS datasets are stored
bids_project_folder = '' # define the name of the BIDS project folder

# --- Main global variables ---
pipver = 'stier'
task = 'rest'  # 'emptyroom'
phases = ['p2', 'p5']
arms = [1, 2]
megtype = 'grad' # only done for gradiometers

overwrite = False

# Processed data to use for extraction of aperiodic parameters
icselection = 'ecg04eog08' #  'allbutecg04' # 'eog08' # 
proc = 'filt' + icselection #'sss' #'clean'

# whether to use epochs or raw data
cropdata = 532 

method = 'specparam' #'fooof' # irasa

lfreq = 0.1 #Hz
hfreq = 145.0 #Hz
fsample = 300.0 #Hz
frange = f"{round(lfreq, 1)}-{int(hfreq)}Hz"

trans = True # Whether to use head transformation or not
zmm = 44 # destination z coordinate head position in mm

# --- Package used for aperiodic fitting ---
package = 'specparam' #'fooof' # irasa

fitting_param = 'finley'

jsonfile = os.path.join(specparamdir, f'aperiodic_fitting_params_{fitting_param}.json')
freq_range = None
if os.path.exists(jsonfile):
    with open(jsonfile) as json_file:
        json_dict = json.load(json_file)
    
    epoch_duration = json_dict['epoch_duration']
    powmethod = json_dict['powmethod']
    fres = json_dict['fres']
    freq_range = json_dict['freq_range']
else:
    raise FileNotFoundError(f'Fitting parameters file {jsonfile} not found.')

psddesc = f'dur{cropdata}sepo{epoch_duration}s{powmethod}'
sfres = '' if fres == 0.1 else f'fres{fres}Hz'

# --- Directories and files ---
if trans:
    load_deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    load_deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz'

# directory where the psd files were stored
taskref = 'rest'
phaseref = 'p5'
armref = 1
#bids_project_folder = f'BIDS_long_{phaseref}_{taskref}_arm{armref}'
deriv_root = os.path.join(maindir, bids_project_folder,
                          'derivatives', load_deriv_folder)

logdir = os.path.join(deriv_root, 'logfiles')
os.makedirs(logdir, exist_ok=True)
# Set up logging
logfile = os.path.join(logdir, f'aperiodic_long_sp_{proc}_{fitting_param}_createtsv.log')
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)

# ---- File with subjects and arms ----
subjlistfile = os.path.join(datadir, f'meglong_{task}_subjects.tsv')

# --- Functions ---
# Main code
def main():
    # Read file with subjects and arms
    subjectsdf = pd.read_csv(subjlistfile, sep='\t').set_index('subject')
    subjects = subjectsdf.index.tolist()   

    countsfile = os.path.join(datadir, f'supp_figure02_source_data.tsv')

    if os.path.isfile(countsfile) and not overwrite:
        msg = f'File {countsfile} already exists. Skipping the creation of the counts dataframe for megtype {megtype}.'
        print(msg)
        logger.info(msg)
        return
    
    else:
        # Create a dataframe to store the counts of detected peaks per frequency bin across all the datasets, per megtype.
        countsdf = pd.DataFrame(columns=['frequency_bin', 'count'])
        fbins = np.arange(freq_range[0], freq_range[1], 0.5) # frequency bins with 0.5 Hz resolution, from 2 to 40 Hz
        countsdf['frequency_bin'] = fbins
        countsdf['count'] = 0

        
        # Code creating the datafile here
        # Create a the dataframe with the variable of interest, and save it
        # in the long format, for statistics
        for phase in phases:                   
            # loop over subjects within this age group
            for id in subjects: 
                armx = subjectsdf.loc[id,'arm']

                # ---- Define the load directory ----
                bids_project_folder = f'BIDS_long_{phase}_{task}_arm{armx}'
                load_derivdir = os.path.join(maindir, bids_project_folder,
                        'derivatives', load_deriv_folder)
                save_megdir = os.path.join(load_derivdir, 'sub-'+id, 'meg')
                
                # --- Define the load file for aperiodic parameters ---
                loadfilename = f'sub-{id}_task-{task}_proc-{proc}_desc-{psddesc}{megtype}{fitting_param}_{package}.tsv'
                loadfile = os.path.join(save_megdir, loadfilename)

                # Check if the files already exist
                if not os.path.isfile(loadfile): 
                    msg = (
                        f'Error file found for subject {id} phase {phase} megtype {megtype}.'
                        ' Skipping this subject.'
                    )
                    print(msg)
                    continue
                
                else:
                    print(f'Processing subject {id} phase {phase} megtype {megtype}')
                    # Load the tsv file with the aperiodic parameters, for each subject
                    df = pd.read_csv(loadfile, sep='\t').set_index('channel')

                    all_peaks_freqs = []
                    for peak in range(0, 6): # loop over the 6 possible peaks 
                        col_name = f'cf_{peak}'
                        if col_name in df.columns:
                            all_peaks_freqs.extend(df[col_name].values)
                    
                    for bin in fbins:
                        count = np.sum((np.array(all_peaks_freqs) >= bin) & (np.array(all_peaks_freqs) < bin + 0.5))
                        countsdf.loc[countsdf['frequency_bin'] == bin, 'count'] += count

        # Save the data to a tsv file
        countsdf.to_csv(countsfile, sep='\t', index=True, index_label='row')
        print(f'Counts dataframe for megtype {megtype} saved to {countsfile}.')

                
if __name__ == "__main__":
    main()