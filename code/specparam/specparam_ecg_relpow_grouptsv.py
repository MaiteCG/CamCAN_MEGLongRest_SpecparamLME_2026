"""
Aggregate Relative ECG Band Power Across Subjects and Phases.

This script aggregates individual sensor-level relative ECG band power outputs 
from 'specparam_ecg_logrelpow.py' into a single group-level master table. 
It maps the relative cardiac power values into canonical frequency bands 
(Theta, Alpha, Beta, Gamma) based on the center frequencies of the 
corresponding neural peaks.

Processing Steps:
1. Iterates through all subjects, phases, and sensor types (grad/mag).
2. Loads individual relative ECG band power files.
3. Classifies the neural peak boundaries into classical frequency bands.
4. Identifies the peak with the maximum relative cardiac power if multiple 
   peaks overlap within a single band definition.
5. Concatenates all subject-specific tables into a final group master table.

Author: Maité Crespo García
Affiliation: MRC Cognition and Brain Sciences Unit, Cambridge, UK
Date: 21-Jan-2026 (created, modified from aperiodic_long_emptyroom_sp_createtsv.py)
"""

# Imports
import json
import numpy as np
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

# --- Main global variables ---
pipver = 'stier'
task = 'rest' 
phases = ['p2', 'p5']
arms = [1, 2]
megtypes = ['grad', 'mag']

overwrite = False

dointerpolation = True # whether to do interpolation of 23.4 Hz noise
sinterp = '' if dointerpolation else '_nointerp'

# Indicate here the components and transforms that were used to create the aperiodic/periodic parameters files
sinterp += '_totalrelpow2-40Hz' # to indicate that the total power spectrum is used minus the aperiodic component

# Processed data to use for extraction of aperiodic parameters
chanselection = 'ECG'
proc = 'sss' + chanselection
likemeg = True # whether to process ECG data in the same way as MEG data (filters)
slikemeg = 'likemeg' if likemeg else ''

# whether to use epochs or raw data
cropdata = 532 

# Define the frequency range to fit
##### Frequency Range used for the Aperiodic Fit ###
freq_range = [3, 35] #0.5-140 Hz in the paper, but we use 1-30 Hz to avoid 50 Hz noise
frangestr = f'{freq_range[0]}-{freq_range[1]}Hz'
method = 'specparam' #'fooof' # irasa

lfreq = 0.1 #Hz
hfreq = 145.0 #Hz
fsample = 300.0 #Hz
frange = f"{round(lfreq, 1)}-{int(hfreq)}Hz"

trans = False # Whether to use head transformation or not
zmm = 44 # destination z coordinate head position in mm

# --- Package used for aperiodic fitting ---
package = 'specparam' #'fooof' # irasa

fitting_param = 'finley' #'oursv2' # 'oursv1' #

jsonfile = os.path.join(dirs.homecamcancodedir, 'sens', f'aperiodic_fitting_params_{fitting_param}.json')
if os.path.exists(jsonfile):
    with open(jsonfile) as json_file:
        json_dict = json.load(json_file)
    
    epoch_duration = json_dict['epoch_duration']
    powmethod = json_dict['powmethod']
    fres = json_dict['fres']
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
bids_project_folder = f'BIDS_long_{phaseref}_{taskref}_arm{armref}'
deriv_root = os.path.join(dirs.mysandboxdatadir, bids_project_folder,
                          'derivatives', load_deriv_folder)

logdir = os.path.join(deriv_root, 'logfiles')
if not os.path.exists(logdir):
    os.makedirs(logdir)
# Set up logging
logfile = os.path.join(logdir, f'aperiodic_long_sp_{proc}{slikemeg}_{fitting_param}{sinterp}_createtsv.log')
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)

statsdir = os.path.join(deriv_root, 'stats')
if not os.path.exists(statsdir):
    os.makedirs(statsdir)

# ---- File with subjects and arms ----
subjlistfile = os.path.join(dirs.mysandboxdatadir, f'meglong_{taskref}_subjects.tsv')

# This file contains all the subjects in the longitudinal study, with the age at each phase
agefilename = os.path.join(dirs.mysandboxdatadir,f'meglong_{taskref}_age.tsv')

bandsdiv = '2betas' # whether to divide beta into 2 or 3 sub-bands

if bandsdiv == '3betas':
    # --- Variables of interest ---
    varsoi = ['exponent', 
                'theta_peak_freq', 'theta_band_power', 
                'alpha_peak_freq', 'alpha_band_power',
                'low_alpha_peak_freq', 'low_alpha_band_power',
                'high_alpha_peak_freq', 'high_alpha_band_power',
                'beta_peak_freq', 'beta_band_power', 
                'low_beta_peak_freq', 'low_beta_band_power',
                'middle_beta_peak_freq', 'middle_beta_band_power',
                'high_beta_peak_freq', 'high_beta_band_power',
                'gamma_peak_freq', 'gamma_band_power',
                'r_squared',]

    bandsdict = {
                'theta':(4,8),
                'alpha':(8,13),
                'beta':(15,30),
                'low_alpha':(8,10),
                'high_alpha':(10,13),
                'low_beta':(13,15),
                'middle_beta':(15,20),
                'high_beta':(20,30),
                'gamma':(30,48),
                }
    
elif bandsdiv == '2betas':
    varsoi = ['exponent', 
                'theta_peak_freq', 'theta_band_power', 
                'alpha_peak_freq', 'alpha_band_power',
                'beta_peak_freq', 'beta_band_power',
                'low_alpha_peak_freq', 'low_alpha_band_power',
                'high_alpha_peak_freq', 'high_alpha_band_power',
                'low_beta_peak_freq', 'low_beta_band_power',
                'high_beta_peak_freq', 'high_beta_band_power',
                'gamma_peak_freq', 'gamma_band_power',
                'r_squared',]
    
    if sinterp == '_totalrelpow2-40Hz':
        varsoi = [var for var in varsoi if 'power' in var] # keep only power variables

    bandsdict = {
                'theta':(4,8),
                'alpha':(8,12),
                'beta':(12,30),
                'low_alpha':(8,10),
                'high_alpha':(10,12),
                'low_beta':(12,20),
                'high_beta':(20,30),
                'gamma':(30,48),
                }

# --- Functions ---
# Main code
def main():
    # Read file with subjects and arms
    subjectsdf = pd.read_csv(subjlistfile, sep='\t').set_index('subject')
    subjects = subjectsdf.index.tolist()

    age = pd.read_csv(agefilename, sep='\t').set_index('ccid')
    age.rename(columns={'p2_meg_age':'p2_age', 'p5_meg_age':'p5_age'}, inplace=True)
    
    for megtype in megtypes:
        columns = []
        for var in varsoi:
            bandname = get_bandname(var)
            if bandname:
                fbandrange = bandsdict[bandname] #0.5-140 Hz in the paper, but we use 1-30 Hz to avoid 50 Hz noise
                fband = f'{fbandrange[0]}-{fbandrange[1]} Hz'
            
            # Create a datafile for each megtype and variable of interest,
            # containing all the subjects, phases, age, and channels
            createdf = True
            datafile = os.path.join(
                statsdir, f'aperiodic_stier_{proc}{slikemeg}_{fitting_param}_{megtype}{var}_{bandsdiv}{sinterp}.tsv'
                )
            
            # --- Create the datafile ---
            # with values of the variable of interest, per participant, phase, alongside the
            # age at phase 2, age lag between p5 and p2, in the long format. To be used as an
            # input for statistical analyses (LME)
            if not os.path.isfile(datafile) or overwrite:
                # Code creating the datafile here
                # Create a the dataframe with the variable of interest, and save it
                # in the long format, for statistics
                for phase in phases:                   
                    # loop over subjects within this age group
                    for id in subjects: 
                        armx = subjectsdf.loc[id,'arm']

                        # ---- Define the load directory ----
                        bids_project_folder = f'BIDS_long_{phase}_{taskref}_arm{armx}'
                        load_derivdir = os.path.join(dirs.mysandboxdatadir, bids_project_folder,
                                'derivatives', load_deriv_folder)
                        save_megdir = os.path.join(load_derivdir, 'sub-'+id, 'meg')
                        if not os.path.exists(save_megdir):
                            os.makedirs(save_megdir)

                        # --- Define the load file for aperiodic parameters ---
                        loadfilename = f'sub-{id}_task-{task}_proc-{proc}{slikemeg}_desc-{psddesc}{megtype}{fitting_param}_{package}{sinterp}.tsv'
                        loadfile = os.path.join(save_megdir, loadfilename)

                        # Check if the files already exist
                        if not os.path.isfile(loadfile): 
                            if os.path.isfile(os.path.join(save_megdir, 'error.txt')):
                                msg = (
                                    f'Error file found for subject {id} phase {phase} megtype {megtype}.'
                                    ' Skipping this subject.'
                                )
                                print(msg)
                                logger.warning(msg)
                                continue
                            else:
                                msg = (
                                    f'Please, run aperiodic sp script first. The '\
                                    'the file below does not exist:'
                                    f'\\n{loadfile}'
                                )
                                print(msg)
                                logger.warning(msg)
                                continue
                                #raise FileNotFoundError(msg)
                        
                        else:
                            print(f'Processing variable {var} for subject {id} phase {phase} megtype {megtype}')
                            # Load the tsv file with the aperiodic parameters, for each subject
                            df = pd.read_csv(loadfile, sep='\t').set_index('channel')
                            #print(df.columns.tolist())

                            # create the list with column names only once
                            if len(columns) == 0:
                                columns = ['subject', 'task', 'phase', 'Age0', 'deltaAge']
                                colchans = len(columns)
                                columns.extend(df.channel_name.tolist())

                            if createdf:
                                newdf = pd.DataFrame(columns=columns)
                                createdf = False

                            newidx = newdf.index.max() + 1 if not newdf.empty else 0
                                                                                    
                            newdf.loc[newidx, 'subject'] = id
                            newdf.loc[newidx, 'task'] = task
                            newdf.loc[newidx, 'phase'] = phase
                            newdf.loc[newidx, 'Age0'] = age.loc[id, 'p2_age']
                            if phase == 'p2':                        
                                newdf.loc[newidx, 'deltaAge'] = 0
                            elif phase == 'p5':
                                newdf.loc[newidx, 'deltaAge'] = age.loc[id, 'p5_age'] - age.loc[id, 'p2_age']

                            # Check if the variable of interest is in the dataframe    
                            if bandname:
                                print(var)
                                newdf.loc[newidx, df['channel_name'].tolist()] = add_var_row(df, var, bandname, bandsdict[bandname])

                            elif var in df.columns: 
                                # Get the data for this variable
                                newdf.loc[newidx, df['channel_name'].tolist()] = df[var].values

                            else:
                                print(f'{var} is was not extracted from the table')


                # Save the data to a tsv file
                newdf.to_csv(datafile, sep='\t', index=True, index_label='row')

            else:
                msg = f'File {datafile} was already created.'
                print(msg)
                logger.info(msg)

            

def get_bandname(var):
    bandname = []
    for name in ['theta', 'low_alpha', 'high_alpha', 'low_beta', 'middle_beta', 'high_beta']:
        if name in var:
            bandname = name
            return bandname
            
    if not bandname:
        for name in ['alpha', 'beta', 'gamma']:
            if name in var:
                bandname = name
                return bandname
    
    return bandname

def add_var_row(df, var, bandname, frange):
    # add a column with the variable name
    df[var] = ''

    prefix = []     
    if 'peak_freq' in var:
        prefix = 'cf'
    elif 'band_power' in var:
        prefix = 'pw'
    elif 'band_width' in var:
        prefix = 'bw'

    fcols = []
    for col in df.columns:
        if 'cf' in col:
            fcols.append(col)
    #print(fcols)

    pcols = []
    for col in df.columns:
        if 'pw' in col:
            pcols.append(col)
    #print(pcols)

    if bandname:
        for i in df.index: # for each channel
            idx = np.where((df.loc[i, fcols] >= frange[0]) & (df.loc[i, fcols] < frange[1]))[0].tolist()
            #print(idx)
            if len(idx) > 1:
                selcols = [pcols[j] for j in idx]
                #print(selcols)
                #print(df.loc[i, selcols])
                #print(df.loc[i, selcols].max())
                idxmax = np.where(df.loc[i, selcols] == df.loc[i, selcols].max())[0].tolist()
                idx = idx[idxmax[0]]
            elif len(idx) == 1:
                idx = idx[0]
            else:
                idx = None

            if idx is not None:
                df.loc[i, var] = df.loc[i, f'{prefix}_{str(idx)}']
        
    return df[var].values
                
if __name__ == "__main__":
    main()