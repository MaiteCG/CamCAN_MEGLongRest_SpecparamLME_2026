"""
Linear Mixed-Effects (LME) Cardiac Modeling with Permutation-Based Max-T.

This script executes longitudinal mixed-effects regression analyses on the standard 
spectral and aperiodic parameters extracted straight from the ECG channel. To maintain 
absolute statistical equivalence with the resting-state models, it subjects the 
cardiac variables to an identical permutation-driven maximum T-statistic correction.

Purpose:
- To test if participant baseline age (Age0), the longitudinal time step (deltaAge), 
  or their interaction yields significant trajectories within the heart's independent 
  signal profiles.
- Serving as a physiological negative control: verifying that any observed aging 
  patterns in the primary brain analysis cannot be explained away as systemic 
  autonomic shifts or physical cardiac artifacts captured by the sensors.

Processing Steps:
1. Imports the master group table of standard ECG metrics ('group_stats_ecg.tsv').
2. Configures the exact frequency sub-bands chosen for the human resting-state 
   models to guarantee alignment.
3. Fits parallel LME regression equations for every band-specific cardiac parameter.
4. Breaks down the actual coupling between age and ECG metrics over 10,000 
   shuffling iterations to construct an empirical maximum-T null distribution.
5. Evaluates corrected p-values to check for spurious cardiovascular trajectories.

Author: Maité Crespo García
Affiliation: MRC Cognition and Brain Sciences Unit, Cambridge, UK
Date: 19-May-2026 (created, modified from rest/aperiodic_long_lme_maxT.py)
"""

#
#module load conda
#module load rstudio
#conda activate maite_pymer4
#rstudio --no-sandbox (or alias rstudio "rstudio --no-sandbox")

# Imports
import argparse
import logging
logger = logging.getLogger(__name__)
import numpy as np
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from pymer4.models import Lmer
import warnings # pymer4 needs to be updated to python3.1, so suppress warning to do with dataframe
warnings.filterwarnings("ignore", category=FutureWarning, message=".*DataFrame.applymap.*")
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
task = 'rest' # 'emptyroom'

overwrite = False

dointerpolation = True #False # whether to do interpolation of 23.4 Hz noise
sinterp = '' if dointerpolation else '_nointerp'

# Processed data to use for extraction of aperiodic parameters

channelselection = 'ECG'
proc = 'sss' + channelselection #'sss' #'raw'
likemeg = True
slikemeg = 'likemeg' if likemeg else ''

lfreq = 0.1 #Hz
hfreq = 145.0 #Hz
fsample = 300.0 #Hz
frange = f"{round(lfreq, 1)}-{int(hfreq)}Hz"

# Use non trans data fro ECG
trans = False # Whether to use head transformation or not
zmm = 44 # destination z coordinate head position in mm

fitting_param = 'finley' # 'schmidt' # 'oursv2' #'oursv1' # 

# ---- Define the version of bands division to use ----
bandsdiv = '2betas' # '2betas' or '3betas'

num_rand = int(1e4) # number of random permutations

# --- Directories and files ---
if trans:
    deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz'

# directory where the psd files were stored
taskref = 'rest'
phaseref = 'p5'
armref = 1
bids_project_folder = f'BIDS_long_{phaseref}_{taskref}_arm{armref}'
deriv_root = os.path.join(dirs.mysandboxdatadir, bids_project_folder,
                          'derivatives', deriv_folder)

loaddir = os.path.join(deriv_root, 'stats')
savedir = os.path.join(loaddir, f'lme_maxT_{fitting_param}_{bandsdiv}{sinterp}_{channelselection}{slikemeg}_{num_rand}rand')
if not os.path.exists(savedir): os.makedirs(savedir)

# ---- Logging ----
logdir = os.path.join(deriv_root, 'logfiles')
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Set up logging
logfile = os.path.join(logdir, f'aperiodic_ecg_long_{proc}{slikemeg}_{fitting_param}_{bandsdiv}{sinterp}_lme_maxT.log')
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)

# --- Other variables ---
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
                'gamma_peak_freq', 'gamma_band_power',]

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
                'gamma_peak_freq', 'gamma_band_power',]

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

effects_of_interest = ['Age0', 'deltaAge', 'Age0:deltaAge']

max_nperm = 100 # maximum number of permutations to run (for testing purposes)

# --- Functions ---
# Main code
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--irand', type=int, default=None, dest='irand', action='store')    
    args = parser.parse_args()
    irand = int(args.irand) if args.irand is not None else None

    # Datafile with all variables means across channels (direct input for LME)
    datafile_allmeans = f'aperiodic_stier_{proc}{slikemeg}_{fitting_param}_{bandsdiv}{sinterp}_allvars_means.tsv'
    datafile_allmeans = os.path.join(savedir, datafile_allmeans)

    permfileroot = f'aperiodic_stier_{proc}{slikemeg}_{fitting_param}_{bandsdiv}{sinterp}_lme_statperm'

    # Check if a permutation was computed already, if irand is provided
    if irand is not None:
        savefile = os.path.join(savedir, f'{permfileroot}_irand{irand}.npy')

        if os.path.exists(savefile) and not overwrite:
            msg = f'File {savefile} already exists! Use --overwrite to overwrite it or run without --irand to compute all permutations at once.'
            print(msg)
            logging.info(msg)
            return
        
    # ---- Create datafile with all variables means across channels, if it does not exist ----
    if not os.path.exists(datafile_allmeans): # or overwrite:
            print(f'File {datafile_allmeans} does not exist. It will be created.')
            count = 0
                        
            for var in varsoi:
                # Datafile for each variable of interest,
                # containing all the subjects, phases, age, and channels
                datafile = os.path.join(
                    loaddir, f'aperiodic_stier_{proc}{slikemeg}_{fitting_param}_{var}_{bandsdiv}{sinterp}.tsv'
                )         
                
                # --- Create the datafile ---
                # with values of the variable of interest, per participant, phase, alongside the
                # age at phase 2, age lag between p5 and p2, in the long format. To be used as an
                # input for statistical analyses (LME)
                if not os.path.exists(datafile):
                    raise ValueError('File does not exist. Stopping here.')
                
                df = pd.read_csv(datafile, sep='\t').drop(columns=['row', 'task'])

                # Find channels data columns
                channels = [channelselection]
                # Create a new index with subject and phase combined
                df['subject_phase'] = df[['subject', 'phase']].agg('_'.join, axis=1)
                df = df.drop(columns=['subject', 'phase'])
                df = df.set_index('subject_phase')                

                # Compute the mean across non-nan channels and save it as a new column 
                # with name var_megtype
                df[f'{var}_{channelselection}'] = df[channels].mean(axis=1, skipna=True)
                df_tmp = df.drop(columns=channels)
                count += 1

                if count == 1:
                    df_all = df_tmp.copy()
                elif count > 1:
                    df_all = df_all.join(df_tmp[f'{var}_{channelselection}'])

                del df_tmp
                del df

                # After processing all variables and megtypes, save the final dataframe
                if count == len(varsoi):
                    # Save the final dataframe                    
                    df_all.to_csv(datafile_allmeans, sep='\t')
                    print(f'Final concatenated dataframe saved to {datafile_allmeans}')
                    del df_all
    # ------------------------------------------------------------------------
    # In any case, load the data
    df, goodvars = get_gooddata(datafile_allmeans)

    if irand is not None:
        t0 = time.time()
        msg = f'Computing only permutation {irand} of {num_rand}...'
        print(msg)
        logging.info(msg)

        # ---- Create the RESAMPLING MATRIX ----
        resampfile = os.path.join(savedir, 'resampmat.npy')
        if not os.path.isfile(resampfile) or overwrite:
            # Compute the matrix with the random sampling of the data and 
            # save it, if it does not exist. This matrix will be an input to 
            # the permute cluster metric function.
            resampmat = create_resampling_matrix(df.shape[0], num_rand)
            # rows: number of samples (subjects*phases), columns: number of random permutations
            np.save(resampfile, resampmat)

        else:
            resampmat = np.load(resampfile)

        # ---- PERMUTED STATISTICS ----
        # Compute the permuted statistics (LME) for each variable of interest (column)
        tperm, pperm = permuted_lme_age(
            {'df': df, 'vars': goodvars, 'effects_of_interest': effects_of_interest}, 
            num_rand=num_rand, irand=irand, resampmat=resampmat
        )
        statperm = {'tperm': tperm, 'pperm': pperm, 
                    'vars': goodvars, 'effects_of_interest': effects_of_interest}
        np.save(savefile, statperm)
        t1 = time.time()
        deltat = t1 - t0
        msg = f'Permutation {irand} computed in {deltat:.2f} seconds.'
        print(msg)
        logger.info(msg)

    else: # If no irand is provided, compute the original statistics, load all the permutations and then compute the max-T corrected p-values
        # ---- Compute ORIGINAL STATISTICS ----
        statorifile = os.path.join(savedir, f'aperiodic_stier_{proc}{slikemeg}_{fitting_param}{sinterp}_lme_statori.npy')
        if not os.path.exists(statorifile) or overwrite or True:
            # Just in case, load the data again
            df, goodvars = get_gooddata(datafile_allmeans)

            # Compute the original statistics (LME) for each variable of interest (column)
            tori, pori, cori, results = lme_age(df, goodvars, effects_of_interest, return_coefficients=True)
            statori = {'tori': tori, 'pori': pori, 'cori': cori, 'vars': goodvars, 'effects_of_interest': effects_of_interest, 'results': results}
            np.save(statorifile, statori)
            for var, result in zip(goodvars, results):
                print(f'\nVariable: {var}')
                result.to_csv(os.path.join(savedir, f'{var}_lme_results.tsv'), sep='\t')
                
            raise ValueError('Original statistics computed and saved. Please run again without --irand to compute the corrected p-values.')
            
        
        # ---- LOAD THE PERMUTED STATISTICS ----
        # Check which permutations have been computed already
        computed_rand = []
        missing_rand = []
        tperm_list = []
        for irand in range(num_rand):
            loadfile = os.path.join(savedir, f'{permfileroot}_irand{irand}.npy')
            if os.path.exists(loadfile):
                try:
                    statperm = np.load(loadfile, allow_pickle=True).item() 
                    tperm_list.append(statperm['tperm'])
                    computed_rand.append(irand)
                except:
                    #os.remove(loadfile)
                    print(irand)
                    missing_rand.append(irand)
            else: 
                missing_rand.append(irand)

        if len(missing_rand) > 0:
            print(f'Missing {len(missing_rand)} permutations:')
            missing_rand_str = ' '.join(map(str, missing_rand))
            print(missing_rand_str)
            print('Please run again with --irand for each missing permutation to complete the analysis.')
            return 

        else:   
            # ---- Obtain MAX-T STATISTICS ----        
            correctedpfile = os.path.join(savedir, f'aperiodic_stier_{proc}_{fitting_param}{sinterp}_maxT_corrected_pvals.npy')

            if not os.path.exists(correctedpfile) or overwrite:
                # For each effect, do:
                # For each permutation, save the maximum t-value across all variables
                max_abst_values_all = np.zeros((num_rand, len(effects_of_interest)))
                for eoi in range(len(effects_of_interest)):
                    # Save the maximum t-values for this effect of interest
                    max_abst_values_all[:,eoi] = np.array([np.abs(tperm[:,eoi]).max(axis=0) for tperm in tperm_list])
                    
                # Load the original t-values
                statori = np.load(statorifile, allow_pickle=True).item()
                tori = statori['tori']
                abst_tori = np.abs(tori)

                # Compute corrected p-values based on the max-t distribution
                pvals_corrected = np.ones(abst_tori.shape, dtype=float)
                for i in range(abst_tori.shape[0]): # for each variable
                    for j in range(abst_tori.shape[1]): # for each effect of interest
                        # Count how many times the maximum t-value in the permutations is greater than
                        # the original t-value
                        pvals_corrected[i,j] = np.sum(max_abst_values_all[:,j] >= abst_tori[i,j]) / num_rand
                np.save(correctedpfile, pvals_corrected)
            
            else:
                pvals_corrected = np.load(correctedpfile)
                print(pvals_corrected.shape)
                statori = np.load(statorifile, allow_pickle=True).item()
                tori = statori['tori']

            pori = np.load(statorifile, allow_pickle=True).item()['pori']        
            vars = np.load(statorifile, allow_pickle=True).item()['vars'] 

            # Print the results
            print('\nCorrected p-values (max-t method):')
            for j, eoi in enumerate(effects_of_interest):
                print(f'\nEffect of interest: {eoi}')
                for i, var in enumerate(vars):
                    if pvals_corrected[i,j] < 0.05:
                        print(f'Variable: {var}, original T-statistic: {tori[i,j]:.4f}, \noriginal p-value: {statori["pori"][i,j]:.4f}, corrected p-value: {pvals_corrected[i,j]:.8f}')
                
# --- Functions ---
def find_goodvars(df, vars):
    goodvars = []
    for var in vars:
        p2sub = df.loc[df.phase == 'p2', ['subject', var]].dropna().subject
        p5sub = df.loc[df.phase == 'p5', ['subject', var]].dropna().subject
        nsubs = len(list(set(p2sub).intersection(p5sub)))
        if nsubs > 70:
            goodvars.append(var)
    return goodvars

def get_gooddata(datafile):
    df = pd.read_csv(datafile, sep='\t', index_col=0)
    if df.index.name == 'subject_phase':
        df.reset_index(inplace=True)
    if not 'phase' in df.columns:
        df['subject'] = df['subject_phase'].str.split('_').str[0]
        df['phase'] = df['subject_phase'].str.split('_').str[1]
        df.drop(columns=['subject_phase'], inplace=True)
    vars = [col for col in df.columns if col not in ['subject', 'phase', 'Age0', 'deltaAge']]
    
    # Find the variables with enough data samples for the LME test
    goodvars = find_goodvars(df, vars)
    # Select only the good variables
    df = df[['subject', 'phase', 'Age0', 'deltaAge'] + goodvars]

    return df, goodvars

def create_resampling_matrix(total_sampled, num_rand):
    # Create a resampling matrix with total_sampled rows and 
    # num_rand columns. (Each column will contain a random 
    # permutation of the original row indexes.)
    resampmat = np.empty((total_sampled, num_rand), dtype= int)
    # Create an array with the indexes of the rows  
    arr = np.arange(total_sampled)
    #print(arr)
    np.random.seed(0) # Set the seed for reproducibility
    for b in range(num_rand):
        ridx = arr.copy()
        # Shuffle the indexes
        np.random.shuffle(ridx) # this changes the array in place
        resampmat[:,b] = ridx 

    return resampmat

def lme_age(df, vars, effects_of_interest, return_coefficients=False):
    # df = dataframe with all the variables
    # vars = list of columns in the dataframe corresponding to all the variables
    #           that will be tested
    # effects of interest = list, e.g., ['Age0','deltaAge' and interaction 'Age0:deltaAge']
    #print('\nRunning LME with pymer4...')
    df['Age0'] = df['Age0'] - df['Age0'].mean(skipna=True) # Center the Age0 variable
    df['deltaAge'] = df['deltaAge'] - df['deltaAge'].mean(skipna=True) # Center the deltaAge variable

    tout = np.zeros((len(vars), len(effects_of_interest)))
    pout = np.zeros((len(vars), len(effects_of_interest)))
    all_results = []
    if return_coefficients:
        cout = np.zeros((len(vars), len(effects_of_interest)))
    for i, f in enumerate(vars):                
        model = Lmer(f"{f} ~ Age0 + deltaAge + Age0*deltaAge + (1|subject)", data=df)
        model.fit(summary=False, verbose=False)

        # model.coefs gives a dataframe with effects (Age0, deltaAge, etc.) in index 
        # and statistical parameters (T-stat, P-val, etc) in columns
        # Use the p values from here, because they have more decimal units than the model
        # dataframe and the printed summary
        results = model.coefs
        all_results.append(results)
        for j, eoi in enumerate(effects_of_interest):            
            tout[i,j] = results.loc[eoi, 'T-stat']
            pout[i,j] = results.loc[eoi, 'P-val']
            if return_coefficients:
                cout[i,j] = results.loc[eoi, 'Estimate']

        #print(f'{(i+1)} / {len(vars)}')   

    if return_coefficients:
        return tout, pout, cout, all_results
    else:
        return tout, pout

def permuted_lme_age(datadict, num_rand=int(1e4), irand=None, resampmat=None):
    # Return vector of cluster size thresholds for samp_size x num_tests matrix y
    # given height threshold(s) (thrs)

    dfori = datadict['df'].copy().reset_index()    

    if resampmat is None:
        # If no resampling matrix is provided, create one
        total_sampled = dfori.shape[0]
        resampmat = create_resampling_matrix(total_sampled, num_rand)

    if irand is not None and isinstance(irand, int):
        #print(f'Running permutation {irand} of {num_rand}...')
        #raise ValueError('Check the irand value. It should be an integer between 0 and num_rand-1.')
        # If a specific permutation index is provided, use it
        nrand_iter = np.array([irand]).astype(int)  # Convert to int
    else:
        nrand_iter = range(num_rand)

    tperm_list = []
    pperm_list = []

    for b in nrand_iter:
        permidx = resampmat[:,b].astype(int)  # Get the random indexes from the resampling matrix
        
        # Make a deep copy of the original dataframe
        dfperm = dfori.copy()        
        
        # Shuffle the data in the freqs columns only
        dfperm[datadict['vars']]= dfori[datadict['vars']].iloc[permidx,:].to_numpy()
        
        # Compute the LME for all the effects of interest at once, and get the
        # t-values and p-values for each variable (rows) and effect (columns)
        tperm, pperm = lme_age(dfperm, datadict['vars'], datadict['effects_of_interest']) 

        if len(nrand_iter) == 1:
            # If only one permutation is requested, return the t and p values
            return tperm, pperm
        else:
            # If multiple permutations are requested, append the t and p values to the lists
            tperm_list.append(tperm)
            pperm_list.append(pperm)   

        if (b+1) % 10 == 0:
            print(f'Permutation {b+1} / {num_rand} done.')
            np.save('tperm_list_temp.npy', tperm_list)
            np.save('pperm_list_temp.npy', pperm_list)     
    
        return tperm_list, pperm_list

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

def add_text(section, title, text, tags):
    global report
    report.add_html(
        html=f'<div>{text}<div>',
        title=title,
        tags=tags, # tuple 
        section=section, 
        replace=False
    )

def add_figure(figurefile, section, title, caption=None, tags=()):
    global report

    report.add_image(
        figurefile, 
        title, 
        caption=caption, 
        tags=tags, # tuple 
        section=section, replace=False
    )

     
if __name__ == "__main__":
    main()