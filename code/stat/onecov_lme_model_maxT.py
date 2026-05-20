"""
Linear Mixed-Effects (LME) Modeling with One Control Covariate and Max-T Correction.

This script executes longitudinal statistical analyses on subject resting-state brain 
metrics while explicitly adjusting for a single nuisance covariate (either 
environmental noise parameters from empty room recordings or physiological features 
from the ECG channel). Multiple comparisons across channels and frequency bands 
are corrected non-parametrically via an empirical maximum T-statistic distribution.

Purpose:
- To test if the longitudinal effects of baseline age (Age0), time changes (deltaAge), 
  or their interaction remain statistically significant when environmental or cardiac 
  confounds are partialled out of the regression matrix.
- To provide mathematical insulation showing that developmental trajectories are 
  not artificial products of physical scanner changes or cardiac activity.

Processing Steps:
1. Loads the primary subject resting-state data ('group_stats_rest.tsv').
2. Merges it with the target covariate file ('group_stats_noise.tsv' or 'group_stats_ecg.tsv').
3. Formulates the adjusted LME equation including the covariate term.
4. Fits the regression model across all targeted channel-by-band brain variables.
5. Performs 10,000 permutation shuffles to map out a corrected maximum absolute 
   T-statistic null distribution.
6. Quantifies family-wise error rate corrected p-values for the primary age predictors.

Author: Maité Crespo García
Affiliation: MRC Cognition and Brain Sciences Unit, Cambridge, UK
Date: 19-May-2026 (last modified)
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

overwrite = False

# Processed data to use for extraction of aperiodic parameters

icselection = 'ecg04eog08' # 'allbutecg04' #'eog08' # 
proc = 'filt' + icselection #'sss' #'clean'

lfreq = 0.1 #Hz
hfreq = 145.0 #Hz
fsample = 300.0 #Hz
frange = f"{round(lfreq, 1)}-{int(hfreq)}Hz"

trans = True # Whether to use head transformation or not
zmm = 44 # destination z coordinate head position in mm

fitting_param = 'finley' # 'schmidt' # 'oursv2' #'oursv1' # 

# ---- Define the version of bands division to use ----
bandsdiv = '2betas' # '2betas' or '3betas'

num_rand = int(1e4) # number of random permutations

covtype = 'emptyroom' #'ecg' # 'ecg' 
covproc = 'filt' if covtype == 'emptyroom' else ('sssECG' if covtype == 'ecg' else '') #'sss' #'clean' #
suffix = 'totalminusaperiodicrest' if covtype == 'emptyroom' else ('totalrelpow2-40Hz' if covtype == 'ecg' else '')
covname = f'_{covtype}{suffix}' if covtype == 'emptyroom' else (f'_{suffix}' if covtype == 'ecg' else '')
covtypesuff = f'_{covtype}{suffix}'
covproctask = (f'{covproc}_{covtype}' if covtype == 'emptyroom' else (covproc if covtype == 'ecg' else ''))

# --- Directories and files ---
if trans:
    deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz'

covtrans = False # set to False for empty room and ECG covariates

if covtrans:
    cov_deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    cov_deriv_folder = f'aperiodic_filt{frange}_fs{int(fsample)}Hz'

# directory where the psd files were stored
taskref = 'rest'
phaseref = 'p5'
armref = 1
deriv_root = os.path.join(maindir, bids_project_folder,
                          'derivatives', deriv_folder)

cov_deriv_root = os.path.join(maindir, bids_project_folder,
                          'derivatives', cov_deriv_folder)

loaddir = os.path.join(deriv_root, 'stats')
covloaddir = os.path.join(cov_deriv_root, 'stats')
savedir = os.path.join(loaddir, f'lme_1cov{covtypesuff}_maxT_{fitting_param}_{bandsdiv}_{icselection}_{num_rand}rand')
if not os.path.exists(savedir): os.makedirs(savedir)

# ---- Logging ----
logdir = os.path.join(deriv_root, 'logfiles')
if not os.path.exists(logdir):
    os.makedirs(logdir)

# Set up logging
logfile = os.path.join(logdir, f'aperiodic_long_{proc}_{fitting_param}_{bandsdiv}_lme_1cov{covtypesuff}_maxT.log')
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
    
    varsoi_cov = [cov for cov in varsoi if cov == 'exponent' or 'band_power' in cov] # Include only band power and exponent as covariates

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
    datafile_allmeans = f'aperiodic_stier_{proc}_{fitting_param}_{bandsdiv}_allvars_means.tsv'
    datafile_allmeans = os.path.join(savedir, datafile_allmeans)

    if not os.path.exists(datafile_allmeans):
        datafile_allmeans_source = f'aperiodic_stier_{proc}_{fitting_param}_{bandsdiv}_allvars_means.tsv'
        statsdir = os.path.join(deriv_root, 'stats')
        sourcedir = os.path.join(statsdir, f'lme_maxT_{fitting_param}_{bandsdiv}_{num_rand}rand')
        datafile_allmeans_source = os.path.join(sourcedir, datafile_allmeans_source)
        if not os.path.exists(datafile_allmeans_source):
            msg = f'File {datafile_allmeans_source} should exist already. Please, check or run aperiodic_long_lme_maxT.py without irand first to create it.'
            raise ValueError(msg)
        else:
            # Copy the datafile from the previous analysis without covariates
            print(f'Copying {datafile_allmeans_source} to {datafile_allmeans}...')
            os.system(f'cp {datafile_allmeans_source} {datafile_allmeans}')

    # Datafile with all variables means across channels (direct input for LME)
    datafile_allmeans_cov = f'aperiodic_stier_{covproc}_{fitting_param}_{bandsdiv}_allvars_means{covname}.tsv'
    datafile_allmeans_cov = os.path.join(savedir, datafile_allmeans_cov)

    permfileroot = f'aperiodic_stier_{proc}_{fitting_param}_{bandsdiv}_lme_1cov_statperm'

    # Check if a permutation was computed already, if irand is provided
    if irand is not None:
        savefile = os.path.join(savedir, f'{permfileroot}_irand{irand}.npy')

        if os.path.exists(savefile) and not overwrite:
            msg = f'File {savefile} already exists! Use --overwrite to overwrite it or run without --irand to compute all permutations at once.'
            print(msg)
            logging.info(msg)
            return
        
    # ---- Create datafile with all variables means across channels, if it does not exist ----
    if not os.path.exists(datafile_allmeans_cov): # or overwrite:
            print(f'File {datafile_allmeans_cov} does not exist. It will be created.')
            count = 0
            for megtype in megtypes:   
                   
                for var in varsoi_cov:
                    # Datafile for each megtype and variable of interest,
                    # containing all the subjects, phases, age, and channels
                    if var == 'exponent':
                        if covtype == 'emptyroom':
                            datafile = os.path.join(                            
                                covloaddir, f'aperiodic_stier_{covproctask}_{fitting_param}_{megtype}{var}_{bandsdiv}.tsv'
                            )
                        elif covtype == 'ecg':
                            datafile = os.path.join(
                                covloaddir, f'aperiodic_stier_{covproctask}_{fitting_param}_{var}_{bandsdiv}.tsv'
                            )
                    elif 'band_power' in var:
                        datafile = os.path.join(
                            covloaddir, f'aperiodic_stier_{covproctask}_{fitting_param}_{megtype}{var}_{bandsdiv}_{suffix}.tsv'
                        )
                    else: # peak frequency
                        continue # peak frequency is not used as covariate    
                        
                    # --- Create the datafile ---
                    # with values of the variable of interest, per participant, phase, alongside the
                    # age at phase 2, age lag between p5 and p2, in the long format. To be used as an
                    # input for statistical analyses (LME)
                    if not os.path.exists(datafile):
                        raise ValueError(f'File {datafile} does not exist. Stopping here.')
                    
                    df = pd.read_csv(datafile, sep='\t').drop(columns=['row', 'task'])

                    if covtype == 'ecg' and var == 'exponent':
                        channels = ['ECG']
                    else:
                        # Find channels data columns
                        channels = [c for c in df.columns if c.startswith('MEG')]
                    # Create a new index with subject and phase combined
                    df['subject_phase'] = df[['subject', 'phase']].agg('_'.join, axis=1)
                    df = df.drop(columns=['subject', 'phase'])
                    df = df.set_index('subject_phase')                

                    # Compute the mean across non-nan channels and save it as a new column 
                    # with name var_megtype
                    df[f'{var}_{megtype}'] = df[channels].mean(axis=1, skipna=True)
                    df_tmp = df.drop(columns=channels)
                    count += 1

                    if count == 1:
                        df_all = df_tmp.copy()
                    elif count > 1:
                        df_all = df_all.join(df_tmp[f'{var}_{megtype}'])

                    del df_tmp
                    del df

                    # After processing all variables and megtypes, save the final dataframe
                    if count == len(varsoi_cov) * len(megtypes):
                        # Save the final dataframe                    
                        df_all.to_csv(datafile_allmeans_cov, sep='\t')
                        print(f'Final concatenated dataframe saved to {datafile_allmeans_cov}')
                        del df_all
    # ------------------------------------------------------------------------
    # In any case, load the data
    df, goodvars = get_gooddata(datafile_allmeans)

    # ---- Read covariates ----
    df_cov, goodvars_cov = get_gooddata(datafile_allmeans_cov)

    covariates_list = goodvars_cov

    # Merge covariates into the main dataframe with all variables
    df = df.merge(df_cov, how='left', on = ['subject', 'phase'], suffixes=('', '_cov'))
    df = df.dropna(axis=0, how='any', subset=['Age0_cov', 'deltaAge_cov'])
    df = df.drop(columns=['Age0_cov', 'deltaAge_cov'])
    
    # ---- PERMUTATION TESTING ----
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
        tperm, pperm = permuted_lme_1cov_age(
            {'df': df, 'vars': goodvars, 'effects_of_interest': effects_of_interest, 'covariates_list': covariates_list}, 
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
        statorifile = os.path.join(savedir, f'aperiodic_stier_{proc}_{fitting_param}_lme_1cov_statori.npy')
        if not os.path.exists(statorifile) or overwrite:
            # Just in case, load the data again
            df, goodvars = get_gooddata(datafile_allmeans)

            # ---- Read covariates ----
            df_cov, goodvars_cov = get_gooddata(datafile_allmeans_cov)

            covariates_list = goodvars_cov

            # Merge covariates into the main dataframe with all variables
            df = df.merge(df_cov, how='left', on = ['subject', 'phase'], suffixes=('', '_cov'))
            df = df.dropna(axis=0, how='any', subset=['Age0_cov', 'deltaAge_cov'])
            df = df.drop(columns=['Age0_cov', 'deltaAge_cov'])

            # Compute the original statistics (LME) for each variable of interest (column)
            tori, pori, cori = lme_1cov_age(df, goodvars, effects_of_interest, covariates_list, return_coefficients=True)
            statori = {'tori': tori, 'pori': pori, 'cori': cori, 'vars': goodvars, 'effects_of_interest': effects_of_interest}
            np.save(statorifile, statori)
            print(f'Original statistics saved to {statorifile}.')
            
        
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
            for megtype in megtypes:
                print(f'\nMEG type: {megtype}')
                # ---- Obtain MAX-T STATISTICS ----        
                correctedpfile = os.path.join(savedir, f'aperiodic_stier_{proc}_{fitting_param}_maxT{megtype}_corrected_pvals.npy')

                # If the corrected p-values file does not exist, or if overwrite is True, compute the corrected p-values and save them. Otherwise, just load the corrected p-values
                if not os.path.exists(correctedpfile) or overwrite or True:
                    # Load the original t-values
                    statori = np.load(statorifile, allow_pickle=True).item()
                    tori = statori['tori']
                    vars = statori['vars']
                    print(vars)
                    megtype_mask = [megtype in var for var in vars]
                    print(megtype_mask)

                    abst_tori = np.abs(tori)
                    abst_tori = abst_tori[megtype_mask,:]

                    # For each effect, do:
                    # - For each permutation, save the maximum absolute t-value across all variables
                    max_abst_values_all = np.zeros((num_rand, len(effects_of_interest)))

                    for eoi in range(len(effects_of_interest)):
                        # Save the maximum t-values for this effect of interest
                        max_abst_values_all[:,eoi] = np.array([np.abs(tperm[megtype_mask,eoi]).max(axis=0) for tperm in tperm_list])                        

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

                pori = np.load(statorifile, allow_pickle=True).item()['pori'][megtype_mask,:]        
                vars = np.load(statorifile, allow_pickle=True).item()['vars']
                vars = [var for var in vars if megtype in var]

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

def lme_1cov_age(df, vars, effects_of_interest, covariates_list, return_coefficients=False):
    # df = dataframe with all the variables
    # vars = list of columns in the dataframe corresponding to all the variables
    #           that will be tested
    # effects of interest = list, e.g., ['Age0','deltaAge' and interaction 'Age0:deltaAge']
    #print('\nRunning LME with pymer4...')
    df['Age0'] = df['Age0'] - df['Age0'].mean(skipna=True) # Center the Age0 variable
    df['deltaAge'] = df['deltaAge'] - df['deltaAge'].mean(skipna=True) # Center the deltaAge variable
    for cov in covariates_list:
        df[cov] = (df[cov] - df[cov].mean(skipna=True))/df[cov].std(ddof=0, skipna=True) # Center the covariate variable and z-score it

    tout = np.zeros((len(vars), len(effects_of_interest)))
    pout = np.zeros((len(vars), len(effects_of_interest)))
    if return_coefficients:
        cout = np.zeros((len(vars), len(effects_of_interest)))
    for i, f in enumerate(vars):
        if ('exponent' in f) or ('band_power' in f):              
            model = Lmer(f"{f} ~ Age0 + deltaAge + Age0*deltaAge + {f}_cov + (1|subject)", data=df)
        else:
            # peak frequency
            model = Lmer(f"{f} ~ Age0 + deltaAge + Age0*deltaAge + (1|subject)", data=df)
        model.fit(summary=False, verbose=False)

        # model.coefs gives a dataframe with effects (Age0, deltaAge, etc.) in index 
        # and statistical parameters (T-stat, P-val, etc) in columns
        # Use the p values from here, because they have more decimal units than the model
        # dataframe and the printed summary
        results = model.coefs
        for j, eoi in enumerate(effects_of_interest):            
            tout[i,j] = results.loc[eoi, 'T-stat']
            pout[i,j] = results.loc[eoi, 'P-val']
            if return_coefficients:
                cout[i,j] = results.loc[eoi, 'Estimate']

        #print(f'{(i+1)} / {len(vars)}')   

    if return_coefficients:
        return tout, pout, cout
    else:
        return tout, pout

def permuted_lme_1cov_age(datadict, num_rand=int(1e4), irand=None, resampmat=None):
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
        tperm, pperm = lme_1cov_age(dfperm, datadict['vars'], datadict['effects_of_interest'], datadict['covariates_list']) 

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