"""
Detects bad epochs in MEG data using muscle artifact z-scoring. Saves the indexes 
of the good epochs to be used later in the pipeline.

This script implements an automated artifact rejection procedure based on 
Stier et al. (2023), NeuroImage 278. It processes MaxFiltered (SSS) data by:
1. Resampling to 300 Hz and high-pass filtering at 1 Hz.
2. Calculating z-scores for muscle activity in the 110-140 Hz band.
3. Marking artifacts using a z-score threshold (default = 14).
4. Segmenting data into 2-second epochs and dropping those containing artifacts.

Key differences from the original paper:
- Data is segmented into 2 s epochs instead of 10 s.

Outputs:
- .npy files containing detected muscle scores and good epoch indices.
- .tsv file summarizing the number of good epochs per subject/phase.
- .html MNE Report for visual quality control of muscle scores and drop logs.

Author: Maité Crespo García
Affiliation: MRC Cognition and Brain Sciences Unit, Cambridge, UK
Date: 20-Oct-2025
"""

import argparse
import mne
from mne.preprocessing import annotate_muscle_zscore
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import logging
logger = logging.getLogger(__name__)

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

trans = True # whether to apply head position transformation to a fixed Z-coordinate for cross-subject alignment
zmm = 44 # destination z coordinate head position in mm

proc =  'sss' # the preprocessing step to which the bad epochs detection will be applied, e.g. 'sss' for MaxFiltered data (no filters applied yet).
cropdata = 532

epochdur = 2  # 2 or 10 seconds, Duration of epochs for bad epoch detection. 

overwrite = False # Whether to overwrite the output files.

# The name of the derivatives folder below should match the one created when 
# running the automatic preprocessing MNE-BIDS pipeline with the config files.
if trans:
    deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz'

# Directory where figures and stats will be saved (also the log file of this step)
phaseref = 'p5' 
armref = 1
deriv_root = os.path.join(maindir, bids_project_folder,
                        'derivatives', deriv_folder)

logdir = os.path.join(deriv_root,'bad_epochs')
if not os.path.exists(logdir): os.makedirs(logdir)

# Set up log file
logfile = os.path.join(logdir, f'custom_bad_epochs.log')
logging.basicConfig(filename=logfile, encoding='utf-8', level=logging.DEBUG)

# ---- File with subjects and arms ----
subjlistfile = os.path.join(maindir,f'meglong_{task}_subjects.tsv')

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

    nepochsfilename = f'NumberOfGoodEpochs_stier_{epochdur}s.tsv'
    nepochsfile = os.path.join(logdir, nepochsfilename)

    if os.path.exists(nepochsfile):
        dfout = pd.read_csv(nepochsfile, sep='\t').set_index('id')
    else:
        dfout = pd.DataFrame({'id': subjects}).set_index('id')
        for phase in phases:
            dfout[phase] = ''

    reportfilename = f'Report_bad_epochs_stier_{epochdur}s.html'
    reportfile = os.path.join(logdir, reportfilename)
    report = mne.Report(title= f"Detection of bad epochs using method in Stier et al., 2023")

    for id in subjects:
        armx = subjectsdf.loc[id,'arm']

        for phase in phases_to_process:
            
            t0 = time.time()

            bids_project_folder = f'BIDS_long_{phase}_{task}_arm{armx}'
            derivdir = os.path.join(maindir, bids_project_folder,
                    'derivatives', deriv_folder)
            
            megdir = os.path.join(derivdir, 'sub-'+id, 'meg')

            sssfilename = f'sub-{id}_task-{task}_run-01_proc-{proc}_raw.fif'
            sssfile = os.path.join(megdir, sssfilename)
            
            if not os.path.exists(sssfile):
                msg = f'File {sssfile} does not exist, skipping subject {id}.'
                print(msg)
                logger.error(msg)
                continue

            savefilename = f'sub-{id}_task-{task}_proc-{proc}_desc-dur{cropdata}sepo{epochdur}s_badepochs.npy'
            badepochsfile = os.path.join(megdir, savefilename)

            if os.path.exists(badepochsfile) and not overwrite:
                msg = f'Subject {id} already processed, skipping.'
                print(msg)
                logger.info(msg)
                continue
            
            elif os.path.exists(sssfile):

                raw = mne.io.read_raw_fif(sssfile, preload=False)               
                
                tmax = cropdata - 0.004  # 10 seconds minus 4 ms for the last sample
                if raw.duration < tmax:
                    msg = f'Raw data for subject {id} is shorter than expected ({raw.duration:.2f} seconds). Skipping.'
                    print(msg)
                    logger.error(msg)
                    continue

                else:
                    # Crop the data to the desired duration
                    raw.crop(tmin=0, tmax=tmax).load_data()

                    raw.filter(l_freq=1, h_freq=None)

                    # We resampled the data to 300 Hz
                    raw.resample(300, npad="auto")

                    # The threshold is data dependent, check the optimal threshold by plotting
                    # ``scores_muscle``.
                    threshold_muscle = 14  # z-score
                    # Choose one channel type, if there are axial gradiometers and magnetometers,
                    # select magnetometers as they are more sensitive to muscle activity.
                    for megtype in ['grad', 'mag']:
                        annot_muscle, scores_muscle = annotate_muscle_zscore(
                            raw,
                            ch_type=megtype,
                            threshold=threshold_muscle,
                            min_length_good=0.2,
                            filter_freq=[110, 140],
                        )

                        # add the annotations to the raw object
                        raw.set_annotations(raw.annotations + annot_muscle)
                    
                    # Plot muscle scores in the whole data, and detected bad segments (z-threshold)
                    fig, ax = plt.subplots()
                    ax.plot(raw.times, scores_muscle)
                    ax.axhline(y=threshold_muscle, color="r")
                    ax.set(xlabel="time, (s)", ylabel="zscore", title="Muscle activity")
                    report.add_figure(
                        fig=fig,
                        title=f'{id} {phase} - Muscle scores',
                        image_format="PNG",
                        tags = [id, phase, str(armx)],
                        section = 'Muscle Scores'
                    )
                    plt.close(fig)  

                    # Create epochs
                    epochs = mne.make_fixed_length_epochs(raw, duration=epochdur, reject_by_annotation=True)
                    
                    # drop the bad epochs based on the annotations
                    epochs.drop_bad()

                    good_epochs = epochs.selection

                    dfout.loc[id,phase] = len(good_epochs)
                    dfout.to_csv(nepochsfile, sep='\t')

                    # Plot the drop log with the percentage of bad epochs removed
                    fig = epochs.plot_drop_log()
                    report.add_figure(
                        fig=fig,
                        title=f'{id} {phase} - Dropped epochs',
                        image_format="PNG",
                        tags = [id, phase, str(armx)],
                        section = 'Drop Logs'
                        
                    )
                    plt.close(fig)  

                    varnames = ['annot_muscle', 'scores_muscle', 'good_epochs']
                    varvals = [annot_muscle, scores_muscle, good_epochs]
                    vardict = dict(zip(varnames,varvals))

                    np.save(badepochsfile, vardict)
                   
                    report.save(reportfile, overwrite=True)
        
                    t1 = time.time()
                    deltat = t1 - t0
                    msg = f'Subject {id} in phase {phase}: bad epochs detected in {deltat:.2f} seconds.'
                    print(msg)
                    logger.info(msg)

if __name__ == "__main__":
    main()