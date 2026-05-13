# bad_epochs_stier_2s.py
"""
bad_epochs_stier_2s.py

Description: This script implements the procedure for detecting bad epochs as described in Stier et al., 2023. NeuroImage 278, except that the data is segmented into 2 s epochs instead of 10 s.

- To be used after maxfilter (sss).
Method from Stier et al. (2023): "We resampled the data to 300 Hz, initially high-pass filtered at 1 Hz (first order Butterworth), and segmented the data into trials of 10 s [NOTE: here 2 s] length. Trials containing artifacts were removed following an automatic approach for both MEG channel types separately (see https://www.fieldtriptoolbox.org/tutorial/automatic_artifact_rejection/ for further details). In brief, the data was bandpass filtered at 110 to 140 Hz (9th order Butterworth) for optimal detection of muscle artifacts and z-transformed for each channel and timepoint. The z- transformed values were averaged over all channels so that artifacts accumulated and could be detected in a time course representing standardized deviations from the mean of all channels. Finally, all time points that belonged to the artifact were marked using artifact padding, and data trials whose z-values were above a threshold of 14 were excluded."

Author: Maité Crespo García
Affiliation: MRC Cognition and Brain Sciences Unit, Cambridge, UK
Date: 20-Oct-2025
"""
# Imports
import argparse
import mne
from mne.preprocessing import annotate_muscle_zscore
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
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

# Main variables
phases = ['p2', 'p5']
task = 'rest'
pipver = 'stier'
arms = [1, 2]

lfreq = 0.1 #Hz
hfreq = 145.0 #Hz
fsample = 300.0 #Hz
frange = f"{round(lfreq, 1)}-{int(hfreq)}Hz"

trans = True
zmm = 44 # destination z coordinate head position in mm

proc =  'sss' 
cropdata = 532

epochdur = 2  #10 #seconds

overwrite = False

# ---- Directories and files ----
if trans:
    deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz_trans_z{zmm}mm'
else:
    deriv_folder = f'mne-bids-pipeline_{pipver}_filt{frange}_fs{int(fsample)}Hz'

# Directory where figures and stats will be saved (also the log file of this step)
phaseref = 'p5' 
armref = 1
bids_project_folder = f'BIDS_long_{phaseref}_{task}_arm{armref}'
deriv_root = os.path.join(dirs.mysandboxdatadir, bids_project_folder,
                        'derivatives', deriv_folder)

logdir = os.path.join(deriv_root,'bad_epochs')
if not os.path.exists(logdir): os.makedirs(logdir)

# Set up log file
logfile = os.path.join(logdir, f'bad_epochs_stier_2s.log')
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
            derivdir = os.path.join(dirs.mysandboxdatadir, bids_project_folder,
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