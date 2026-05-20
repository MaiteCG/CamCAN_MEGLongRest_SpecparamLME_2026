# CamCAN_MEGLongRest_SpecParamLME_2026

This repository contains analysis code for the following manuscript: 

Title: **Longitudinal versus Cross-sectional effects of Age on MEG Power Spectra Parameters: Implications for Normative Models and Brain Ageing**

Authors: Maite Crespo-Garcia, Dace Apšvalka, Ina Demetriou, Adam Attaheri, Tina Emery, Máté Aller, Cam-CAN & Richard Henson.



## Code Description

### Preprocessing

The configuration files for the **MNE-BIDS automatic pipeline** are located in `code/prep/mnebids`:

- `config_rest.py`:  Configuration for automatic preprocessing of **rest MEG** data.

- `config_emptyroom.py`: Configuration for automatic preprocessing of **empty room** MEG data associated to resting-state MEG data.
  
  

The preprocessing of the **ECG channel** is performed within the script that computes the spectral power (see below). 



The **custom** Python files for other **preprocessing steps** are located in `code/prep`:

- `custom_bad_epochs.py`: Detects bad epochs in **rest MEG** data using muscle artifact z-scoring. Saves the indexes of the good epochs to be used later in the pipeline.

- `custom_icastep.py`: Executes a customized ICA fitting procedure on filtered **rest MEG** data, after removing bad epochs detected with the `custom_bad_epocs.py` script.

- `custom_icaartifacts_schmidt.py`: This script identifies ICA components corresponding to EOG (ocular) and ECG (cardiac) artifacts by applying the correlation-based thresholding method described in Schmidt et al. (2025 eLife).

- `custom_remove_icaartifacts.py`: Applies the ICA rejection identified in `custom_icaartifacts_schmidt.py` to the filtered **rest MEG** data. It serves as the final cleaning step before downstream analysis. Alternatively, it can also create another **control rest MEG data only with ocular artifacts (but not cardiac) removed**.

- `custom_remove_allbutecg04.py`: Performs a selective ICA-based cleaning to create a **control dataset only with cardiac ICs (most of brain activity removed)**. It removes all independent components EXCEPT those identified as cardiac (ECG-related) artifacts in `custom_icaartifacts_schmidt.py`.
  
  

### Spectral power computation

- `psd_rest.py`: Computes the **power spectrum** for **rest MEG** data that has been cleaned of ocular and cardiac ICA artifacts (via `custom_remove_icaartifacts.py`), and other **control MEG datasets** with different ICA removal treatments.

- `psd_emptyroom.py`: Computes the **power spectrum** for **empty room** recordings associated with specific rest MEG recordings.

- `psd_ecg.py`: Preprocesses and computes the **power spectrum** of the **ECG channel**.
  
  

### Spectral power parametrisation

#### For basic models:

- `specparam_rest.py`: Extracts spectral parameters by fitting the SpecParam (FOOOF) model to the Power Spectral Density (PSD) of MEG data to separate the 1/f **aperiodic** signal (offset and exponent) from **periodic peaks** (centre frequency, power, band width). It can be applied to the PSDs of **rest MEG** (main analysis) or **control MEG** datasets processed with different ICA treatments. It includes the **interpolation of the ~24 Hz noise**, which was **switched off for a control analysis**. It also includes the option of setting the **`aperiodic_mode`**, which was set in ***fixed*** mode for the main analyses, but was set to ***knee*** for a **control analysis**. The parameters are saved in a subject- and phase-specific .tsv file.

- `specparam_emptyroom.py`: Extracts spectral parameters by fitting the SpecParam (FOOOF) model to **empty room** PSDs to characterize the background noise of the MEG system. It serves as a critical control to ensure that subject-level aperiodic results are not driven by sensor noise. The parameters are saved in a subject- and phase-specific .tsv file.

- `specparam_ecg.py`: Extracts spectral parameters by fitting the SpecParam (FOOOF) model to the Power Spectral Density (PSD) of the **ECG channel**. By using the same frequency range and model constraints as the MEG analysis, it allows for a direct comparison between heart and brain age effects on the aperiodic exponent. It includes the option of setting the **`aperiodic_mode`**, which was set in ***knee*** mode for the main ECG channel analysis, but was set to ***fixed*** for a **control analysis**. The parameters are saved in a subject- and phase-specific .tsv file.
  
  

The .tsv files resulting from the spectral parametrisation step of MEG rest and empty room datasets contain all the spectral parameters for each channel, and the peak parameters have not been yet classified to a cannonical frequency band (e.g. theta). **The scripts below** read these subject- and phase-specific files, **classify the peak parameters into frequency bands,** and **generate other .tsv files that include data from all subjects and phases**, for each parameter (e.g., exponent, low_alpha_power) and meg type (gradiometers or magnetometers).

- `specparam_rest_grouptsv.py`: Aggregates the individual SpecParam outputs (.tsv files) from `specparam_rest.py` for all subjects and phases into a single master dataset. It performs the critical step of mapping individual model peaks to canonical frequency bands (e.g., alpha, beta). This file can be applied to outputs from **rest MEG** and other **control MEG datasets**.

- `specparam_emptyroom_grouptsv.py`: Aggregates individual SpecParam outputs from **empty room** recordings (`specparam_emptyroom.py`) for all subjects and sessions. It classifies any detected environmental peaks into canonical frequency bands (Theta, Alpha, Beta, Gamma) for comparison with resting-state neural data. Set sinterp += '' on line 48.

- `specparam_ecg_grouptsv.py`: Aggregates individual **ECG channel** spectral parameters (generated by `specparam_ecg.py`) into a single master dataset. It classifies any periodic peaks found in the ECG signal into standard frequency bands. This applies also to the **control spectral parametrisation on ECG channel with fixed mode**.
  
  

#### For the **LME models with one covariate**:

Parameters from **ECG channel** or **empty room** data, we extracted the exponent from the subject- and phase-specific files (above), then used the peaks parameters from the  rest MEG data to **obtain ECG and empty room power values**. The **ECG power values were relative** to the total power spectrum, whereas the empty room values were computed as a **log10 transform of the ratio between total empty room power and rest aperiodic power** (see Methods section of the paper). The scripts below compute these new parameters to be used as covariates and create the aggregated .tsv files for each parameter, as above.

- `specparam_components.py`: Used to obtain the **aperiodic PSD of rest MEG data** (created with `psd_rest.py`), for later normalization of empty room peak power values (see Methods section in the paper). It can also be used to obtain the periodic or total PSD (epoch average) of rest MEG data.

- `specparam_emptyroom_logdiffaper.py`: Quantifies **empty room power** specifically within the frequency ranges where neural peaks were detected in the associated rest MEG data. It calculates a **log-difference between empty room power and the subject's aperiodic power**. Then, it saves the new parameters in a subject- and phase-specific .tsv files. 
  **NOTE**: To aggregate the aperiodic parameters from all participants and phases into a single tsv file, use `specparam_emptyroom_grouptsv.py ` above with sinterp += '_totalminusaperiodicrest' on line 48.

- `specparam_ecg_logrelpow.py`: Quantifies **relative ECG power** specifically within the frequency ranges where neural peaks were detected in the associated rest MEG data. It calculates a log-difference between ECG power and the total ECG power between 2-40 Hz. Then, it saves the new parameters in a subject- and phase-specific .tsv files.

- `specparam_ecg_relpow_grouptsv.py`: Aggregates individual **ECG channel** spectral parameters (generated by `specparam_ecg_logrelpow.py`) into a single master dataset. It classifies any periodic peaks found in the ECG signal into standard frequency bands. 
  
  

### Statistical analyses (LME models)

- `basic_lme_model_maxT.py`: Executes longitudinal linear mixed-effects (LME) models on the aggregated spectral and aperiodic parameters from **rest MEG** data. It tests for the effects of baseline age (Age0), longitudinal age changes (deltaAge), and their interaction. To address the multiple comparisons problem across parameters, the script implements a non-parametric permutation test using the maximum T-statistic distribution to control the family-wise error rate. Used **for main analysis and all control analyses based on rest MEG data** using the basic model.

- `basic_lme_model_maxT_emptyroom.py`: Executes longitudinal linear mixed-effects (LME) models on the aggregated spectral and aperiodic parameters from **empty room** data. It tests for the effects of baseline age (Age0), longitudinal age changes (deltaAge), and their interaction. To address the multiple comparisons problem across parameters, the script implements a non-parametric permutation test using the maximum T-statistic distribution to control the family-wise error rate.

- `basic_lme_model_maxT_ecg.py`: Executes longitudinal linear mixed-effects (LME) models on the aggregated spectral and aperiodic parameters from **ECG channel** data. It tests for the effects of baseline age (Age0), longitudinal age changes (deltaAge), and their interaction. To address the multiple comparisons problem across parameters, the script implements a non-parametric permutation test using the maximum T-statistic distribution to control the family-wise error rate. Used for **control analyses with knee or fixed mode**.

- `onecov_lme_model_maxT.py`: Executes longitudinal linear mixed-effects (LME) models on the aggregated spectral and aperiodic parameters from **rest MEG** data. It tests for the effects of baseline age (Age0), longitudinal age changes (deltaAge), and their interaction, but also adds **one covariate** derived from **empty room (power to aperiodic ratio)** or **ECG channel (relative power)**. To address the multiple comparisons problem across parameters, the script implements a non-parametric permutation test using the maximum T-statistic distribution to control the family-wise error rate. Used **for main analysis and all control analyses based on rest MEG data** using the basic model.

- `sixcov_lme_model_maxT.py`: Executes longitudinal linear mixed-effects (LME) models on the aggregated spectral and aperiodic parameters from **rest MEG** data. It tests for the effects of baseline age (Age0), longitudinal age changes (deltaAge), and their interaction, but also adds **6 covariates (head position and motion, sex, and TIV)**. To address the multiple comparisons problem across parameters, the script implements a non-parametric permutation test using the maximum T-statistic distribution to control the family-wise error rate.
  
  

### Figures and tables



## Dependencies:

- Python

- MNE-Python, MNE-BIDS pipeline

- pandas, numpy, matplotlib

- specparam 2.0

- pymer4



## Data Availability

The raw MEG data from Phase 2 are already available on the CamCAN repository ([Cam-CAN Data Repository](https://cam-can.mrc-cbu.cam.ac.uk/dataset)).

Data from Phase 5 will be added soon (or available on request until then).
