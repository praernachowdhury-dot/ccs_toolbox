# Module Index

This index summarises the modules present in the local `ccstools` source tree. It is meant as a navigation aid, not as a substitute for function-level docstrings.

## Core Analysis

- `ccstools.eegfeatures`: PSD, FOOOF, IRASA, ACW, Catch22, nonlinear complexity, Lempel-Ziv variants, and connectivity helpers.
- `ccstools.eegfeatures_fast`: faster multi-feature table generation for larger EEG datasets.
- `ccstools.eegfeatures_fast_v01`: earlier fast feature extraction implementation kept for compatibility/reference.
- `ccstools.yasafeatures`: YASA-oriented spectral, aperiodic, nonlinear, and PAC feature functions.
- `ccstools.noneegfeatures`: ECG, HRV, respiration, and quadrant HRV features.
- `ccstools.psv_sdg_brain_heart_model`: PSV/SDG and brain-heart modelling classes.

## Signal Processing And Statistics

- `ccstools.sigproc`: ERP bootstrapping, PCA, sine-wave/wavelet generation, amplitude/phase extraction, delta-wave and ERP peak detection.
- `ccstools.emd`: empirical mode decomposition helper.
- `ccstools.corrstats`: dependent and independent correlation comparison utilities.
- `ccstools.limo_tfce_replicate`: Python replication of LIMO-style cluster and TFCE helpers.
- `ccstools.lz_functions`: Lempel-Ziv complexity helpers.
- `ccstools.pci`: perturbational complexity index helpers.
- `ccstools.similarity_matrix`: clustering similarity matrix and ensemble clustering helpers.
- `ccstools.waves`: noise/wave generation utilities.

## File I/O And Parsers

- `ccstools.fileio`: BESS waveform text import.
- `ccstools.curryreader`: Curry data reader.
- `ccstools.embla`: EMBLA file and sleep-stage helpers.
- `ccstools.read_eprimetxt`: E-Prime text parser and dataframe utilities.
- `ccstools.save_edf`: MNE Raw to EDF writer.
- `ccstools.mne2EDF`: MNE to EDF export helper.

## EEG Cleaning And MNE Workflows

- `ccstools.ccs_eeg.pipeline`: CCS MNE preprocessing pipeline, bad-channel detection, and ICA helpers.
- `ccstools.ccs_eeg.utils`: montage loading and EEG waveform plotting helpers.
- `ccstools.ccs_eeg.gedai.gedai_algo`: GEDAI and SENSAI cleaning algorithms.
- `ccstools.ccs_eeg.gedai.utils`: lead-field, covariance, MODWT/MRA, subspace-angle, and weighting helpers.
- `ccstools.mne_asr`: Artifact Subspace Reconstruction implementation.
- `ccstools.mne_asr_utils`: ASR support routines, filters, medians, and covariance helpers.

## Plotting, Real-Time, And Devices

- `ccstools.plot`: waveform, hypnogram, and head-map plotting utilities.
- `ccstools.recording`: Muse and Xampl10 recording/device helper classes.
- `ccstools.ml_realtime`: real-time EEG and model-streaming helpers.
- `ccstools.stimulation`: stimulation-device helper class.

## Public Root Imports

The package root currently exposes:

```python
from ccstools import compute_psd, compute_fooof, compute_irasa, compute_nonlinear, compute_acw
from ccstools import bootstrapERP, pca, smooth, detecterppeak
```

Specialised functions should usually be imported from their module directly, for example `from ccstools.ccs_eeg.pipeline import run_ccs_pipeline`.
