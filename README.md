# CCS EEG Toolbox (`ccstools`)

`ccstools` is a Python research toolbox for EEG and related physiology workflows at CCS, NIMHANS. It collects utilities for feature extraction, signal processing, file conversion, plotting, EEG cleaning, ASR/GEDAI-style artefact handling, sleep/HRV features, real-time acquisition helpers, and connectivity/statistical analysis.

The package is intentionally organised as a set of importable modules rather than one monolithic API. Existing function names and module paths are kept stable because notebooks, scripts, and analysis pipelines often import these functions directly.

## What Is Included

- EEG feature extraction: PSD, FOOOF-style aperiodic fits, IRASA, ACW, Catch22, entropy, fractal dimension, Lempel-Ziv complexity, and multi-epoch feature tables.
- Faster feature extraction paths in `eegfeatures_fast.py` and `eegfeatures_fast_v01.py` for workflows that benefit from vectorisation or Numba-style acceleration.
- Signal processing helpers for ERP bootstrapping, PCA, sine-wave and wavelet generation, smoothing, instantaneous amplitude/phase extraction, delta-wave detection, and ERP peak detection.
- File I/O and conversion helpers for BESS waveform text, Curry files, EDF writing, MNE-to-EDF export, E-Prime text parsing, and EMBLA sleep files.
- EEG cleaning and preprocessing under `ccstools.ccs_eeg`, including bad-channel detection, ICA-based cleaning helpers, montage utilities, and GEDAI/SENSAI-style cleaning code.
- Plotting utilities for EEG waveforms, hypnograms, and head maps.
- Non-EEG physiology features for ECG, HRV, respiration, PSV/SDG modelling, and sleep-analysis workflows.
- Real-time and device-facing helpers for Muse, Xampl10, stimulation devices, and streaming model workflows.

## Installation

For active development, install the package in editable mode:

```bash
git clone https://github.com/arunsasidharan84/ccs_toolbox.git
cd ccs_toolbox
python -m pip install -e .
```

For a fuller environment with scientific dependencies, use one of the provided environment files:

```bash
python -m pip install -r ccs_requirements.txt
```

or, with conda/mamba:

```bash
conda env create -f ccs_environment.yml
conda activate ccs_toolbox
```

The package metadata requires Python 3.9 or newer.

## Quick Start

```python
import numpy as np
from ccstools.eegfeatures import generate_multieegfeatures

# Shape expected by many feature helpers: epochs x channels x samples
data = np.random.randn(5, 3, 1000)
srate = 500
chanlist = ["Fz", "Cz", "Pz"]

features = generate_multieegfeatures(data, srate, chanlist)
print(features.head())
```

Common direct imports are also exposed from the package root:

```python
from ccstools import compute_psd, compute_fooof, compute_irasa, compute_nonlinear, compute_acw
from ccstools import bootstrapERP, pca, smooth, detecterppeak
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `ccstools/eegfeatures.py` | Main EEG feature extraction functions, including PSD, FOOOF, IRASA, ACW, Catch22, nonlinear features, and connectivity helpers. |
| `ccstools/eegfeatures_fast.py` | Faster multi-feature extraction implementation for larger EEG feature tables. |
| `ccstools/yasafeatures.py` | YASA-oriented PSD, IRASA, FOOOF, nonlinear, and PAC feature helpers. |
| `ccstools/sigproc.py` | General signal-processing utilities: ERP bootstrapping, PCA, wavelets, amplitude/phase, delta waves, and ERP peaks. |
| `ccstools/fileio.py`, `curryreader.py`, `embla.py`, `read_eprimetxt.py`, `save_edf.py`, `mne2EDF.py` | File import/export and parser utilities. |
| `ccstools/ccs_eeg/` | MNE-oriented EEG preprocessing, channel-location utilities, plotting helpers, and GEDAI/SENSAI cleaning code. |
| `ccstools/plot.py` | Waveform, hypnogram, and head-map plotting utilities. |
| `ccstools/noneegfeatures.py` | ECG, HRV, respiratory, and quadrant HRV features. |
| `ccstools/psv_sdg_brain_heart_model.py` | Brain-heart modelling helpers around PSV/SDG-style analyses. |
| `ccstools/mne_asr.py`, `mne_asr_utils.py` | Artifact Subspace Reconstruction implementation and support routines. |
| `ccstools/recording.py`, `ml_realtime.py`, `stimulation.py` | Device, streaming, real-time model, and stimulation helpers. |
| `verify_ccstools.py` | Lightweight import/verification script. |
| `docs/module_index.md` | Human-readable module index generated from the local source tree. |

## Main Workflows

### EEG Feature Extraction

Use `ccstools.eegfeatures` for the most complete API:

```python
from ccstools.eegfeatures import compute_psd, compute_irasa, compute_acw

psdvals, freqs = compute_psd(data, srate)
irasa_features = compute_irasa(data, srate)
acw_features = compute_acw(data, srate)
```

For large feature tables, start with `ccstools.eegfeatures_fast.generate_multieegfeatures` if the local dependency stack supports it.

### Signal Processing

```python
from ccstools.sigproc import generate_sinewave, wavelet_amplitudephase, detecterppeak

sinewave, timepoints = generate_sinewave(frequency=10, amplitude=1, samplingrate=500, duration=2)
```

`generate_moreletwavelet` keeps its historical function name for compatibility, but the comments describe the underlying Morlet wavelet concept.

### File I/O

```python
from ccstools.fileio import importBessWaveform
from ccstools.read_eprimetxt import read_eprimetxt

EEGdata, srate, chanlist, times = importBessWaveform("waveform.txt")
eprime_df = read_eprimetxt("experiment.txt")
```

### CCS EEG Pipeline

```python
from ccstools.ccs_eeg.pipeline import run_ccs_pipeline

clean_raw, report = run_ccs_pipeline(raw_input, output_dir="outputs")
```

The pipeline utilities are MNE-oriented and may require channel locations, ICALABEL-compatible dependencies, and configured output folders depending on the workflow.

## Dependency Notes

Core package dependencies are listed in `pyproject.toml`; environment-level dependencies are also captured in `ccs_requirements.txt` and `ccs_environment.yml`. Some optional workflows require hardware-specific or heavier libraries. If an import fails, first confirm that the relevant optional stack is installed for that workflow.

The toolbox includes bundled resources under `ccstools/ccs_eeg/resources/`, including channel-location and lead-field support files used by plotting and GEDAI-related routines.

## Development Notes

- Keep public function names, module paths, argument order, and return order stable unless all dependent notebooks/scripts are updated together.
- Prefer adding docstrings, examples, and small compatibility wrappers over renaming existing functions.
- Keep device-specific code isolated from pure analysis functions so offline workflows remain importable.
- Run lightweight syntax/import checks before committing documentation or comment updates.
- Review file-level licence and authorship comments before redistributing bundled or adapted code.

## Verification

A quick local check is:

```bash
python -m compileall ccstools
python verify_ccstools.py
```

`compileall` catches syntax errors without requiring the full scientific stack to run every workflow. `verify_ccstools.py` performs a higher-level package check and may require installed dependencies.

## Licence

This project is released under the MIT License. See `LICENSE` for details.
