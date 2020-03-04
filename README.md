# KALI

## Investigating the effect of the calibration data length on the performance of hydrological models

In the present repository, you can find the materials for the paper **Ayzel G., Heistermann M. The effect of calibration data length on the performance of conceptual versus data-driven hydrological models"** submitted to *the Journal of Hydroinformatics*.

About the idea and considered models



### Code

The code is written in [*Python*](https://docs.python.org/) programming language (v3.6) using open-source software libraries, such as [*numpy*](https://numpy.org/), [*pandas*](https://pandas.pydata.org/), [*scipy*](https://www.scipy.org/), [*numba*](http://numba.pydata.org/), [*tensorflow*](https://www.tensorflow.org/), and [*keras*](https://keras.io/). The analysis of obtained results was done also using [*jupyter notebooks*](https://jupyter.org/) and [*matplotlib*](https://matplotlib.org/) plotting library.

You can install all the required dependencies using [*conda*](https://docs.conda.io/projects/conda/en/latest/index.html) -- an open-source package management system. First, [install conda itself](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), then use the provided `environment.yml` file to create the isolated environment:

```bash
conda env create -f environment.yml
```

There are three files in the `code` directory:
1. [`experiment.py`](https://github.com/hydrogo/KALI/blob/master/code/experiment.py)

\- describes the workflow for the main calibration/validation experiment. 

2. [`metrics.py`](https://github.com/hydrogo/KALI/blob/master/code/metrics.py)

\- calculates and aggregates the evaluation metrics based on obtained results of streamflow simulation [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3696832.svg)](https://doi.org/10.5281/zenodo.3696832).

3. [`analysis.ipynb`](https://github.com/hydrogo/KALI/blob/master/code/analysis.ipynb)

\- represents the analysis of the effect of calibration data length on the performance of hydrological models.


### Models

There are two files in the `models` directory:
1. `gr4h.py`

\- holds the code for the GR4H hydrological model. 

2. `anns.py`

\- holds the code for generating data-driven hydrological models based on different architectures of artificial neural networks: MLP, RNN, LSTM, and GRU.

These files are used as modules in `experiment.py`.


### Results

There are two files that aggregate evaluation metrics for the calibration and validation periods: `summary_calibration.npy` and `summary_validation.npy`, respectively. The `figures` subfolder consists of figures that were generated using the `analysis.ipynb` jupyter notebook.