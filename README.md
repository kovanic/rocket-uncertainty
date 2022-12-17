This is the repo for Skoltech BMML course project:  **"Uncertainty estimation via ROCKET".**

> The objective of the project is to test various approaches of uncertainty estimation and numerically compare them on time-series from UCR archive.
> The base model is ROCKET, followed by linear classifier. I either estimate uncertainty on ensemble of ROCKETs or via dropout.

To replicate results from the report you need to:

1)  Clone this repo
2)  Install `requirements.txt`
3)  Download [data](http://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_ts.zip) and unzip it in `data` folder of the project
4)  Run notebboks

The structure of the project is as follows:


.
├── README
├── data
│   ├── results_ucr_additional.csv      #results of original paper
│   └── results_ucr_bakeoff.csv
├── notebooks
│   ├── Dropout uncertainties.ipynb     #calculate uncertainties via dropout
│   ├── Ensemble uncertainties.ipynb    #calculate uncertainties on ensemble of ROCKETs
│   └── Reproducing results.ipynb       #ensemble of ROCKETs and ridge classifier
├── requirements.txt
├── results
│   ├── all_results.csv                 #all uncertainties metrics (dropout, ensemble), accs and original results
│   ├── example_rc.pdf
│   ├── reproduced_results_ridge.csv
│   ├── uncertainty_results_dropout.csv
│   └── uncertainty_results_ensemble.csv
└── src
    ├── rocket.py                       #ROCKET in torch
    ├── ucr_utils.py                    #load and preprocess UCR archive
    └── uncertainty_estimation.py       #metrics of uncertainties

Author: Valerii Kornilov
Email: Valerii.Kornilov@skoltech.ru
