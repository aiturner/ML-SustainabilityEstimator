# Results Directory

## Dataset

* Rotten Tomatoes sentiment analysis dataset
* Binary classification (positive/negative sentiment)

## Metrics Collected

* Accuracy
* Runtime (seconds)
* Energy Consumption (Joules)
* Energy Consumption (kWh)
* Carbon Emissions (kgCO2e)

## Current Models

1. Logistic Regression
2. Multinomial Naive Bayes

## Energy Measurement Method

Energy consumption is measured using:

```bash
sudo turbostat -q --Joules --show Pkg_J <command>
```

The reported `Pkg_J` value represents CPU package energy consumption in Joules.

## Emissions Calculation

Energy is converted from Joules to kWh:

```text
kWh = Joules / 3,600,000
```

Carbon emissions are estimated using:

```text
Emissions (kgCO2e) =
Energy (kWh) × Carbon Intensity (gCO2e/kWh) / 1000
```

## Main Results File

`model_comparison.csv`

To regenerate results run

rm results/model_comparison.csv
python src/logistic_regression.py
python src/naive_bayes.py
