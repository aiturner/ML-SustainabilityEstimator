# ML Sustainability Estimator

## Overview

ML Sustainability Estimator is a project that evaluates machine learning models based on both predictive performance and environmental impact.

The project benchmarks multiple text classification models on the Rotten Tomatoes sentiment analysis dataset and compares:

* Classification accuracy
* Runtime
* Energy consumption
* Carbon emissions

The goal is to identify the model that provides the best balance between performance and sustainability.

---

## Dataset

The project uses the Amazon Polarity dataset from Hugging Face.

Task:

Binary sentiment classification
Positive vs Negative product reviews

Dataset size:

Full training dataset: 3.6 million reviews
Full test dataset: 400,000 reviews

To reduce execution time while maintaining realistic scale, the experiments use:

20,000 training samples
5,000 test samples

---

## Models Evaluated

### Logistic Regression

A linear classification model that predicts class probabilities using a weighted combination of input features.

### Multinomial Naive Bayes

A probabilistic classifier commonly used for text classification. It assumes words occur independently and predicts the most likely class using Bayes' Theorem.

### Linear SVM

A Support Vector Machine classifier that finds the decision boundary with the maximum margin between classes.

---

## Methodology

### Performance Measurement

Model performance is evaluated using classification accuracy on the test dataset.

### Energy Measurement

Energy consumption is measured using Intel's `turbostat` utility.

Example:

```bash
sudo turbostat -q --Joules --show Pkg_J python <script>
```

The reported `Pkg_J` value represents CPU package energy consumption in Joules.

Three measurements are collected for each model and averaged to reduce variability.

### Carbon Emissions

Energy is converted from Joules to kilowatt-hours:

```text
kWh = Joules / 3,600,000
```

Carbon emissions are estimated using:

```text
Emissions (kgCO2e) =
Energy (kWh) × Carbon Intensity (gCO2e/kWh) / 1000
```

---

## Results

| Model                   | Accuracy | Energy (J) | Runtime (s) |
| ----------------------- | -------: | ---------: | ----------: |
| Logistic Regression     |   0.7702 |      49.55 |        3.43 |
| Multinomial Naive Bayes |   0.7758 |      40.95 |        3.43 |
| Linear SVM              |   0.7514 |      43.03 |        3.43 |

### Observations

* Multinomial Naive Bayes achieved the highest accuracy.
* Multinomial Naive Bayes also consumed the least energy.
* Logistic Regression consumed the most energy of the evaluated models.
* Linear SVM achieved the lowest accuracy on this dataset.

Based on the collected measurements, Multinomial Naive Bayes provides the best balance between predictive performance and sustainability.

---

---

## Author

Arthur Turner

University of Oxford

