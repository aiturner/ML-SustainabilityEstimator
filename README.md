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
Positive vs negative product reviews

Dataset size:

Full training dataset: 3.6 million reviews
Full test dataset: 400,000 reviews

To keep experiments manageable while maintaining realistic scale, a sampled subset is generated locally:

250,000 training examples
50,000 test examples

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
Three independent measurements are collected
Energy consumption is averaged
Runtime is averaged

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

### Results
Model	Accuracy	Energy (J)	Runtime (s)
Logistic Regression	0.8948	309.02	16.28
Multinomial Naive Bayes	0.8502	283.94	15.62
Linear SVM	0.8961	362.97	20.11


### Sustainability-Based Model Selection

Rather than selecting a model solely based on accuracy, the project evaluates a combination of:

Accuracy
Energy consumption
Runtime
Carbon emissions

Each metric is normalised relative to the best-performing model for that metric.

A weighted scoring system is then used:

Score =
Accuracy Weight × Accuracy Score
+ Emissions Weight × Emissions Score
+ Runtime Weight × Runtime Score

The model with the highest overall score is selected as the recommended model.

---

## Author

Arthur Turner

University of Oxford

