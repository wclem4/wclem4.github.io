---
layout: post
title:  "Health Indicators and Stroke Prevalence"

date:   2023-10-01 1:00:00 -0400
categories: blog
---

## Introducing the problem
Stroke, a medical emergency that is characterized by a sudden distribution of
bloodflow to the brain, occurs over 200,000 times per year. Because of its
immediate nature, it is vital to understand who might be at risk of this
condition.

The problem at hand is twofold: \
\- Identify the factors associated with an increased risk of stroke \
\- Develop a deeper understanding of the underlying patterns and
correlations within the data.

Key Questions I seek to answer: \
\- What are the primary risk factors associated with stroke incidence? \
\- How do lifestyle choices impact stroke risk?

Leveraging the power of classification models, we can shed light on what might
cause this medical emergency.

## Introducing the data

The dataset, "Stroke Prediction Dataset", which sourced from Kaggle, contains
12 columns and 5110 rows of health information about an array of patients. The
following outlines the useful features this dataset offers:

\- Gender (Male, Female, or Other) \
\- Age (Number) \
\- Hypertension (0 if patient does not have, 1 if the patient does) \
\- Heart Disease (0 if patient does not have, 1 if the patient does) \
\- Ever Married (yes or no) \
\- Work Type (Children, Govt Job, Never Worked, Private, or Self-employed) \
\- Residence Type (Rural or Urban) \
\- Averagae Glucose Level (Number) \
\- BMI (Number) \
\- Smoking Status (Formerly Smoked, Never Smoked, Smokes, Unknown) \
\- Stroke (0 if patient never has had one, 1 if the patient has)

Below is a small preview of these columns:
```
import pandas as pd
data = pd.read_csv("healthcare-dataset-stroke-data.csv")
data.head()
```
![Data Head](/assets/p2/head.png)

## Pre-processing the data
To scrub the data, the columns with null values need to be identified.
```
null_counts = data.isna().sum()
null_counts[null_counts > 0]
```
![Nulls](/assets/p2/nulls.png)

As seen above, there is only one column with any null values: BMI. There are a
couple ways to handle a row with a null; set it to a value or drop the row. In
this case I will be dropping the row, because guessing a BMI value will skew
the data when looking at BMI as a factor. Because the length of the data adds
up to 5110 rows, dropping only 201 rows should not affect the results.

Below is how we can drop those null rows:

```
data = data.dropna(axis=0)
```
That leaves us with 4909 rows. As a final check, we can use describe() to
ensure there are no outliers in the data.
```
data.describe()
```
![Describe](/assets/p2/describe.png)

All these values look typical to me, so we can continue.

## Data Understanding/Visualization

To truly understand the data, here are some different visualizations for all
the features of the dataset.



![Gender Distribution](/assets/p2/gender_distribution.png)

First, we have the Gender Distribution in a bar chart. It demonstrates that in
this dataset, there are more females with strokes in this dataset than males.

![Age Distribution](/assets/p2/age_distribution.png)

The age distribution really demonstrates how much strokes occur in older ages.

![HyperTension/Heart Disease Distribution](/assets/p2/hypertension_heart_disease_distribution.png)

Hypertension and Heart Disease are definitely prevalent in patients that had a
stroke.

![Describe](/assets/p2/marital_status_distribution.png)

There are much more married patients than non married patients that had a
stroke, but there is also much more married patients in the dataset.

![Describe](/assets/p2/work_type_distribution.png)

The work type seems to correlate with the general data as well, there are no specific outliers here. Majority of the data works in the private sector.

![Describe](/assets/p2/residence_type_distribution.png)

The residential type is pretty dead even as well.

![Describe](/assets/p2/glucose_distribution.png)

The glucose levels of stroke patients seem to be relatively high (expected
should be around 70-100mg/dL), and could be a possible important column that
could correlate with stroke.

![Describe](/assets/p2/bmi_distribution.png)

The BMI for most stroke cases seems to be on the higher end, but is not as
noticeable of a difference from the rest of the data.

![Describe](/assets/p2/smoking_status_distribution.png)

Lastly, the smoking status seems to be fairly even across the board.

These visuals answer our intitial questions on which risk factors associate
with stroke. From these diagrams, we can establish that the age and high
glucose levels seem to be factors that influence a stoke the most.


## Modeling

For the classification model, I chose to use the Random Forest algorithm
becauses of its accuracy and easy computational model. It is a popular ensemble
learning mode that consists of a collection of decision trees. Each tree is
trained on a subset of data, and the final prediction is based on the majority
vote of the individual tree predictions. It handles complex relationships and
provides feature importantance that other models do not provide.

## Evaluation

The model was evaluated using several metrics from a confusion matrix as well
as a classification report.

![Evaluation](/assets/p2/evaluation.png)

Looking at the confusion matrix, we have 862 true positives, 67 false
positives, 39 false negatives, and 14 true negatives. Though predicting a
stroke is difficult, the model could use more tweaking to better make
predictions. \
Also, according to the classification report, the precision is high for
patients without a stroke (0) but quite low for patients with a stroke (1).

## Storytelling

While our exploration revealed some neat findings, it also posed important
challenges and questions that warrant further investigation. The visualizations
told us that age, heart disease/hypertension, and glucose levels all seemed to
affect the likeliness of a stroke. The model has more tweaking to do to fully
classify the dataset, but it still manages to get some correct.

## Impact
Healthcare Improvement: The project has the potential to make a significant
positive impact on healthcare by enhancing our understanding of stroke risk
factors. Identifying these factors can lead to better prevention and early
intervention strategies, ultimately reducing the burden of stroke casualties.

Ethical Considerations: There is a risk of overreliance on data-driven models,
potentially leading to dehumanized healthcare. Clinicians should complement
data insights with their expertise and judgment, treating patients as
individuals rather than solely relying on statistical predictions.

## References
[Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)\
[Stroke Background Information from NHLBI](https://www.nhlbi.nih.gov/health/stroke)

## Code
You can view all the code for this project [here](/assets/p2/project_two_code.html).
