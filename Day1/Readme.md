# Warm Up: Predict Blood Donations
![Banner Image](https://s3.amazonaws.com:443/drivendata/comp_images/2.jpg)

## Introduction

Blood donation has been around for a long time. The first successful recorded transfusion was between two dogs in 1665, and the first medical use of human blood in a transfusion occurred in 1818. Even today, donated blood remains a critical resource during emergencies.

Our dataset is from a mobile blood donation vehicle in Taiwan. The Blood Transfusion Service Center drives to different universities and collects blood as part of a blood drive.

## Problem description:

The UCI Machine Learning Repository is a great resource for practicing your data science skills. They provide a wide range of datasets for testing machine learning algorithms. Finding a subject matter you're interested in can be a great way to test yourself on real-world data problems. Given our mission, we're interested in predicting if a blood donor will donate within a given time window.

Here's what the first few rows of the training set look like:

        | Months since Last Donation | # of Donations |Total Volume Donated (c.c.) | Months since First Donation | Donation in March 2007
---     | ---                        | ---            | ---                        | ---                          | ---
**619** |                           2|              50| 12500                      | 98                           | 1
**664** |                           0|              13| 3250                       | 28                           | 1
**441** |                          1 |              16| 4000                       | 35                           | 1
**160** |                          2 |              20| 5000                       | 45                           | 1
**358** |                          1 |              24| 6000                       | 77                           | 0

Months since Last Donation	Number of Donations	Total Volume Donated (c.c.)	Months since First Donation	Made Donation in March 2007

## Dataset

There are 14 columns in the dataset, where the patient_id column is a unique and random identifier. The remaining 13 features are described in the section below.

slope_of_peak_exercise_st_segment (type: int): the slope of the peak exercise ST segment, an electrocardiography read out indicating quality of blood flow to the heart
thal (type: categorical): results of thallium stress test measuring blood flow to the heart, with possible values normal, fixed_defect, reversible_defect
resting_blood_pressure (type: int): resting blood pressure
chest_pain_type (type: int): chest pain type (4 values)
num_major_vessels (type: int): number of major vessels (0-3) colored by flourosopy
fasting_blood_sugar_gt_120_mg_per_dl (type: binary): fasting blood sugar > 120 mg/dl
resting_ekg_results (type: int): resting electrocardiographic results (values 0,1,2)
serum_cholesterol_mg_per_dl (type: int): serum cholestoral in mg/dl
oldpeak_eq_st_depression (type: float): oldpeak = ST depression induced by exercise relative to rest, a measure of abnormality in electrocardiograms
sex (type: binary): 0: female, 1: male
age (type: int): age in years
max_heart_rate_achieved (type: int): maximum heart rate achieved (beats per minute)
exercise_induced_angina (type: binary): exercise-induced chest pain (0: False, 1: True)
