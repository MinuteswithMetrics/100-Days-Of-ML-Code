
# Warm Up: Predict Blood Donations
![Banner Image](https://s3.amazonaws.com:443/drivendata/comp_images/2.jpg)

## Introduction

Blood donation has been around for a long time. The first successful recorded transfusion was between two dogs in 1665, and the first medical use of human blood in a transfusion occurred in 1818. Even today, donated blood remains a critical resource during emergencies.

Our dataset is from a mobile blood donation vehicle in Taiwan. The Blood Transfusion Service Center drives to different universities and collects blood as part of a blood drive.

## Problem description:

The UCI Machine Learning Repository is a great resource for practicing your data science skills. They provide a wide range of datasets for testing machine learning algorithms. Finding a subject matter you're interested in can be a great way to test yourself on real-world data problems. Given our mission, we're interested in predicting if a blood donor will donate within a given time window.

Here's what the first few rows of the training set look like:

 Row ID|**Months since Last Donation**|**Number of Donations**|**Total Volume Donated (c.c.)**|**Months since First Donation**|**Made Donation in March 2007**
:-----:|:-----:|:-----:|:-----:|:-----:|:-----:
619|2|50|12500|98|1
664|0|13|3250|28|1
441|1|16|4000|35|1
160|2|20|5000|45|1
358|1|24|6000|77|0

**Use information about each donor's history**

* **Months since Last Donation:** this is the number of monthis since this donor's most recent donation.
* **Number of Donations:** this is the total number of donations that the donor has made.
* **Total Volume Donated:** this is the total amound of blood that the donor has donated in cubuc centimeters.
* **Months since First Donation:** this is the number of months since the donor's first donation.



## Data citation
Data is courtesy of Yeh, I-Cheng via the UCI Machine Learning repository:

Yeh, I-Cheng, Yang, King-Jang, and Ting, Tao-Ming, "Knowledge discovery on RFM model using Bernoulli sequence, "Expert Systems with Applications, 2008, doi:10.1016/j.eswa.2008.07.018.
