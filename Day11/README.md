 ## Titanic: Machine Learning from Disaster
*Predict survival on the Titanic passengers

![GitHub](https://d1s0cxawdx09re.cloudfront.net/uploads/2015/04/09_titanic.jpg)


In this notebook, we are building a 3-layer neural network with numpy for the Kaggle Titanic Dataset, and comparing the performance difference between a standard Stochastic Gradient Descent and Adam.

## Introduction
The sinking of the RMS Titanic is one of the most infamous shipwrecks in history. On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.


```python
# Imports
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np # For Mathematical functions 
import pandas as pd # Widely used tool for data manipulation
import matplotlib.pyplot as plt # Visualization
```
