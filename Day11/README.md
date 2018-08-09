 ## Titanic: Machine Learning from Disaster
Predict survival on the Titanic passengers
![GitHub](https://d1s0cxawdx09re.cloudfront.net/uploads/2015/04/09_titanic.jpg)


In this notebook, we are building a 3-layer neural network with numpy for the Kaggle Titanic Dataset, and comparing the performance difference between a standard Stochastic Gradient Descent and Adam.




```python
# Imports
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import numpy as np # For Mathematical functions 
import pandas as pd # Widely used tool for data manipulation
import matplotlib.pyplot as plt # Visualization
```
