
# coding: utf-8

# In[29]:

get_ipython().magic(u'matplotlib inline')

from bq.edx2bigquery.edx2bigquery import bqutil

import pandas as pd
import numpy as np
import scipy
import sklearn
from sklearn.datasets import fetch_mldata
from sklearn.cluster import KMeans
from numpy import random

from matplotlib import pyplot as plt
import matplotlib.cm as cm # colors
import matplotlib
import matplotlib.ticker as mtick #Add percents to x-axis

import datetime as dt
from datetime import datetime
import gzip
import pickle
import pycountry
import time
import sys
import json
import os

from difflib import SequenceMatcher

from math import radians, cos, sin, asin, sqrt


# In[30]:

import pandas as pd
import numpy as np


# In[31]:

#pd.DataFrame(zip(range(100), [float(np.math.factorial(i)) for i in range(100)])).to_csv('factorial.csv', index = False, header = False)


# In[ ]:



