
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')

import os
if os.name != 'nt':
    from bq.edx2bigquery.edx2bigquery import bqutil
    import pycountry

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
import time
import sys
import json

import networkx as nx

from difflib import SequenceMatcher

from math import radians, cos, sin, asin, sqrt

from IPython.display import clear_output


# In[ ]:



