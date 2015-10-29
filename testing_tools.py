
# coding: utf-8

# In[1]:

from modules import *


# In[2]:

X = tools.generateRandomClusters(seperation = 7)


# In[3]:

tools.prettyplotdf(pd.DataFrame(X), 'Curtis', 'Nicasia', 'hello world', s=100)


# In[4]:

matplotlib.rc('font', **tools.getfont())

tools.figexamplemultiscatter(niter = 2000, dof = [1,3,5,7,9,50,100,200])


# In[2]:

tools.countrymap('United States')


# In[7]:

tools.prettyhistpercent(pd.Series(np.arange(.01,1,.01)**2))


# In[ ]:



