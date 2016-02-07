
# coding: utf-8

# In[5]:

from modules import *
import tools


# In[80]:

pd.DataFrame(X).to_csv('cgn_data.csv', index = False, header = False)


# In[78]:

X = tools.generateRandomClusters(n=500, seperation = 10, uniformity = 1, gridSize=3, randomness=2.5)
plt.scatter(pd.DataFrame(X)[0], pd.DataFrame(X)[1])


# In[74]:

plt.scatter(pd.DataFrame(X)[0], pd.DataFrame(X)[1])


# In[11]:

plt.scatter(X[0], X[1])


# In[7]:

tools.prettyplotdf(pd.DataFrame(X), 'Curtis', 'Nicasia', 'hello world', s=100)


# In[4]:

matplotlib.rc('font', **tools.getfont())

tools.figexamplemultiscatter(niter = 2000, dof = [1,3,5,7,9,50,100,200])


# In[2]:

tools.countrymap('United States')


# In[7]:

tools.prettyhistpercent(pd.Series(np.arange(.01,1,.01)**2))


# In[ ]:



