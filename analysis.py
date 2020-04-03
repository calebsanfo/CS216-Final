#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[134]:


drug = pd.read_csv('nc/NC_Drug_Poisoning.csv')
gdp = pd.read_csv('nc/NC_GDP.csv')
unemployment = pd.read_csv('nc/NC_Unemployment.csv')
unemployment = unemployment[1:]


# In[135]:


gdp = gdp.set_index('DATE').transpose()
gdp =gdp.reset_index()
gdp = gdp.melt(id_vars=["index"], 
        var_name="Year", 
        value_name="GDP")
gdp.columns=[
    'FIPS',
    'Year',
    'GDP'
]
gdp


# In[137]:


gdp['FIPS'] = gdp['FIPS'].astype(int)
gdp['Year'] = gdp['Year'].astype(int)
# unemployment['FIPS'] = unemployment['FIPS'].astype(int) 
drug['FIPS'] = drug['FIPS'].astype(int)
drug['Year'] = drug['Year'].astype(int)

# merged1 = gdp.merge(unemployment, left_on='FIPS', right_on='FIPS')
# merged2 = merged1.merge(pop, left_on='FIPS', right_on='FIPS')


# In[141]:


drug


# In[142]:


new_df = pd.merge(gdp, drug,  how='right', left_on=['FIPS','Year'], right_on = ['FIPS','Year'])


# In[143]:


new_df


# In[ ]:




