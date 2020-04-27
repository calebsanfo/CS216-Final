#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


drug = pd.read_csv('NC_Drug_Poisoning.csv')
gdp = pd.read_csv('NC_GDP.csv')
unemployment = pd.read_csv('NC_Unemployment.csv')
unemployment = unemployment[1:]
unemployment


# In[3]:


#cleaning up unemployment data
unemployment = unemployment.drop(["Civilian_labor_force_2010","Employed_2010","Unemployed_2010", "Civilian_labor_force_2011","Employed_2011","Unemployed_2011", "Civilian_labor_force_2012","Employed_2012","Unemployed_2012", "Civilian_labor_force_2013","Employed_2013","Unemployed_2013",
"Civilian_labor_force_2014","Employed_2014","Unemployed_2014",
"Civilian_labor_force_2015","Employed_2015","Unemployed_2015",
"Civilian_labor_force_2016","Employed_2016","Unemployed_2016",
"Civilian_labor_force_2017","Employed_2017","Unemployed_2017",
"Civilian_labor_force_2018","Employed_2018","Unemployed_2018","Med_HH_Income_Percent_of_State_Total_2018"], axis = 1)

unemployment = unemployment.rename(columns={"Metro_2013":"Metro Code 2008"})

unemployment["Rural_urban_continuum_code_2013"] = unemployment['Metro Code 2008']
unemployment = unemployment.rename(columns={"Rural_urban_continuum_code_2013":"Metro Code 2007"})
unemployment["Urban_influence_code_2013"] = unemployment['Metro Code 2008']
unemployment = unemployment.rename(columns={"Urban_influence_code_2013":"Metro Code 2009"})
unemployment["Civilian_labor_force_2007"] = unemployment['Metro Code 2008']
unemployment = unemployment.rename(columns={"Civilian_labor_force_2007":"Metro Code 2010"})
unemployment["Employed_2007"] = unemployment['Metro Code 2008']
unemployment = unemployment.rename(columns={"Employed_2007":"Metro Code 2011"})
unemployment["Unemployed_2007"] = unemployment['Metro Code 2008']
unemployment = unemployment.rename(columns={"Unemployed_2007":"Metro Code 2012"})
unemployment["Civilian_labor_force_2008"] = unemployment['Metro Code 2008']
unemployment = unemployment.rename(columns={"Civilian_labor_force_2008":"Metro Code 2013"})
unemployment["Employed_2008"] = unemployment['Metro Code 2008']
unemployment = unemployment.rename(columns={"Employed_2008":"Metro Code 2014"})
unemployment["Unemployed_2008"] = unemployment['Metro Code 2008']
unemployment = unemployment.rename(columns={"Unemployed_2008":"Metro Code 2015"})
unemployment["Civilian_labor_force_2009"] = unemployment['Metro Code 2008']
unemployment = unemployment.rename(columns={"Civilian_labor_force_2009":"Metro Code 2016"})
unemployment["Employed_2009"] = unemployment['Metro Code 2008']
unemployment = unemployment.rename(columns={"Employed_2009":"Metro Code 2017"})
unemployment["Unemployed_2009"] = unemployment['Metro Code 2008']
unemployment = unemployment.rename(columns={"Unemployed_2009":"Metro Code 2018"})

#unemployment = unemployment.insert(4, "Metro Code 2008", [unemployment.loc[:,'Metro Code 2007'].tolist()], allow_duplicates =True)

unemployment = unemployment.transpose()

unemployment.head()


# In[4]:


years = unemployment[3:]
rate=years[6:]
rate = rate.sort_index(ascending=True)
med_income = rate.iloc[0]
rate = rate[7:-1]
rate

#transforming unemployment rate
urate = []
for i in range(11):
    urate.append(rate.iloc[i].tolist())  
urate

flat_list = []
for sublist in urate:
    for item in sublist:
        flat_list.append(item)
flat_list

#transforming median income, need to repeat each number 11 times
#problem with this is the only median income was reported in 2013 --> for more accurate analysis, need more data

income_list = []
for i in range(11):
    income_list.append(med_income.tolist())
        
flat_list2 = []
for sublist in income_list:
    for item in sublist:
        flat_list2.append(item)


# In[5]:


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


# In[6]:


gdp['FIPS'] = gdp['FIPS'].astype(int)
gdp['Year'] = gdp['Year'].astype(int)
# unemployment['FIPS'] = unemployment['FIPS'].astype(int) 
drug['FIPS'] = drug['FIPS'].astype(int)
drug['Year'] = drug['Year'].astype(int)

# merged1 = gdp.merge(unemployment, left_on='FIPS', right_on='FIPS')
# merged2 = merged1.merge(pop, left_on='FIPS', right_on='FIPS')
drug


# In[7]:


new_drug = drug[drug['Year']>2006]
new_drug.head()


# In[8]:


new_df = pd.merge(gdp, new_drug,  how='right', left_on=['FIPS','Year'], right_on = ['FIPS','Year'])
new_df = new_df.drop(["Census Division"], axis = 1)

metro = []
for i in range(1100):
    if new_df['Urban/Rural Category'].iloc[i] == "Noncore":
        metro.append(0)
    else:
        metro.append(1)

new_df["Metro Code"] = metro
new_df = new_df.drop(["Urban/Rural Category"], axis = 1)
new_df["Unemployment Rate"] = flat_list
new_df["Median Income"] = flat_list2

# remove commas from populations
new_df['Population'] = new_df['Population'].str.replace(',', '').astype(int)

new_df


# In[9]:


import plotly
import plotly.figure_factory as ff
import geopandas
import shapely

def ncMap(dataset, values, title):
    fips = list(set(dataset['FIPS'].tolist()))
    fig = ff.create_choropleth(
        fips=fips,
        values=values,
        scope=['North Carolina'],
        show_state_data=True,
        binning_endpoints = list(np.mgrid[min(values):max(values):10j]),
        plot_bgcolor='rgb(229,229,229)',
        paper_bgcolor='rgb(229,229,229)',
        legend_title=title,
        county_outline={'color': 'rgb(255,255,255)', 'width': 0.5},)
    fig.layout.template = None
    fig.show()


# In[10]:


values = new_df.query('Year==2017')['Population'].astype(int).tolist()
title = 'Population by County'
ncMap(new_df, values, title)


# In[11]:


values = new_df.query('Year==2017')['Model-based Death Rate'].astype(float).tolist()
title = 'Model-based Death Rate by County'
ncMap(new_df, values, title)


# In[12]:


values = new_df.query('Year==2017')['GDP'].astype(int).tolist()
title = 'GDP by county'
ncMap(new_df, values, title)


# In[13]:


values = new_df.query('Year==2017')['Unemployment Rate'].astype(float).tolist()
title = 'Unemployment Rate by County'
ncMap(new_df, values, title)


# In[14]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[15]:


ax = sns.regplot(x = "GDP", y="Model-based Death Rate", data=new_df)
plt.title("County GDP by Model-based Death Rate (2007-2017)")
correlation_matrix = np.corrcoef(new_df["GDP"], new_df["Model-based Death Rate"])
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print("r_squared:",r_squared)


# In[16]:


ax = sns.regplot(x = "GDP", y="Model-based Death Rate", data=new_df.query('Year==2008'))
plt.title("County GDP by Model-based Death Rate (2008)")


# In[17]:


ax = sns.regplot(x = "GDP", y="Model-based Death Rate", data=new_df.query('Year==2014'))
plt.title("County GDP by Model-based Death Rate (2014)")


# In[18]:


ax = sns.regplot(x = "Unemployment Rate", y="Model-based Death Rate", data=new_df)
plt.title("County Unemployment Rate by Model-based Death Rate (2007-2017)")
correlation_matrix = np.corrcoef(new_df["Unemployment Rate"], new_df["Model-based Death Rate"])
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print("r_squared:",r_squared)


# In[19]:


ax = sns.regplot(x = "Unemployment Rate", y="Model-based Death Rate", data=new_df.query('Year==2008'))
plt.title("County Unemployment Rate by Model-based Death Rate (2008)")


# In[20]:


ax = sns.regplot(x = "Unemployment Rate", y="Model-based Death Rate", data=new_df.query('Year==2016'))
plt.title("County Unemployment Rate by Model-based Death Rate (2016)")


# In[21]:


ax = sns.regplot(x = "Unemployment Rate", y="Model-based Death Rate", data=new_df.query('`Metro Code`==1').query('Year==2016'))
plt.title("County Unemployment Rate by Model-based Death Rate (2016, Metro Code 1)")


# In[22]:


ax = sns.regplot(x = "Unemployment Rate", y="Model-based Death Rate", data=new_df.query('`Metro Code`==0').query('Year==2016'))
plt.title("County Unemployment Rate by Model-based Death Rate (2016, Metro Code 0)")


# In[23]:


correlation_matrix = np.corrcoef(np.log(new_df["Unemployment Rate"]), new_df["Model-based Death Rate"])
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print("r_squared:",r_squared)


# In[24]:


correlation_matrix = np.corrcoef(new_df["Unemployment Rate"], np.log(new_df["Model-based Death Rate"]))
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print("r_squared:",r_squared)


# In[25]:


correlation_matrix = np.corrcoef(np.log(new_df["Unemployment Rate"]), np.log(new_df["Model-based Death Rate"]))
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print("r_squared:",r_squared)


# In[26]:


correlation_matrix = np.corrcoef(np.log(new_df["GDP"]), new_df["Model-based Death Rate"])
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print("r_squared:",r_squared)


# In[27]:


correlation_matrix = np.corrcoef(new_df["GDP"], np.log(new_df["Model-based Death Rate"]))
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print("r_squared:",r_squared)


# In[28]:


correlation_matrix = np.corrcoef(new_df["GDP"], np.log(new_df["Model-based Death Rate"]))
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print("r_squared:",r_squared)


# In[29]:


correlation_matrix = np.corrcoef(new_df.query("Year==2008")["GDP"], np.log(new_df.query("Year==2008")["Model-based Death Rate"]))
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print("r_squared:",r_squared)


# In[30]:


correlation_matrix = np.corrcoef(np.log(new_df["GDP"]), np.log(new_df["Model-based Death Rate"]))
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2
print("r_squared:",r_squared)


# In[31]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def linearRegressionPrediction(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, shuffle=True, random_state=1)
    reg = LinearRegression().fit(X_train, y_train)
    reg_prediction = reg.predict(X_test)
    print("Features: " + str(x.columns.values) + 
          "\nMean squared error: " + str(mean_squared_error(y_test, reg_prediction)),
          end="\n\n")


# In[32]:


from sklearn import metrics
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_validate

y = new_df['Model-based Death Rate']

x = new_df[['Unemployment Rate']]
linearRegressionPrediction(x, y)

x = new_df[['GDP', 'Population']]
linearRegressionPrediction(x, y)

x = new_df[['Unemployment Rate', 'Population', 'GDP']]
linearRegressionPrediction(x, y)

x = new_df[['Unemployment Rate', 'GDP']]
linearRegressionPrediction(x, y)

x = new_df[['Unemployment Rate', 'Population']]
linearRegressionPrediction(x, y)

