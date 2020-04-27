#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'


# In[2]:


#incarceration rates from 2001-2016
incarceration= pd.read_csv("crime_and_incarceration_by_state.csv")
incarceration = incarceration.rename(columns={"jurisdiction":"State"})

#Capitalize all states to follow the format of other datasets
for (i, state) in incarceration["State"].iteritems():
    lst = state.split() #capitalize()
    fixed= []
    for word in lst:
        word= word.capitalize()
        fixed.append(word)
    new= " ".join(fixed)
    incarceration.at[i,"State"] = new

incarceration= incarceration.rename(columns={"year": "Year"})  
incarceration= incarceration.drop("state_population", axis=1)
incarceration.groupby('State', as_index=False)
incarceration.tail(50)


# In[3]:


#drug overdose rate from 1999-2017
drug= pd.read_csv("Drug_Poisoning_Mortality_by_State.csv")
#drop unneeded columns
drug = drug.drop(["Sex","Age Group","Race and Hispanic Origin", "Crude Death Rate",
"Standard Error for Crude Rate","Low Confidence Limit for Crude Rate","Upper Confidence Limit for Crude Rate",
"Standard Error Age-adjusted Rate","Lower Confidence Limit for Age-adjusted Rate","Upper Confidence for Age-adjusted Rate",
"State Crude Rate in Range","US Crude Rate","US Age-adjusted Rate",
"Unit"], axis = 1)
drug= drug.rename(columns={"Age-adjusted Rate":"Overdose Rate"})

states = drug["State"]
for i in range(len(drug)):
    if states[i] == "District of Columbia":
         drug = drug.drop([i])
    if states[i] == "United States":
        drug = drug.drop([i])
drug

#to match incareration data set
drug2 = drug[drug['Year']>2000]
drug2 = drug2[drug2['Year']<2017]
drug2


# In[4]:


#unemployment rates from 1980-2018
unemployment= pd.read_csv("unemployment_rate.csv")
#drop unneeded years data
unemployment= unemployment.drop(["Fips", "1980", "1981", "1982", "1983", "1984", "1985", "1986",
        "1987", "1988", "1989", "1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998"],axis=1)
unemployment = unemployment.rename(columns={"Area":"State"})
unemployment = unemployment.drop([8,51,52,53])
unemployment.head(10)

#of len 20 (num of years), new col for state names
new = []
for i in range(50):
    for x in range(20):
        new.append(unemployment["State"].iloc[i])
#new

years = unemployment.iloc[:,1:]
years = years.transpose()
years

#single unemployment rate col
un = []
for i in range(20):
    un.append(years.iloc[i].tolist())  
#un

unemploy = []
for sublist in un:
    for item in sublist:
        unemploy.append(item)
#unemploy

#col of years
yr = [2018,2017,2016,2015,2014,2013,2012,2011,2010,2009,2008,2007,2006,2005,2004,2003,2002,2001,2000,1999]
yr2 = []
for i in range(50):
    yr2.append(yr)  
#yr2

year = []
for sublist in yr2:
    for item in sublist:
        year.append(item)
#year

#combining
finalunemployment = pd.DataFrame({"State": new, "Year": year,"Unemployment Rate" :unemploy})
finalunemployment = finalunemployment[finalunemployment["Year"]<2018]

#to match incarceraton data
un2 = finalunemployment[finalunemployment['Year']>2000]
un2 = un2[un2['Year']<2017]
un2


# In[5]:


#median income from 1984-2018
income= pd.read_csv("income_fixed.csv")
#drop uneeded years
income= income.drop(["1984 (19)", "1985 (20)", "1986",
        "1987 (21)", "1988", "1989", "1990", "1991", "1992 (22)", "1993 (23)", "1994 (24)", "1995 (25)",
        "1996", "1997", "1998", "2013 (39)"],axis=1)
income = income.rename(columns={"2004(revised)":"2004"})
income = income.rename(columns={"2000 (30)":"2000"})
income = income.rename(columns={"1999 (29)":"1999"})
income = income.rename(columns={"2013 (38)":"2013"})
income = income.rename(columns={"2010 (37)":"2010"})
income = income.rename(columns={"2009 (36)": "2009"})
income = income.drop([8])
#change to wide data
#new state column (this matches the number of years so we need to figure out the specific date range)
#need to drop DC row

years = income.iloc[:,1:]
years = years.transpose()
years

#stack all 0-51 on each other for single col of income
income2 = []
for i in range(20):
    income2.append(years.iloc[i].tolist())  
income2

flat_list = []
for sublist in income2:
    for item in sublist:
        flat_list.append(item)
flat_list

#combining cols
finalincome = pd.DataFrame({"State": new, "Year": year,"Median Income" :flat_list})
finalincome = finalincome[finalincome["Year"]<2018]
finalincome

#to match incarceration data
in2 = finalincome[finalincome['Year']>2000]
in2 = in2[in2['Year']<2017]
in2


# In[6]:


#merging data
#dataframe 1: with all above data (drug2, un2,in2, incarceration)
new_df = pd.merge(drug2, un2,  how='left', left_on=['State','Year'], right_on = ['State','Year'])
new_df = pd.merge(new_df, in2,  how='left', left_on=['State','Year'], right_on = ['State','Year'])
new_df = pd.merge(new_df, incarceration,  how='inner', left_on=['State','Year'], right_on = ['State','Year'])
new_df.tail(50)
#incarceration rate data has some missing values


# In[7]:


#dataframe 2: without incarceration data (drug, finalunemployment, finalincome) 1999-2017
new_df2 = pd.merge(drug, finalunemployment,  how='left', left_on=['State','Year'], right_on = ['State','Year'])
new_df2 = pd.merge(new_df2, finalincome,how='left', left_on=['State','Year'], right_on = ['State','Year'])
new_df2


# In[8]:


state_abbrev_map = {'Alabama': 'AL','Alaska': 'AK','American Samoa': 'AS','Arizona': 'AZ','Arkansas': 'AR','California': 'CA','Colorado': 'CO','Connecticut': 'CT','Delaware': 'DE','District of Columbia': 'DC','Florida': 'FL','Georgia': 'GA','Guam': 'GU','Hawaii': 'HI','Idaho': 'ID','Illinois': 'IL','Indiana': 'IN','Iowa': 'IA','Kansas': 'KS','Kentucky': 'KY','Louisiana': 'LA','Maine': 'ME','Maryland': 'MD','Massachusetts': 'MA','Michigan': 'MI','Minnesota': 'MN','Mississippi': 'MS','Missouri': 'MO','Montana': 'MT','Nebraska': 'NE','Nevada': 'NV','New Hampshire': 'NH','New Jersey': 'NJ','New Mexico': 'NM','New York': 'NY','North Carolina': 'NC','North Dakota': 'ND','Northern Mariana Islands':'MP','Ohio': 'OH','Oklahoma': 'OK','Oregon': 'OR','Pennsylvania': 'PA','Puerto Rico': 'PR','Rhode Island': 'RI','South Carolina': 'SC','South Dakota': 'SD','Tennessee': 'TN','Texas': 'TX',
                    'Utah': 'UT','Vermont': 'VT','Virgin Islands': 'VI','Virginia': 'VA',
                    'Washington': 'WA','West Virginia': 'WV','Wisconsin': 'WI','Wyoming': 'WY'}

# Convert state names to abbrevations
for i in range(0, len(new_df)):
    if(new_df['State'][i] in state_abbrev_map):
        new_df['State'][i] = state_abbrev_map[new_df['State'][i]]
for i in range(0, len(new_df2)):
    if(new_df2['State'][i] in state_abbrev_map):
        new_df2['State'][i] = state_abbrev_map[new_df2['State'][i]]

new_df2


# In[9]:


new_df = new_df.replace(",","", regex=True)
new_df2 = new_df2.replace(",","", regex=True)
new_df2["Median Income"] = new_df["Median Income"].astype(float)


# In[10]:


import seaborn as sns
import matplotlib as plt
ax = sns.regplot(x = "Population", y="Deaths", data=new_df)


# In[11]:


ax = sns.regplot(x = "Population", y="Overdose Rate", data=new_df)


# In[12]:


ax = sns.regplot(x = "Deaths", y="Overdose Rate", data=new_df)


# In[13]:


ax = sns.regplot(x = "Unemployment Rate", y="Overdose Rate", data=new_df2)


# In[14]:


ax = sns.regplot(x = "Median Income", y="Overdose Rate", data=new_df2)


# In[15]:


import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


# In[16]:


get_ipython().system('pip install -q git+https://github.com/tensorflow/docs')


# In[17]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


# In[18]:


import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


# In[19]:


dataset = new_df.copy()
dataset = dataset.drop(columns=["State", "Year"])
dataset = dataset.replace(',','', regex=True)
dataset = dataset.drop([510])
dataset.to_csv("out.csv")
dataset.astype(float)
dataset["Median Income"] = dataset["Median Income"].astype(float)
dataset["incarceration rate (%)"] = dataset["incarceration rate (%)"].astype(float)


# In[20]:


dataset.tail()


# In[21]:


dataset.isna().sum()


# In[22]:


train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)


# In[23]:


sns.pairplot(train_dataset[["Deaths", "Population", "Overdose Rate", "Unemployment Rate", "Median Income", "prisoner_count", "incarceration rate (%)"]], diag_kind="kde")


# In[24]:


train_stats = train_dataset.describe(include='all')
train_stats
train_stats.pop("Deaths")
train_stats = train_stats.transpose()


# In[25]:


train_labels = train_dataset.pop('Deaths')
test_labels = test_dataset.pop('Deaths')


# In[26]:


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset.astype(float))
normed_test_data = norm(test_dataset.astype(float))
normed_train_data


# In[27]:


def build_model():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
    return model

model = build_model()


# In[28]:


model.summary()


# In[29]:


example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
example_result


# In[30]:


EPOCHS = 10000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[tfdocs.modeling.EpochDots()])


# In[31]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# In[32]:


plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)


# In[33]:


plotter.plot({'Basic': history}, metric = "mae")
plt.ylabel('MAE [Deaths]')


# In[34]:


plotter.plot({'Basic': history}, metric = "mse")
plt.ylabel('MSE [Deaths^2]')


# In[35]:


test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Deaths]')
plt.ylabel('Predictions [Deaths]')


# In[36]:


error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [Deaths]")
_ = plt.ylabel("Count")


# In[37]:


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


# In[38]:


import plotly.graph_objects as go

year = 2000
df_by_year = new_df2[(new_df2['Year']== year )]

fig = go.Figure(data=go.Choropleth(
    locations=new_df2['State'], # Spatial coordinates
    z = new_df2['Overdose Rate'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Overdose Rate",
))

fig.update_layout(
    title_text = str(year) + ' State Overdose Rates',
    geo_scope='usa', # limite map scope to USA
)
plotly.offline.iplot(fig)
#df_by_year


# In[39]:


def createMapOverMultipleYears(dataframe, col_name, unit_specifier):
    data_slider = []
    max_val = max(dataframe[col_name])
    min_val = min(dataframe[col_name])
    
    for year in dataframe.Year.unique():
        df_by_year = dataframe[(dataframe['Year']== year )]

        for col in df_by_year.columns:
            df_by_year[col] = df_by_year[col].astype(str)
    
        df_by_year['text'] = df_by_year['State'] + 'Pop: ' + df_by_year['Population'] + col_name + ': ' + " " + df_by_year[col_name] + unit_specifier
    
        data_by_year = dict(
            type='choropleth',
            locations=df_by_year['State'], # Spatial coordinates
            z = df_by_year[col_name].astype(float), # Data to be color-coded
            locationmode = 'USA-states', # set of locations match entries in `locations`
            colorscale = 'Reds',
            colorbar_title = col_name + '\n' + unit_specifier,
            zmin = min_val,
            zmax = max_val)
    
        data_slider.append(data_by_year)  # I add the dictionary to the list of dictionaries for the slider

    steps = []
    for i in range(len(data_slider)):
        step = dict(method='restyle',
                    args=['visible', [False] * len(data_slider)],
                    label='Year {}'.format(i + min(dataframe['Year']))) # label to be displayed for each step (year)
        step['args'][1][i] = True
        steps.append(step)
    sliders = [dict(active=0, pad={"t": 1}, steps=steps)]
    
    layout = dict(geo=dict(scope='usa',
                           projection={'type': 'albers usa'}),
                  sliders=sliders)
    
    
    final_fig = dict(data=data_slider, layout=layout) 
    plotly.offline.iplot(final_fig)


# In[40]:


createMapOverMultipleYears(new_df2, 'Overdose Rate', 'per 100,000 people')


# In[41]:


createMapOverMultipleYears(new_df2, 'Population', '')


# In[42]:


createMapOverMultipleYears(new_df, 'prisoner_count', '')


# In[43]:


createMapOverMultipleYears(new_df2, 'Unemployment Rate', '%')

