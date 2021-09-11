#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
#Importing the data
df = pd.read_csv("Penntable2.csv")

#print(df.head(10))
#turning it into a pivot table
dfa = df.pivot(
    columns="VariableCode",
    index="RegionCode",
    values="AggValue").reset_index()
print(dfa.head(10))


# In[2]:


#Selecting and renaming the relevant columns
dfa.columns = ["country", "avg_h_worked", "employment", "human_capital", "price_lvl", "population", "r_gdp", "capital_stk"]
print(dfa.head())


# 

# In[3]:


from matplotlib import pyplot as plt

#Applying some descriptive statistics
plt.hist(dfa.r_gdp, bins = 30, edgecolor="pink")
plt.xlabel("Log Real GDP")
plt.ylabel("Percentage")
plt.title("Distribution of the Log real GDP")
plt.show()
plt.clf()


# 

# In[4]:


dfa["r_gdp_per_capita"] = dfa["r_gdp"]/dfa["population"]
print(dfa.head())


# In[5]:


plt.hist(dfa.r_gdp_per_capita, bins = 30, edgecolor="pink")
plt.xlabel("Real GDP per Capita")
plt.ylabel("Percentage")
plt.title("Distribution of the Real GDP per Capita")
plt.show()
plt.clf()


# In[6]:


plt.scatter(dfa.employment, dfa.population)
plt.xlabel("Employment")
plt.ylabel("Population")
plt.title("Relationship between a country's population and its employment")
plt.show()


# In[7]:


x = dfa.employment
y = dfa.population
mask = y < 400
plt.scatter(x[mask], y[mask])
plt.xlabel("Employment")
plt.ylabel("Population")
plt.title("Relationship between a country's population and its employment removed from strong outliers")
plt.show()
plt.clf()


# In[8]:


plt.scatter(dfa.human_capital, dfa.r_gdp_per_capita)
plt.xlabel("Human Capital")
plt.ylabel("Real GDP per Capita")
plt.title("Relationship between human capital and GDP")
plt.show()
plt.clf()


# In[9]:


#Obtaining a correlation table
dfa.corr()


# In[10]:


#As well as a simple statistics table
dfa.describe()


# In[11]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#First regression

dfa = dfa.dropna()
print(len(dfa))
print(dfa.head())
X = dfa[["avg_h_worked", "employment", "human_capital", "price_lvl", "capital_stk"]]
Y = dfa["r_gdp"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.2, random_state = 1)

model = LinearRegression()
model.fit(x_train, y_train)
model.predict(x_test)
R2 = model.score(x_test, y_test)

print(pd.DataFrame({"Feature":x_train.columns.tolist(), "Coefficients":model.coef_}))

print(R2)


# In[12]:


#We should log-transform all the data with huge numbers
dfa["log_emp"] = np.log(dfa.employment)
dfa["log_pop"] = np.log(dfa.population)
dfa["log_rgdp"] = np.log(dfa.r_gdp)
dfa["log_cap"] = np.log(dfa.capital_stk)
dfa["log_rgdp_p_cpt"] = np.log(dfa.r_gdp_per_capita)


# In[13]:


#Descriptive statistics on the logged data:

plt.hist(dfa.log_rgdp, bins = 20, edgecolor = "pink")
plt.xlabel("Log-Real GDP")
plt.ylabel("Frequency")
plt.title("Logged Real GDP Histogram")
plt.show()
plt.clf
plt.scatter(dfa.log_rgdp, dfa.avg_h_worked)
plt.xlabel("Log-Real GDP")
plt.ylabel("Annual average hours worked")
plt.title("Relationship between a country's logged GDP and hours worked")
plt.show()


# In[14]:


#Did log-transforming the data improve the fit of the model?

dfa = dfa.dropna()
print(len(dfa))
print(dfa.head())
X = dfa[["avg_h_worked", "log_emp", "human_capital", "price_lvl", "log_cap"]]
Y = dfa["log_rgdp"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.2, random_state = 2)

model = LinearRegression()
model.fit(x_train, y_train)
model.predict(x_test)
R2 = model.score(x_test, y_test)
print(R2)

pd.DataFrame({"Feature":x_train.columns.tolist(), "Coefficients":model.coef_})

#Since the value of R2 rose, it was a good idea to log-transform the data


# In[15]:


#Is population a better predictor than employment?

X = dfa[["avg_h_worked", "log_pop", "human_capital", "price_lvl", "log_cap"]]
Y = dfa["log_rgdp"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.2, random_state = 2)

model = LinearRegression()
model.fit(x_train, y_train)
model.predict(x_test)
R2 = model.score(x_test, y_test)
print(R2)

pd.DataFrame({"Feature":x_train.columns.tolist(), "Coefficients":model.coef_})

#A bigger R2 indicates that that is the case. Presumably, a bigger population, means more consumers AND more producers/workers.


# 

# In[16]:


#So far, we have worked towards predicting GDP. How about GDP per capita?
#Population is excluded as GDP per capita, in a way, already includes that measure
X = dfa[["avg_h_worked", "human_capital", "price_lvl", "log_cap"]]
Y = dfa["log_rgdp_p_cpt"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.2, random_state = 2)

model = LinearRegression()
model.fit(x_train, y_train)
model.predict(x_test)
R2 = model.score(x_test, y_test)
print(R2)

pd.DataFrame({"Feature":x_train.columns.tolist(), "Coefficients":model.coef_})

#The model performs better at predicting GDP rather than GDP per capita


# In[ ]:




