# %% [markdown]
# Introduction:

# %% [markdown]
# In this project, I am analysing customers who are churning (i.e., cancelling their subscription with the firm) from a top e-commerce company's dataset in order to develop a churn prediction model.

# %% [markdown]
# Goal:

# %% [markdown]
# • Construct a churn prediction model that makes accurate predictions about which customers will leave your business and which will stay. The organisation may then use this information to take preventative measures to keep these clients and lower the churn rate.
# 
# • Run an in-depth exploratory study of the given client data to learn more about their habits and preferences. This entails looking for generalisations and trends in data. The results of this study may be used to gain a deeper understanding of the company's clientele and to guide strategic planning.

# %% [markdown]
# Features:

# %% [markdown]
# CustomerID: Individual client ID
# Churn: Churn Flag
# Tenure: Tenure of the client inside the company
# PreferredLoginDevice: Preferred device of client login
# CityTier: Tier city
# WarehouseToHome: The distance from the warehouse to the customer's residence
# PreferredPaymentMode: Customer's preferred form of payment
# Gender: Gender of the client
# HourSpendOnApp: Hours spent using mobile applications or websites
# NumberOfDeviceRegistered: Total number of devices of a certain consumer are recorded
# PreferedOrderCat: Customer's preferred order category from the previous month
# SatisfactionScore: Customer satisfaction rating for the service
# MaritalStatus: Marital status of customer
# NumberOfAddress: Total number of additional additions for a certain client
# OrderAmountHikeFromlastYear: Ordered percentage gains from the previous year
# CouponUsed: The total number of coupons utilised in the previous month
# OrderCount: The total amount of orders placed throughout the last month
# DaySinceLastOrder: Day since the customer's previous purchase
# CashbackAmount: Average cashback for the previous month

# %% [markdown]
# Project Plan

# %% [markdown]
# • Overview of Dataset
# Take some time to read through the available customer data and get acquainted with the variables and how they are organised.
# Verify the completeness of the data and look for any discrepancies or missing numbers.
# Find out if any pre-processing of the data is required.
# 
# • Preliminary Analysis of Data
# Find anomalies by examining the variables' distributions.
# Look for patterns and correlations by probing the interplay of different factors.
# Create a data visualisation to better understand your customers' habits and personas.
# 
# • Pre-Processing
# Deal with missing values, transform variables to suitable data types, and fix any other problems with the data to make it usable.
# Determine which variables are most crucial to including in the model you're developing.
# 
# • ML Modelling
# Create a model that can foresee which clients are most likely to cancel their service.
# 

# %% [markdown]
# Coding

# %%
#Imporing all required libraries for the project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler , LabelEncoder
from sklearn.svm import SVC

# Additional imports
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score , confusion_matrix , classification_report
from sklearn.model_selection import GridSearchCV, cross_validate

import warnings
warnings.simplefilter(action='ignore')

# %%
# Data loading from an excel sheet and overviewing the primary data

df = pd.read_excel('E Commerce Dataset.xlsx', sheet_name='E Comm')
df.head()

# %%
df.shape

# %%
df.info()

# %%
df.nunique()

# %%
# Creating columns to list

columns = df.columns.to_list()
columns

# %%
df.select_dtypes(exclude=np.number).columns

# %%
df.describe(include='O').style.background_gradient(axis=None , cmap = "Blues" , vmin = 0 , vmax = 9000  )

# %%
# Showing the unique values on each column

for col in df.columns:
    if df[col].dtype == object:
        print(str(col) + ' : ' + str(df[col].unique()))
        print(df[col].value_counts())
        print("________________________________________________________________________________")

# %%
df.select_dtypes(include=np.number).columns

# %%
# Transposing df's stats. Show mean as bars and color gradient for std, median, and max.

df.describe().T.style.bar(subset=['mean']).background_gradient(subset=['std','50%','max'])

# %%
for col in df.columns:
    if df[col].dtype == float or df[col].dtype == int:
        print(str(col) + ' : ' + str(df[col].unique()))
        print(df[col].value_counts())
        print("________________________________________________________________________________")

# %%
# Merging phone and mobile in one because they are the same

df.loc[df['PreferredLoginDevice'] == 'Phone', 'PreferredLoginDevice' ] = 'Mobile Phone'
df.loc[df['PreferedOrderCat'] == 'Mobile', 'PreferedOrderCat' ] = 'Mobile Phone'

# %%
df['PreferredLoginDevice'].value_counts()

# %%
# As COD is Cash on Delivery and CC is Credit Card so merging them

df.loc[df['PreferredPaymentMode'] == 'COD', 'PreferredPaymentMode' ] = 'Cash on Delivery'   # uses loc function
df.loc[df['PreferredPaymentMode'] == 'CC', 'PreferredPaymentMode' ] = 'Credit Card'

# %%
df['PreferredPaymentMode'].value_counts()

# %%
# converting num_cols to categories
df2 = df.copy()
for col in df2.columns:
  if col == 'CustomerID':
    continue

  else:
    if df2[col].dtype == 'int':
      df2[col] = df[col].astype(str)

df2.dtypes

# %%
# Categorical columns after conversion 

df2.describe(include='O').style.background_gradient(axis=None , cmap = "Blues" , vmin = 0 , vmax = 9000  )

# %%
# Numerical columns after conversion

df2.describe().T.style.bar(subset=['mean']).background_gradient(subset=['std','50%','max'])

# %%
df.duplicated().sum()

# %%
# Creating the sum of null values

grouped_data = []
for col in columns:
    n_missing = df[col].isnull().sum()
    percentage = n_missing / df.shape[0] * 100
    grouped_data.append([col, n_missing, percentage])

# Create a new DataFrame from the grouped data
grouped_df = pd.DataFrame(grouped_data, columns=['column', 'n_missing', 'percentage'])

# Group by 'col', 'n_missing', and 'percentage'
result = grouped_df.groupby(['column', 'n_missing', 'percentage']).size()
result

# %%
import sweetviz as sv

# Generate the report
report = sv.analyze(df)

# Show the report in a Jupyter notebook
report.show_html()

# %% [markdown]
# EDA (Exploratory Data Analysis)

# %% [markdown]
# Here are some business questions can be asked.

# %% [markdown]
# 1. Does churn and gender have a connection? What Gender Has the Most Orders?
# 2. Which martial status has the highest rate of turnover?
# 3. Which CityTier has a larger OrderCount and Tenure?
# 4. Does a customer with a high satisfaction rating spend a lot of time on the app?
# 5. Is there a relationship between HourSpendOnApp and SatisfactionScore?
# 6. What CityTier's HourSpendOnApp is the highest?
# 7. What connection exists between CityTier and NumberOfAddress in the churn segment?
# 8. How are complaints and days since last order related?
# 9. Does PreferredLoginDevice have any connection to Churn?
# 10. How far is it from a warehouse to a customer's home in a different city tier?
# 11. Are there distinct preferred items throughout the CityTiers?
# 12. What type of payment method does each CityTier prefer?
# 13. Which CityTier's OrderCount is the highest?
# 14. Does the order amount's percentage rise from the previous year have an impact on churn rate?
# 15. How are complaints and days since last order related?
# 16. What is the ordercount for users who spend a lot of time on the app?
# 17. Does the selected order category influence the turnover rate?
# 18. Do patrons who used more coupons have reduced rates of churn?
# 19. Is there a relationship between the quantity of orders placed in the previous month and the satisfaction score?
# 20. Do order counts inside churn and CashbackAmount have a relationship?
# 21. Are unhappy consumers more inclined to leave?

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
binary_cat_cols = ['Complain']
outcome = ['Churn']
cat_cols = ['PreferredLoginDevice', 'CityTier', 'PreferredPaymentMode',
       'Gender', 'NumberOfDeviceRegistered', 'PreferedOrderCat',
       'SatisfactionScore', 'MaritalStatus', 'NumberOfAddress', 'Complain']
num_cols = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 'OrderAmountHikeFromlastYear', 'CouponUsed', 'OrderCount', 'DaySinceLastOrder', 'CashbackAmount']

# %%
df_c = df[df['Churn']==1].copy()
df_nc = df[df['Churn']==0].copy()

fig, ax = plt.subplots(2,4,figsize=(20, 15))
fig.suptitle('Density of Numeric Features by Churn', fontsize=20)
ax = ax.flatten()

for idx,c in enumerate(num_cols):
    sns.kdeplot(df_c[c], linewidth= 3,
             label = 'Churn',ax=ax[idx])
    sns.kdeplot(df_nc[c], linewidth= 3,
             label = 'No Churn',ax=ax[idx])

    ax[idx].legend(loc='upper right')

plt.show()

# %% [markdown]
# Distributions Insights Of the Numeric Features

# %% [markdown]
# Longer-tenured clients appear to be less inclined to turnover. Understandable because a longer stay suggests satisfaction.
# Churn rate appears to be consistent across levels for CityTier. City tier does not appear to be a factor in churn.
# WarehouseToHome: A lower churn rate is associated with closer warehouses to homes. Deliveries made more quickly could increase satisfaction.
# HourSpendOnApp: Longer app usage is associated with lower churn. Engagement with an app is positive.
# More registered devices are associated with reduced churn in terms of NumberOfDeviceRegistered. Convenience is increased by access across devices.
# Higher satisfaction scores closely correlate with lesser churn, as would be predicted. driving force.
# NumberOfAddress: There is a slight decline in churn as the number of addresses rises. Loyalty is indicated by more addresses.
# More complaints are linked to increased churn, but the connection isn't very strong. Remarks reduce satisfaction.
# Big spenders from last year are less likely to churn, according to OrderAmountHikeFromLastYear. good to keep major clients.
# CouponUsed: The use of coupons is associated with lesser churn. Coupons increase allegiance.
# OrderCount: Lower churn is associated with higher order counts. Regular use creates habits.
# DaySinceLastOrder: A longer time period since the last order is associated with greater churn. A excellent predictor is recent events.

# %%
df2 = df.copy()

df_c = df2[df2['Churn']=='1'].copy()
df_nc = df2[df2['Churn']=='0'].copy()

fig, ax = plt.subplots(4,3,figsize=(20, 18))
fig.suptitle('Density of Numeric Features by Churn', fontsize=20)
ax = ax.flatten()

for idx,c in enumerate(cat_cols):
    sns.histplot(df_c[c], linewidth= 3,
             label = 'Churn',ax=ax[idx])
    sns.histplot(df_nc[c], linewidth= 3,
             label = 'No Churn',ax=ax[idx])

    ax[idx].legend(loc='upper right')

plt.show()

# %%
# creating some color palettes for vizualizations

pie_palette = ['#3E885B','#7694B6','#85BDA6', '#80AEBD', '#2F4B26', '#3A506B']
green_palette = ['#2F4B26', '#3E885B', '#85BDA6', '#BEDCFE', '#C0D7BB']
blue_palette = ['#3A506B', '#7694B6', '#80AEBD', '#5BC0BE', '#3E92CC']
custom_palette = ['#3A506B', '#7694B6', '#80AEBD', '#3E885B', '#85BDA6']
red_palette = ['#410B13', '#CD5D67', '#BA1F33', '#421820', '#91171F']

# %% [markdown]
# 1-Is there a relationship between Gender and Churn? & Which Gender has more Orders?

# %%
df['Gender'].value_counts()

# %%
df.groupby("Churn")["Gender"].value_counts()

# %%
df.groupby("PreferredLoginDevice")["OrderCount"].value_counts()

# %%
gender_orders = df.groupby('Gender')['OrderCount'].mean().plot(kind='bar')

gender_orders

# %%
percentageM =600/3384 * 100

percentageM

# %%
percentageF =348/2246 * 100

percentageF

# %%
import pandas as pd
import plotly.express as px

# Create figure
fig = px.pie(df, values='Churn', names='Gender')
fig.update_traces(marker=dict(colors=['pink ', 'baby blue']))

# Update layout
fig.update_layout(
  title='Churn Rate by Gender',
  legend_title='Gender'
)

# Show plot
fig.show()

# %% [markdown]
# It appears that men are more likely to cancel their subscriptions to the app than women are; 63.3% of men have left the service. This suggests that the company would benefit from focusing more on things that appeal to men.

# %% [markdown]
# 2-Which MartialStatus has the highest Churn rate?

# %%
df.groupby("Churn")["MaritalStatus"].value_counts()

# %%
sns.countplot(x='MaritalStatus',hue='Churn',data=df,palette='Set2')
plt.title("churn Rates by MaritalStatus")
plt.ylabel("Churn Rate")

# %% [markdown]
# The married are the highest customer segment in the comapny may be the comapny should consider taking care of the products that suits the single and the married customers as the singles are the most likely to churn from the app.

# %% [markdown]
# 3-Which CityTier has higher Tenure and OrderCount?

# %%
df_grouped_tenure = df.groupby('CityTier')['Tenure'].agg(['mean', 'max'])
df_grouped_tenure

# %%
df_grouped_OrderCount = df.groupby('CityTier')['OrderCount'].agg(['mean', 'max'])
df_grouped_OrderCount

# %%
df.groupby("CityTier")["OrderCount"].mean()

# %% [markdown]
# Citytier 3 has the highest order avg but it not to be a strong factor in the customer churning

# %% [markdown]
# 4 - Is Customer with High SatisfactionScore have high HourSpendOnApp?

# %%
df['SatisfactionScore'].dtypes

# %%
import matplotlib.pyplot as plt

# plot
fig = px.histogram(df2, x="HourSpendOnApp", y="SatisfactionScore", orientation="h", color="Churn" ,text_auto= True , title="<b>"+'HourSpendOnApp Vs SatisfactionScore' , color_discrete_sequence = ['#BA1F33','#3A506B','#3E885B'])

# Customize the plot
fig.update_layout(hovermode='x',title_font_size=30)
fig.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
xaxis_title='HourSpendOnApp',
yaxis_title='SatisfactionScore',
)
fig.show()

# %% [markdown]
# As we see people with less satisfaction score spend less time on the app than the people of satisfaction score 5 but also i do not think there is any realation between the satisfaction score and people's spent time on the app.

# %% [markdown]
# 5-Which CityTier has the most HourSpendOnApp?

# %%
g = sns.FacetGrid(df, col='CityTier')
g.map(sns.distplot, 'HourSpendOnApp')

# %% [markdown]
# City tier 1 has the most spended hours on the app.

# %% [markdown]
# 6 - What is the relation between NumberOfAddress and CityTier within the churn segment?

# %%
df.groupby("CityTier")["NumberOfAddress"].value_counts()

# %%
import seaborn as sns
sns.violinplot(x='CityTier', y='NumberOfAddress', data=df[df['Churn']==1])

# %% [markdown]
# City tier and address count are inversely related. The average number of addresses decreases as the CityTier increases, and the distribution becomes more concentrated. Customers in CityTier 1 cities often have more addresses than those in CityTier 2 and 3 cities. The correlation between address density and location type (metro vs. smaller cities vs. towns) implies that population density has an effect on the number of addresses customers have.

# %% [markdown]
# 7 - What is the relation between Complain and DaySinceLastOrder?

# %%
df[['DaySinceLastOrder', 'Complain']].corr()

# %%
import plotly.express as px

fig = px.scatter(df, x='DaySinceLastOrder', y='Complain', facet_col='Churn')
fig.update_layout(hovermode='closest')
fig.show()

# %% [markdown]
# There is a weak negative relation between complainig and the number of dayes since last order.

# %% [markdown]
# 8-Is there a relationship between PreferredLoginDevice and churn?

# %%
# Bar chart with churn rate
import seaborn as sns
# sns.catplot(x='PreferredLoginDevice', y='Churn', data=df, kind='bar')

# Group the data by 'OverTime' and 'Attrition', and calculate the count
grouped_data = df.groupby(['PreferredLoginDevice', 'Churn']).size().unstack().plot(kind='bar', stacked=True)

# Set the plot title, x-label, and y-label
plt.title('Churn by PreferredLoginDevice ')
plt.xlabel('PreferredLoginDevice')
plt.ylabel('Count')

# Show the plot
plt.show()

# %% [markdown]
# Mobile phone users are likely to churn may be this indicates a problem on the app user experience on the app mobile version.

# %% [markdown]
# 9 - What is distancebetween warehosue to customer house in different city tier?

# %%
df3 = df.copy()

df3['CityTier'].astype('str')
plt.figure(figsize = (5,7))
sns.stripplot(x = 'CityTier', y = 'WarehouseToHome', data = df3, jitter = False)
plt.ylabel(' Distance between warehouse to home');

# %% [markdown]
# Inference: As the distance from warehouse to home is similar in all city tier which means company had build warehouse in lower city tier also.

# %% [markdown]
# 10 - Does different citytiers has different prefered products?

# %%
import plotly.express as px
earth_palette = ["#A67C52", "#8F704D", "#B09B71", "#7E786E"]


fig=px.histogram(df,x="PreferedOrderCat",facet_col="CityTier",color="CityTier",color_discrete_sequence=earth_palette,text_auto= True , title="<b>"+'CityTier Vs PreferedOrderCat')

# Customize the plot
fig.update_layout(hovermode='x',title_font_size=30)
fig.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
xaxis_title='PreferredPaymentMode',
yaxis_title='count',
)
fig.show()

# %% [markdown]
# Laptop & accesories and mobile phones are the prefered category for all the city tiers.

# %% [markdown]
# 11- What is the preferred payment mode for different CityTiers?

# %%
df2['PreferredPaymentMode'].value_counts()

# %%
df2.groupby('CityTier')[['PreferredPaymentMode']].value_counts()

# %%
import plotly.express as px

fig=px.histogram(df2,x="PreferredPaymentMode",facet_col="CityTier",color="CityTier",color_discrete_sequence=red_palette,text_auto= True , title="<b>"+'CityTier Vs PaymentMethod')

# Customize the plot
fig.update_layout(hovermode='x',title_font_size=30)
fig.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
xaxis_title='PreferredPaymentMode',
yaxis_title='count',
)
fig.show()

# %% [markdown]
# preferred payment method for CityTier '1' ==> DebitCard
# preferred payment method for CityTier '2' ==> UPI
# preferred payment method for CityTier '3' ==> E wallet

# %% [markdown]
# 12-Which CityTier has the highest OrderCount?

# %%
df2.groupby('CityTier')[['OrderCount']].sum()

# %%
fig = px.histogram(df2, x="OrderCount", y="CityTier", orientation="h", color="CityTier" ,text_auto= True , title="<b>"+'CityTier Vs Sum of OrderCount' , color_discrete_sequence = ['#BA1F33','#3A506B','#3E885B'])

# Customize the plot
fig.update_layout(hovermode='x',title_font_size=30)
fig.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
xaxis_title='Sum of OrderCount',
yaxis_title='count',
)
fig.show()

# %% [markdown]
# CityTier '1' has highest order count with 10298 orders.

# %% [markdown]
# 13-Does the percentage increase in order amount from last year affect churn rate?

# %%
df2['OrderAmountHikeFromlastYear'].value_counts()

# %%
df2.groupby('OrderAmountHikeFromlastYear')['Churn'].count()

# %%
comp_ten = df2.groupby(["OrderAmountHikeFromlastYear", "Churn"]).size().reset_index(name="Count")

# Create a bubble chart using Plotly
fig_bubble = px.scatter(comp_ten, x="OrderAmountHikeFromlastYear", y="Count", size="Count", color="Churn", title="<b>"+'OrderAmountHikeFromlastYear VS Churn',
                        color_discrete_sequence=["#d62728", "#1f77b4"])

# Customize the plot
fig_bubble.update_layout(hovermode='x',title_font_size=30)
fig_bubble.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
xaxis_title='OrderAmountHikeFromlastYear',
yaxis_title='count',
)
fig_bubble.show()

# %% [markdown]
# OrderAmountHikeFromlastYear has a positive effect on churn rate, as shown by the graph, and we should prioritise customers with a turnover rate between 12 and 14 percent.

# %% [markdown]
# 14-What is the relation between Complain and DaySinceLastOrder for churned customers?

# %%
df_c.groupby('Complain')[['DaySinceLastOrder']].sum()

# %%
fig = px.histogram(df2, x="DaySinceLastOrder", color="Complain",text_auto= True , title="<b>DaySinceLastOrder Vs Complain" , color_discrete_sequence = ['#BA1F33','#3A506B'],
                   marginal="box") # or violin, rug)

# Customize the plot
fig.update_layout(hovermode='x',title_font_size=30)
fig.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
xaxis_title='DaySinceLastOrder',
yaxis_title='count',
)
fig.show()

# %% [markdown]
# There is one customer whose DaySinceLastOrder is significantly greater than the average, but he or she is an anomaly who can be eliminated by focusing on the remaining customers.

# %% [markdown]
# 15-What is the order counts for customers with high HourSpendOnApp?

# %%
df2['HourSpendOnApp'].agg(['min','max'])

# %%
# Define the bin range
bins = [0 , 1 , 3 , 6]
label = ['low' , 'medium' , 'high']
# Create a new column 'HourSpendOnApp_bins' with the binned values
df2['HourSpendOnApp_bins'] = pd.cut(df2['HourSpendOnApp'], bins=bins , labels = label)

# %%
df2.groupby(['HourSpendOnApp_bins','OrderCount'])[['CustomerID']].count()

# %%
sunbrust_gr = df2.loc[:,['HourSpendOnApp_bins','OrderCount']].dropna()

# %%
fig = px.sunburst(sunbrust_gr,path=['HourSpendOnApp_bins','OrderCount'],title="<b>"+'HourSpendOnApp VS OrderCount',template="plotly" , color_discrete_sequence=["#78b4d5", "#d57f86" ,'#3E885B'])
fig.update_layout(hovermode='x',title_font_size=30)
fig.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
)
fig.update_traces(textinfo="label+percent parent")

fig.show()

# %% [markdown]
# Segment of customers has high spendtime on App has OrderCount 2 with percentage 67%

# %% [markdown]
# 16-Is there a relationship between preferred order category and churn rate?

# %%
df2.groupby(['PreferedOrderCat' , 'Gender'])[['CustomerID']].count()

# %%
# Group and count by 'PreferedOrderCat' and 'Churn'
ordercat_churnrate = pd.DataFrame(df2.groupby('PreferedOrderCat')['Gender'].value_counts())
ordercat_churnrate = ordercat_churnrate.rename(columns={'Gender': 'Count'})
ordercat_churnrate = ordercat_churnrate.reset_index()


fig = px.histogram(ordercat_churnrate, x='PreferedOrderCat', y = 'Count',color='Gender', barmode='group',color_discrete_sequence=pie_palette,title="<b>"+'Prefered Category Vs Gender', barnorm = "percent",text_auto= True)
fig.update_layout(hovermode='x',title_font_size=30)
fig.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
xaxis_title='PreferedOrderCat',
yaxis_title='count',
)
fig.show()

# %% [markdown]
# Top 2 Preferd Category For Males == > [ Others , Mobile Phone ]
# Top 2 Preferd Category For Females == > [ Grocery , Fashion ]

# %% [markdown]
# 17-Do customers who used more coupons have lower churn rates?

# %%
df2.groupby(['CouponUsed' , 'Churn'])[['CustomerID']].count()

# %%
# Group and count by 'Coup' and 'Churn'
coupoun_churnrate = pd.DataFrame(df2.groupby('CouponUsed')['Churn'].value_counts())
coupoun_churnrate = coupoun_churnrate.rename(columns={'Churn': 'Count'})
coupoun_churnrate = coupoun_churnrate.reset_index()


fig = px.bar(coupoun_churnrate, x='CouponUsed', y = 'Count',color='Churn', barmode='group',color_discrete_sequence=['rgba(58, 71, 80, 0.6)' ,'rgba(246, 78, 139, 1.0)'],title="<b>"+'CouponUsed Vs Churn Rate',text_auto= True)
fig.update_layout(hovermode='x',title_font_size=30)
fig.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
xaxis_title='CouponUsed',
yaxis_title='count',
)
fig.show()

# %% [markdown]
# Churn decreases as a function of coupon usage, according to Graph.

# %% [markdown]
# 18-Is there a connection between satisfaction score and number of orders in the past month?

# %%
df2.groupby('SatisfactionScore')[['OrderCount']].count()

# %%
fig = px.box(df2, y="OrderCount", x='SatisfactionScore', color="SatisfactionScore", title="<b>"+'SatisfactionScore Vs OrderCount',
             boxmode="overlay", points='all')
fig.update_layout(hovermode='x',title_font_size=30)
fig.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
xaxis_title='SatisfactionScore',
yaxis_title='OrderCount',
)
fig.show()

# %% [markdown]
# StatisfactionScore doesn't have affect on OrderCount.

# %% [markdown]
# 19-There is relation between CashbackAmount and order counts within churn?

# %%
df_c.groupby(['OrderCount','CashbackAmount'])[['Churn']].count()

# %%
fig = px.histogram(df2, x='CashbackAmount', y='OrderCount' ,color = 'Churn', title="<b>"+'CashbackAmount Vs OrderCount within churn', color_discrete_sequence=["#d62728", "#1f77b4"])

# Customize the plot
fig.update_layout(hovermode='x',title_font_size=30)
fig.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
xaxis_title='CashbackAmount',
yaxis_title='OrderCount',
)
fig.show()

# %% [markdown]
# Amount of cash rebate has a positive correlation with churn rate, but no correlation with order count.

# %% [markdown]
# 20-Are customers who complained more likely to churn?

# %%
df2.groupby('Complain')[['Churn']].count()

# %%
comp_churn = pd.DataFrame(df2.groupby('Complain')['Churn'].value_counts())
comp_churn = comp_churn.rename(columns={'Churn': 'Count'})
comp_churn = comp_churn.reset_index()
print(comp_churn)

comp_churn['Complain'].replace('0' , 'No Complain' , inplace = True)
comp_churn['Complain'].replace('1' , 'Complain' , inplace = True)
comp_churn['Churn'].replace('0' , 'No Churn' , inplace = True)
comp_churn['Churn'].replace('1' , 'Churn' , inplace = True)
print(comp_churn)

# Tree map
fig = px.treemap(comp_churn, path=[px.Constant("all"), 'Complain', 'Churn'], values='Count' , color_discrete_sequence=["#2F4B26" , '#FF0000'],title="<b>"+'Complain Vs Churn')
fig.update_traces(textinfo="label+percent parent+value" ,root_color="lightgrey")
fig.update_layout(margin = dict(t=70, l=25, r=25, b=25))

# red_palette = ['#410B13', '#CD5D67', '#BA1F33', '#421820', '#91171F']
# Customize the plot
fig.update_layout(hovermode='x',title_font_size=30)
fig.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
)
fig.show()

# %% [markdown]
# All Insights from EDA.
# 
# There is not much of a distinction between the sexes: normalised order. It appears that men are more likely to cancel their subscriptions to the app than women are; 63.3% of the app's former users were men. This suggests that the company would benefit from focusing more on things that appeal to men.
# 
# Since single users are more likely to abandon the app, the company may want to focus on satisfying married clients while still catering to the needs of its many single ones.
# 
# The highest tenancy rate is in CityTier 2, however this does not appear to be a significant impact.
# 
# Despite having the highest order average, CityTier 3 appears to have no effect on client retention.
# 
# Users who gave the app a lower satisfaction score used it less frequently than those who gave it a higher one, but I don't think there's any correlation between the two.
# 
# Tier 1 cities have the highest average app usage time.
# 
# City tier and address count are inversely related. This relationship suggests that address density and type of locality (metro vs. smaller cities vs. towns) impact how many addresses customers have across city types, with customers in larger cities (CityTier 1) having more addresses on average than those in smaller cities and towns in lower tiers.
# 
# The amount of days since a customer's last order is weakly correlated with the frequency with which they complain.
# 
# There may be an issue with the mobile app's user experience if users are leaving in large numbers.
# 
# Since the distance from the warehouse to the residence is same throughout all city tiers, this suggests that the corporation also constructed warehouses in the lower city tiers.
# 
# All socioeconomic strata in the city favour the categories of laptops and accessories and mobile phones.
# 
# Typical method of exchange for CityTier '1' ==> CityTier '2's Preferred Payment Method is a Debit Card, while CityTier '3's Preferred Payment Method is a UPI, and so on. digital currency wallet
# Debit card use is quite widespread across all income brackets.
# 
# More than any other CityTier, CityTier '1' has 10298 orders pending fulfilment.
# The largest mean order count is found in CityTier 3, indicating that this city has a large number of orders despite its relatively low population.
# 
# OrderAmountHikeFromlastYear has a positive effect on Churn rate, and we should prioritise customers with percentages between 12 and 15 percent while doing so.
# 
# There is one customer whose DaySinceLastOrder is significantly greater than the average, but he or she is an anomaly who can be eliminated by focusing on the remaining customers.
# 
# OrderCount 2 represents 67% of the population with the highest percentage of time spent in the app.
# 
# Males' Two Favourite Groups Are: [Others] and [Mobile Phones].
# The two most popular categories among female shoppers are [Grocery, Fashion].
# 
# When more coupons are used, customer turnover decreases.
# 
# The quantity of orders is unaffected by the satisfaction rating.
# 
# There is a positive correlation between cash back amount and turnover rate, but no correlation between cash back amount and order count.
# 
# Complaints have no effect on customer retention; 68% of those who complain do not leave.

# %% [markdown]
# Data Preprocessing

# %%
# Handling missing values
round((df.isnull().sum()*100 / df.shape[0]),2)

# %%
msno.matrix(df)

# %%
msno.bar(df , color="tab:green")

# %%
sns.kdeplot(df , x='Tenure')

# %%
df['Tenure'] = df['Tenure'].fillna(method = 'bfill')

# %%
sns.kdeplot(df , x='Tenure')

# %%
df['Tenure'].isnull().sum()

# %%
sns.kdeplot(df , x='WarehouseToHome')

# %%
from sklearn.impute import SimpleImputer
s_imp = SimpleImputer(missing_values=np.nan , strategy = 'most_frequent')
df['WarehouseToHome'] = s_imp.fit_transform(pd.DataFrame(df['WarehouseToHome']))

# %%
sns.kdeplot(df , x='WarehouseToHome')

# %%
sns.kdeplot(df , x='HourSpendOnApp')

# %%
fill_list = df['HourSpendOnApp'].dropna()
df['HourSpendOnApp'] = df['HourSpendOnApp'].fillna(pd.Series(np.random.choice(fill_list , size = len(df['HourSpendOnApp'].index))))

# %%
sns.kdeplot(df , x='HourSpendOnApp')

# %%
sns.kdeplot(df , x='OrderAmountHikeFromlastYear')

# %%
df['OrderAmountHikeFromlastYear'] = df['OrderAmountHikeFromlastYear'].fillna(method = 'ffill')

# %%
sns.kdeplot(df , x='OrderAmountHikeFromlastYear')

# %%
sns.kdeplot(df , x='CouponUsed')

# %%
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2)
df['CouponUsed']=imputer.fit_transform(df[['CouponUsed']])

# %%
sns.kdeplot(df , x='CouponUsed')

# %%
sns.kdeplot(df , x='OrderCount')

# %%
imputer_2 = KNNImputer(n_neighbors=2)
df['OrderCount']=imputer_2.fit_transform(df[['OrderCount']])

# %%
sns.kdeplot(df , x='OrderCount')

# %%
sns.kdeplot(df , x='DaySinceLastOrder')

# %%
df['DaySinceLastOrder'] = df['DaySinceLastOrder'].fillna(method = 'bfill')

# %%
sns.kdeplot(df , x='DaySinceLastOrder')

# %%
# After I Checked the data the Customer ID Column not important for our Models so I drop it
df.drop('CustomerID' , axis = 1 , inplace = True)

# %%
df.shape

# %% [markdown]
# Encoding

# %%
for i in df.columns:
    if df[i].dtype == 'object':
        print(df[i].value_counts())
        print('*' * 40)

# %%
data = df[df.select_dtypes(exclude=np.number).columns]
data

# %%
le = LabelEncoder()

# %%
# Encode for cat_cols
for i in df.columns:
  if df[i].dtype == 'object':
    df[i] = le.fit_transform(df[i])

df.head(4)

# %%
for i in data.columns:
    data[i] = le.fit_transform(data[i])

data.head(4)

# %% [markdown]
# Handling Outliers

# %%
df.dtypes

# %%
fig = plt.figure(figsize=(12,18))
for i in range(len(df.columns)):
    fig.add_subplot(9,4,i+1)
    sns.boxplot(y=df.iloc[:,i])

plt.tight_layout()
plt.show()

# %%
# detecting True Outliers
def handle_outliers(df , column_name):
  Q1 = df[column_name].quantile(0.25)
  Q3 = df[column_name].quantile(0.75)
  IQR = Q3 - Q1

  # Define Upper and lower boundaries
  Upper = Q3 + IQR * 1.5
  lower = Q1 - IQR * 1.5

  # lets make filter for col values
  new_df = df[ (df[column_name] > lower) & (df[column_name] < Upper) ]

  return new_df

# %%
df.columns

# %%
# lets Give our Functions columns contains outlier
cols_outliers = ['Tenure' , 'WarehouseToHome' , 'NumberOfAddress' , 'DaySinceLastOrder' , 'HourSpendOnApp' , 'NumberOfDeviceRegistered']

for col in cols_outliers:
    df = handle_outliers(df , col)

df.head(4)

# %%
fig = plt.figure(figsize=(12,18))
for i in range(len(df.columns)):
    fig.add_subplot(9,4,i+1)
    sns.boxplot(y=df.iloc[:,i])

plt.tight_layout()
plt.show()

# %% [markdown]
# I performed Trim on columns containing outliers, but when I checked, I discovered that a great deal of data had been lost.

# %%
corr_matrix = df.corr()
corr_matrix

# %%
plt.figure(figsize = (18,15))
sns.heatmap(df.corr() , annot = True , cmap = 'Blues')

# %%
churn_corr_vector = corr_matrix['Churn'].sort_values(ascending = False)
churn_corr_vector

# %%
plt.figure(figsize = (10,10))
sns.barplot(x = churn_corr_vector , y = churn_corr_vector.index , palette = 'coolwarm')
plt.title('Relation Between Features and target')

# %%
fig = px.histogram(df2, x="Churn", color="Churn" ,text_auto= True , title="<b>"+'Check Imbalance' , color_discrete_sequence = ['#BA1F33','#3A506B'])

# Customize the plot
fig.update_layout(hovermode='x',title_font_size=30)
fig.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
xaxis_title='Churn',
yaxis_title='count',
)
fig.show()

# %% [markdown]
# Handling Imbalancd Data

# %%
X = df.drop('Churn' , axis = 1)
Y = df['Churn']

# %%
from imblearn.combine import SMOTETomek

# %%
smt = SMOTETomek(random_state=42)
x_over , y_over = smt.fit_resample(X , Y)

# %%
x_over.shape, y_over.shape

# %% [markdown]
# Split Data

# %%
x_train , x_test , y_train , y_test = train_test_split(x_over , y_over , test_size = 0.30 , random_state = 42)

# %%
# Now I will make normalization for all data to make them in commom range
from sklearn.preprocessing import MinMaxScaler , StandardScaler , RobustScaler

MN = MinMaxScaler()
# SC = StandardScaler()
# Rb = RobustScaler()
x_train_scaled = MN.fit_transform(x_train)
x_test_scaled = MN.fit_transform(x_test)

# %% [markdown]
# Modelling

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
import warnings

warnings.filterwarnings("ignore")

# %%
logisreg_clf = LogisticRegression()
svm_clf = SVC()
dt_clf = DecisionTreeClassifier()
rf_clf = RandomForestClassifier()
XGB_clf = XGBClassifier()
ada_clf = AdaBoostClassifier()

# %%
clf_list = [logisreg_clf, svm_clf, dt_clf, rf_clf, XGB_clf, ada_clf]
clf_name_list = ['Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'Random Forest', 'XGBClassifier' , 'AdaBoostClassifier']

for clf in clf_list:
    clf.fit(x_train_scaled,y_train)

# %%
train_acc_list = []
test_acc_list = []

for clf,name in zip(clf_list,clf_name_list):
    y_pred_train = clf.predict(x_train_scaled)
    y_pred_test = clf.predict(x_test_scaled)
    print(f'Using model: {name}')
    print(f'Trainning Score: {clf.score(x_train_scaled, y_train)}')
    print(f'Test Score: {clf.score(x_test_scaled, y_test)}')
    print(f'Acc Train: {accuracy_score(y_train, y_pred_train)}')
    print(f'Acc Test: {accuracy_score(y_test, y_pred_test)}')
    train_acc_list.append(accuracy_score(y_train, y_pred_train))
    test_acc_list.append(accuracy_score(y_test, y_pred_test))
    print(' ' * 60)
    print('*' * 60)
    print(' ' * 60)

# %%
all_models = pd.DataFrame({'Train_Accuarcy': train_acc_list , 'Test_Accuarcy' : test_acc_list}  , index = clf_name_list)
all_models

# %%
# Models vs Train Accuracies
fig = px.bar(all_models, x=all_models['Train_Accuarcy'], y = all_models.index ,color=all_models['Train_Accuarcy'],title="<b>"+'Models Vs Train Accuracies',text_auto= True , color_continuous_scale='Reds')
fig.update_layout(hovermode='x',title_font_size=30)
fig.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
xaxis_title='Train Sccracy',
yaxis_title='Models Names',
)
fig.show()


# Models vs Test Accuracies
fig = px.bar(all_models, x=all_models['Test_Accuarcy'], y = all_models.index ,color=all_models['Test_Accuarcy'],title="<b>"+'Models Vs Test Accuracies',text_auto= True , color_continuous_scale='Reds')
fig.update_layout(hovermode='x',title_font_size=30)
fig.update_layout(
title_font_color="black",
template="plotly",
title_font_size=30,
hoverlabel_font_size=20,
title_x=0.5,
xaxis_title='Test Accuarcy',
yaxis_title='Models Names',
)
fig.show()

# %%
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, RocCurveDisplay

# %%
model= LogisticRegression()
model.fit(x_train_scaled,y_train)
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
roc_auc1 = roc_auc_score(y_test, y_pred)
print("Accuracy = {}".format(accuracy))
print("ROC Area under Curve = {}".format(roc_auc1))
print(classification_report(y_test,y_pred,digits=5))
plot_confusion_matrix(confusion_matrix(y_test , y_pred))
print('*' * 70)
RocCurveDisplay.from_estimator(model , x_test_scaled , y_test)

# %%
model=SVC()
model.fit(x_train_scaled,y_train)
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
roc_auc2 = roc_auc_score(y_test, y_pred)
print("Accuracy = {}".format(accuracy))
print("ROC Area under Curve = {}".format(roc_auc2))
print(classification_report(y_test,y_pred,digits=5))
plot_confusion_matrix(confusion_matrix(y_test , y_pred))
RocCurveDisplay.from_estimator(model , x_test_scaled , y_test)

# %%
model=DecisionTreeClassifier()
model.fit(x_train_scaled,y_train)
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
roc_auc3 = roc_auc_score(y_test, y_pred)
print("Accuracy = {}".format(accuracy))
print("ROC Area under Curve = {}".format(roc_auc3))
print(classification_report(y_test,y_pred,digits=5))
plot_confusion_matrix(confusion_matrix(y_test , y_pred))
RocCurveDisplay.from_estimator(model , x_test_scaled , y_test)

# %%
model=RandomForestClassifier()
model.fit(x_train_scaled,y_train)
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
roc_auc4 = roc_auc_score(y_test, y_pred)
print("Accuracy = {}".format(accuracy))
print("ROC Area under Curve = {}".format(roc_auc4))
print(classification_report(y_test,y_pred,digits=5))
plot_confusion_matrix(confusion_matrix(y_test , y_pred))
RocCurveDisplay.from_estimator(model , x_test_scaled , y_test)

# %%
model=XGBClassifier()
model.fit(x_train_scaled,y_train)
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
roc_auc5 = roc_auc_score(y_test, y_pred)
print("Accuracy = {}".format(accuracy))
print("ROC Area under Curve = {}".format(roc_auc5))
print(classification_report(y_test,y_pred,digits=5))
plot_confusion_matrix(confusion_matrix(y_test , y_pred))
RocCurveDisplay.from_estimator(model , x_test_scaled , y_test)

# %%
model=AdaBoostClassifier()
model.fit(x_train_scaled,y_train)
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
roc_auc6 = roc_auc_score(y_test, y_pred)
print("Accuracy = {}".format(accuracy))
print("ROC Area under Curve = {}".format(roc_auc6))
print(classification_report(y_test,y_pred,digits=5))
plot_confusion_matrix(confusion_matrix(y_test , y_pred))
RocCurveDisplay.from_estimator(model , x_test_scaled , y_test)