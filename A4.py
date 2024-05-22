#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import seaborn as sns
f1='london_merges.csv'
df=pd.read_csv(f1)
print(df.info())


# In[5]:


import pandas as pd
import seaborn as sns
f2='bk.csv'
df=pd.read_csv(f2)
print(df.info())


# In[18]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("ReadCSV").getOrCreate()
f1 = "london_merged.csv"
df = spark.read.csv(f1, header=True, inferSchema=True)
df.printSchema()
df.show()


# In[7]:


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("ReadCSV").getOrCreate()
f2 = "bk.csv"
df2 = spark.read.csv(f2, header=True, inferSchema=True)
df2.printSchema()
df2.show()


# In[3]:


from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DatasetAnalysis").getOrCreate()

f1 = "london_merges.csv"
df = spark.read.csv(f1, header=True, inferSchema=True)

df.show()

columns = df.columns
print("column name:", columns)

num_rows = df.count()
num_columns = len(columns)
print("row count:", num_rows)
print("column count:", num_columns)

df.printSchema()

df.describe().show()

for i in columns:
    print(f"column '{i}' unique values and count:")
    df.groupBy(i).count().show()


# In[4]:


from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("DatasetAnalysis").getOrCreate()

f2 = "bk.csv"
df1 = spark.read.csv(f2, header=True, inferSchema=True)

df1.show()

columns = df1.columns
print("column name:", columns)

num_row = df1.count()
num_column = len(columns)
print("row count:", num_rows)
print("column count:", num_columns)

df1.printSchema()

df1.describe().show()

for i in columns:
    print(f"column '{i}' unique values and count:")
    df1.groupBy(i).count().show()


# In[32]:


f1 = "london_merges.csv"
df = spark.read.csv(f1, header=True, inferSchema=True)
df.describe()


# In[10]:


f2 = "bk.csv"
df1 = spark.read.csv(f2, header=True, inferSchema=True)
df1.describe()


# In[18]:


import seaborn as sns
import matplotlib.pyplot as plt

# Convert the 'count' column of the PySpark DataFrame to a Pandas Series
count_series = df.select('cnt').toPandas()['cnt']

# Plot the distribution
sns.distplot(count_series)
plt.show()


# In[16]:


from pyspark.sql.functions import col, count, when

# Function to check for missing and duplicate values
def check_data_issues(df, df_name):
    # Check for missing values
    missing_values = df.select([count(when(col(c).isNull(), c)).alias(c) for c in df.columns])
    print(f'Missing values in {df_name}:')
    missing_values.show()

    # Check for duplicate values
    duplicates = df.groupBy(df.columns).agg(count('*').alias('count_occurrences')).filter('count_occurrences > 1')
    print(f'Duplicate values in {df_name}:')
    duplicates.show()

# Check issues for df
check_data_issues(df, "london_merges.csv")

# Check issues for df2
check_data_issues(df1, 'bk.csv')


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt

# Convert the necessary columns of the PySpark DataFrame to a Pandas DataFrame in one go
pdf = df.select('t1', 't2', 'hum', 'wind_speed').toPandas()

# Setup the figure and axes
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot the distributions using seaborn
sns.histplot(pdf['t1'], kde=True, ax=axes[0, 0])
sns.histplot(pdf['t2'], kde=True, ax=axes[0, 1])
sns.histplot(pdf['hum'], kde=True, ax=axes[1, 0])
sns.histplot(pdf['wind_speed'], kde=True, ax=axes[1, 1])

# Set titles and labels for each subplot
axes[0, 0].set(title='Distribution of t1', xlabel='t1')
axes[0, 1].set(title='Distribution of t2', xlabel='t2')
axes[1, 0].set(title='Distribution of Humidity', xlabel='Humidity')
axes[1, 1].set(title='Distribution of Wind Speed', xlabel='Wind Speed')

plt.tight_layout()
plt.show()


# In[20]:


import seaborn as sns
import matplotlib.pyplot as plt

# Directly convert the PySpark DataFrame to a Pandas DataFrame with the selected columns
df_pandas = df.select('t1', 't2', 'hum', 'wind_speed', 'cnt').toPandas()

# Use seaborn's pairplot to visualize pairwise relationships in the dataset
sns.pairplot(df_pandas)
plt.show()  # Ensure the plot is displayed


# In[34]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
# Update the path to your file location
data = pd.read_csv(f1)

# Calculate the frequency of each 'weather_code' before replacing the outlier
weather_counts_before = data['weather_code'].value_counts()

# Replace the outlier '26' with the mode '1'
data['weather_code'].replace(26, 1, inplace=True)

# Calculate the frequency of each 'weather_code' after replacing the outlier
weather_counts_after = data['weather_code'].value_counts()

# Set up the figure for two subplots (before and after)
fig, axs = plt.subplots(1, 2, figsize=(14, 7), sharey=True)

# Convert series to dataframe for plotting
weather_counts_before_df = weather_counts_before.reset_index()
weather_counts_before_df.columns = ['Weather Code', 'Frequency']
weather_counts_after_df = weather_counts_after.reset_index()
weather_counts_after_df.columns = ['Weather Code', 'Frequency']

# Plot before replacement
sns.barplot(x='Weather Code', y='Frequency', data=weather_counts_before_df, ax=axs[0])
axs[0].set_title('Weather Code Distribution Before Replacement')
axs[0].set_xlabel('Weather Code')
axs[0].set_ylabel('Frequency')

# Plot after replacement
sns.barplot(x='Weather Code', y='Frequency', data=weather_counts_after_df, ax=axs[1])
axs[1].set_title('Weather Code Distribution After Replacement')
axs[1].set_xlabel('Weather Code')

plt.show()



# In[36]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame and it's already loaded

data = pd.read_csv(f1)

# Calculate mean and standard deviation of 'cnt'
mean_cnt = data['cnt'].mean()
standard_cnt = data['cnt'].std()

# Filter out outliers
data_outliers = data[(data['cnt'] - mean_cnt).abs() < 3 * standard_cnt]

# Set up the figure for three subplots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Distribution plot for 'cnt' on the first and second axis
sns.histplot(data_outliers['cnt'], ax=ax1, kde=True, bins=30)
ax1.set_title('Distribution of Count without Outliers')
ax1.set_xticks(range(0, 1200, 200))

sns.histplot(data_outliers['cnt'], ax=ax2, kde=True, bins=30)
ax2.set_title('Distribution of Count without Outliers')
ax2.set_xticks(range(0, 1200, 200))

# Log-transformed distribution plot on the third axis
y_axis_log = np.log(data_outliers['cnt'] + 1)  # Use log(y+1) to handle zero counts
sns.histplot(y_axis_log, ax=ax3, kde=True, bins=30)
ax3.set_title('Log Transformed Distribution of Count')

plt.show()


# In[38]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset

data = pd.read_csv(f1)

# Ensure the data is appropriate for correlation analysis (numeric columns)
numeric_data = data.select_dtypes(include=[np.number])

# Compute the correlation matrix for 'cnt'
corr_matrix = numeric_data.corr()["cnt"]
corr_matrix = corr_matrix.drop("cnt", axis=0).sort_values(ascending=False)

# Plotting the correlation
plt.figure(figsize=(10,5))
sns.set(font_scale=0.8)
sns.barplot(x=corr_matrix.index, y=corr_matrix)
plt.xticks(rotation=90)
plt.ylim(-0.02, 0.05)  # Setting this range can highlight small variations
plt.title("Relationship of variables with cnt", fontsize=15)
plt.show()


# In[45]:


import pandas as pd

# Assuming 'f1' is your DataFrame and it's already loaded  # Update the path to your file location

data = pd.read_csv(f1)
# Calculate mean and standard deviation of 'cnt'
mean_cnt = data['cnt'].mean()
standard_cnt = data['cnt'].std()

# Filter out outliers
f1_outliers = data[abs(data['cnt'] - mean_cnt) < 3 * standard_cnt]
count = f1_outliers.count()
print(count)


# In[48]:


import pandas as pd

# Assuming 'f1' is your DataFrame and it's already loaded  # Update the path to your file location

data = pd.read_csv(f1)
len(data[(data['t1']-data['t2'])>10])
len(data[data['wind_speed']==0])
data=data[data['wind_speed']!=0]


# In[50]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, stddev

# Initialize Spark Session
spark = SparkSession.builder.appName("Outlier Removal").getOrCreate()

# Load the data into a DataFrame
df = spark.read.csv('london_merges.csv', header=True, inferSchema=True)

# Calculate mean and standard deviation of 'cnt'
mean_val = df.agg(mean(df['cnt'])).collect()[0][0]
std_val = df.agg(stddev(df['cnt'])).collect()[0][0]

# Filter out outliers using the calculated mean and standard deviation
df_without_outliers = df.filter((df['cnt'] - mean_val).between(-3 * std_val, 3 * std_val))

# Show the result to verify
df_without_outliers.show()

# Stop the Spark session
spark.stop()


# In[11]:


from datetime import datetime
# Convert the PySpark DataFrame to a Pandas DataFrame
df_pd = df.toPandas()

# Now concatenate the two Pandas DataFrames
Bike_data = pd.concat([f1_outliers, df_pd], ignore_index=True)

# Continue with the datetime split operation as before
def split_datetime(row):
    date_parts = row.split()
    date = date_parts[0]
    time = date_parts[1]
    year, month, _ = date.split('/')
    hour = int(time.split(':')[0])
    weekday = datetime.strptime(date, '%d/%m/%Y').isoweekday()
    return date, hour, int(year), int(month), weekday

Bike_data[['date', 'hour', 'year', 'month', 'weekday']] = Bike_data['datetime'].apply(split_datetime).apply(pd.Series)   


# In[16]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file

data = pd.read_csv(f1)

# Set the size of the figure
plt.figure(figsize=(16, 10))

# Generate a heatmap of the correlation matrix
sns.heatmap(data.corr(), annot=True)

# Show the plot
plt.show()


# In[ ]:




