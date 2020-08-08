# %% Read csv file
import pandas as pd

df = pd.read_csv("2012.csv")

# %% Rows and columns in dataframe

df.shape

# %% Describe the data

df.describe(include="all")

# %% Checkout out dataframe info

df.info()

# %% Datatypes of columns

df.dtypes

# %% Make sure numeric columns have numbers, remove rows that are not
cols = [
    "perobs",
    "perstop",
    "age",
    "weight",
    "ht_feet",
    "ht_inch",
    "datestop",
    "timestop",
    "xcoord",
    "ycoord",
]

for col in cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()

# %% Total time spent on observation before taking any action and total time of stop for suspect

total_time_of_obs = df["perobs"].sum()
total_time_of_stop = df["perstop"].sum()

print(total_time_of_obs)
print(total_time_of_stop)

# %% Man hours spent on this program
# Assumption: Average total number of officers in New York City:36000
#             Officer works for 5 days a week for 8 hours a day

working_days = 366-(52*2)
total_amount_of_man_hours_NYPD_per_year = 36000 * working_days * 8

total_time_obs_stop_per_year = (total_time_of_obs + total_time_of_stop)/60
hours_spent_obs_stop_per_year = (total_time_obs_stop_per_year)/36000
percent_of_man_hour_spent_on_obs_stop = ((total_time_obs_stop_per_year)/total_amount_of_man_hours_NYPD_per_year)*100
print(f"{percent_of_man_hour_spent_on_obs_stop :.2f} %")


# %% make_datetime column by using datestop and timestop columns

df["datestop"] = df["datestop"].astype(str).str.zfill(8)
df["timestop"] = df["timestop"].astype(str).str.zfill(4)

from datetime import datetime

def make_datetime(datestop, timestop):
    year = int(datestop[-4:])
    month = int(datestop[:2])
    day = int(datestop[2:4])

    hour = int(timestop[:2])
    minute = int(timestop[2:])

    return datetime(year, month, day, hour, minute)


df["datetime"] = df.apply(
    lambda r: make_datetime(r["datestop"], r["timestop"]),
    axis=1
)


# %% convert all value to label in the dataframe, remove rows that cannot be mapped
import numpy as np
from tqdm import tqdm

value_label = pd.read_excel(
    "2012 SQF File Spec.xlsx",
    sheet_name="Value Labels",
    skiprows=range(4)
)
value_label["Field Name"] = value_label["Field Name"].fillna(
    method="ffill"
)
value_label["Field Name"] = value_label["Field Name"].str.lower()
value_label["Value"] = value_label["Value"].fillna(" ")
value_label = value_label.groupby("Field Name").apply(
    lambda x: dict([(row["Value"], row["Label"]) for row in x.to_dict("records")])
)

cols = [col for col in df.columns if col in value_label]

for col in tqdm(cols):
    df[col] = df[col].apply(
        lambda val: value_label[col].get(val, np.nan)
    )

df["trhsloc"] = df["trhsloc"].fillna("Neither")
df = df.dropna()


# %% convert xcoord and ycoord to (lon, lat)
import pyproj

srs = "+proj=lcc +lat_1=41.03333333333333 +lat_2=40.66666666666666 +lat_0=40.16666666666666 +lon_0=-74 +x_0=300000.0000000001 +y_0=0 +ellps=GRS80 +datum=NAD83 +to_meter=0.3048006096012192 +no_defs"
p = pyproj.Proj(srs)

df["coord"] = df.apply(
    lambda r: p(r["xcoord"], r["ycoord"], inverse=True), axis=1
)

# %% convert height in feet/inch to cm
df["height"] = (df["ht_feet"] * 12 + df["ht_inch"]) * 2.54


# %% remove outlier
df = df[(df["age"] <= 100) & (df["age"] >= 10)]
df = df[(df["weight"] <= 350) & (df["weight"] >= 50)]


# %% delete columns that are not needed
df = df.drop(
    columns=[
        # processed columns
        "datestop",
        "timestop",
        "ht_feet",
        "ht_inch",
        "xcoord",
        "ycoord",

        # not useful
        "year",
        "recstat",
        "crimsusp",
        "dob",
        "ser_num",
        "arstoffn",
        "sumoffen",
        "compyear",
        "comppct",
        "othfeatr",
        "adtlrept",
        "dettypcm",
        "linecm",
        "repcmd",
        "revcmd",

        # location of stop
        # only use coord and city
        "addrtyp",
        "rescode",
        "premtype",
        "premname",
        "addrnum",
        "stname",
        "stinter",
        "crossst",
        "aptnum",
        "state",
        "zip",
        "addrpct",
        "sector",
        "beat",
        "post",
    ]
)

# %% Number of instances w.r.t month

import folium
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(df["datetime"].dt.month)
plt.title("Number of instances w.r.t month")
plt.xlabel("Month")
plt.ylabel("Number of instances")

# %% Number of instances w.r.t weekday

ax = sns.countplot(df["datetime"].dt.weekday)
# The day of the week with Monday=0, Sunday=6. See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DatetimeIndex.weekday.html
ax.set_xticklabels(
    ["Mon", "Tue", "Wed", "Thur", "Fri", "Sat", "Sun"]
)
plt.title("Number of instances w.r.t weekday")
plt.xlabel("Day of Week")
plt.ylabel("Number of instances")


# %% Number of instances w.r.t hour

sns.countplot(df["datetime"].dt.hour)
plt.title("Number of instances w.r.t hour")
plt.xlabel("Hour")
plt.ylabel("Number of instances")

# %% Number of instances w.r.t race
ax = sns.countplot(data=df, x="race")
labels = ['BLACK',
        'WHITE-HISP',
        'WHITE',
        'BLACK-HISP',
        'ASIAN/PACI',
        'OTHER',
        'AMER IND/ALASK']
ax.set_xticklabels(
    labels, rotation=45, fontsize=10
)
for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)

plt.title("Number of instances w.r.t race")
plt.xlabel("Race")
plt.ylabel("Number of instances")

# %% Number of instances w.r.t race and sex

ax = sns.countplot(y="race", hue=df["sex"],data=df)

plt.title("Number of instances w.r.t sex and race")
plt.ylabel("Sex")
plt.xlabel("Number of instances")


# %% Number of instances w.r.t race and city

ax = sns.countplot("race", hue=df["city"],data=df)
labels = ['BLACK',
        'WHITE-HISP',
        'WHITE',
        'BLACK-HISP',
        'ASIAN/PACI',
        'OTHER',
        'AMER IND/ALASK']

ax.set_xticklabels(labels,rotation=45,fontsize=10)
plt.title(label='Incident count as per race in different cities', loc='center')
plt.ylabel("Number of instances")
plt.xlabel("Race")

# %% Age of Suspects

sns.distplot(df["age"],kde=False,bins=10)

plt.title(label='Age of Suspects', loc='center')
plt.xlabel("Age")
plt.ylabel("Count")

# %% Stops vs city

ax = sns.barplot(x="city", y="perstop",data=df)

plt.title(label='Stops vs City', loc='center')
plt.xlabel("City")
plt.ylabel("Period of stop")


# %% Relationships between attributes

df.corr()
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='Blues')

# %%
nyc = (40.730610, -73.935242)

m = folium.Map(location=nyc)

# %%
for coord in df.loc[df["detailcm"]=="BURGLARY", "coord"]:
    folium.CircleMarker(
        location=(coord[1], coord[0]), radius=1, color="red"
    ).add_to(m)

m


# %%
for coord in df.loc[df["detailcm"]=="KIDNAPPING", "coord"]:
    folium.CircleMarker(
        location=(coord[1], coord[0]), radius=1, color="yellow"
    ).add_to(m)

m

# %% Force used by officers

ax = sns.countplot(data=df, x="pf_hands")

for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)

# %% Reasons for Stop, Question and Frisk

ax = sns.countplot(data=df, x="cs_furtv")

for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)

# %% Comparison between force used and reasons for SQF

ax = sns.countplot("cs_furtv", hue=df["pf_hands"],data=df)

for p in ax.patches:
    percentage = p.get_height() * 100 / df.shape[0]
    txt = f"{percentage:.1f}%"
    x_loc = p.get_x()
    y_loc = p.get_y() + p.get_height()
    ax.text(x_loc, y_loc, txt)

plt.title("Number of instances w.r.t reason for SQF and force used")
plt.xlabel("Reason:Furtive Movement")
plt.ylabel("Number of instances")


# %%
df.to_pickle("sqf.pkl")

