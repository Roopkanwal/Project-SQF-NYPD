# %% read dataframe from part1
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_pickle("sqf.pkl")

# %% select some yes/no columns to convert into a dataframe of boolean values

pfs = [col for col in df.columns if col.startswith("pf_")]   # force

rfs = [col for col in df.columns if col.startswith("rf_")]   #frisked

armed = [
    "contrabn",
    "pistol",
    "riflshot",
    "asltweap",
    "knifcuti",
    "machgun",
    "othrweap",
]

x = df[ pfs +  rfs + armed ]
x = x == "YES"


# %% select some categorical columns and do one hot encoding

for val in df["race"].unique():
    x[f"race_{val}"] = df["race"] == val


for val in df["city"].value_counts().index:
    x[f"city_{val}"] = df["city"] == val


for val in df["detailcm"].value_counts().index:
    x[f"crime_{val}"] = df["detailcm"] == val



# %% apply frequent itemsets mining

frequent_itemsets = apriori(x, min_support=0.01, use_colnames=True)

# %% apply association rules mining

rules = association_rules(frequent_itemsets, min_threshold=0.6)
rules

#%% %% sort according to confidence,lift,support

rules.sort_values(["confidence"], ascending=False).head(10)

# %% sort rules by confidence and select rules within "crime_CPW" in it
rules.sort_values(["confidence","lift","support"], ascending=False)[
    rules.apply(
        lambda r: "crime_CPW" in r["antecedents"]
        or "crime_CPW" in r["consequents"],
        axis=1,
    )
]
# %% scatterplot lift vs confidence

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=rules["lift"], y=rules["confidence"], alpha=0.5)
plt.xlabel("Lift")
plt.ylabel("Confidence")
plt.title("Lift vs Confidence")


# %% scatterplot support vs lift

sns.scatterplot(x=rules["support"], y=rules["lift"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("Lift")
plt.title("Support vs Lift")