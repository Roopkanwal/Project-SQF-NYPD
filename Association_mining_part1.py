# %% read dataframe from part1
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

df = pd.read_pickle("sqf.pkl")

# %% Convert dataset in transactional format

dataset = df.groupby(["datetime"]).apply(lambda f: f["detailcm"].tolist())

# %% Apply TransactionEncoder to tranform dataset into a one-hot encoded boolean array

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)

# %% apply frequent itemsets mining

frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

# %% apply association rules mining

rules = association_rules(frequent_itemsets, min_threshold=0.6)
rules

#%% %% sort according to confidence,lift,support

rules.sort_values(["confidence","lift","support"], ascending=False).head(10)


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

# %%
