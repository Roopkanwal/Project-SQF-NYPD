# %% read dataframe from part1
import pandas as pd

df = pd.read_pickle("sqf.pkl")


# %% select some yes/no columns and convert into booleans
pfs = [col for col in df.columns if col.startswith("pf_")]
css = [col for col in df.columns if col.startswith("cs_")]
armed = [
    "contrabn",
    "pistol",
    "riflshot",
    "asltweap",
    "knifcuti",
    "machgun",
    "othrweap",
]

x = df[pfs + css + armed]
x = x == "YES"

# create label for the dataset
y = df["arstmade"]
y = y == "YES"

# %% grab some number and category features
num_cols = ["age", "height", "weight"]
cat_cols = ["race", "city", "build"]

x[num_cols] = df[num_cols]
x[cat_cols] = df[cat_cols]


# %% split into training / testing sets
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2020)


# %% massage the categorical columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

enc = OneHotEncoder(handle_unknown="ignore")
ct = ColumnTransformer([("ohe", enc, cat_cols),], remainder="passthrough")

x_train = ct.fit_transform(x_train)
x_test = ct.transform(x_test)

# %% Function for printing metric scores for classifer

def metric_results(clf,y_train,pred_train,y_test,pred_test):
    print(clf.__class__.__name__)
    print("Training Result:")
    print(f"Accuracy Score:  {accuracy_score(y_train, pred_train):.3f}")
    print(f"Precision Score: {precision_score(y_train, pred_train):.3f}")
    print(f"Recall Score:    {recall_score(y_train, pred_train):.3f}")
    print(f"F1 Score:        {f1_score(y_train, pred_train):.3f}")
    print("Testing Result:")
    print(f"Accuracy Score:  {accuracy_score(y_test, pred_test):.3f}")
    print(f"Precision Score: {precision_score(y_test, pred_test):.3f}")
    print(f"Recall Score:    {recall_score(y_test, pred_test):.3f}")
    print(f"F1 Score:        {f1_score(y_test, pred_test):.3f}")


# %% apply decision tree classifier and check results
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)
metric_results(clf,y_train,pred_train,y_test,pred_test)

# %% plot the decision tree and look for important features
from sklearn.tree import plot_tree

plot_tree(clf, filled=True, max_depth=2, feature_names=ct.get_feature_names())

# %% apply logistic regression classifier and check results
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(x_train, y_train)
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)
metric_results(clf,y_train,pred_train,y_test,pred_test)

# %% look for important features
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
sns.barplot(y=ct.get_feature_names(), x=clf.coef_[0])


# %% apply naive bayes classifier and check results
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(x_train, y_train)
pred_train = clf.predict(x_train)
pred_test = clf.predict(x_test)
metric_results(clf,y_train,pred_train,y_test,pred_test)

# %% look for important features
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
sns.barplot(y=ct.get_feature_names(), x=clf.coef_[0])


# %%
