import os
from classification.naive_bayes import NaiveBayes
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT_DIR, "model")
DATA_DIR = os.path.join(ROOT_DIR, "data")

df = pd.read_csv(os.path.join(DATA_DIR, "NBA_Season_Stats(1984-2015).csv"))
# 对同时包含在两个类别的数据进行筛选
df = df[df["Pos"].isin(["PF", "PG", "SG", "C", "SF"])]
df.fillna("missing")

x = df.iloc[:, 1:]
y = df.iloc[:, :1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

nb = NaiveBayes()

nb.fit(x_train=x_train, y_train=y_train)
print(nb.evaluation(x_test=x_test, y_test=y_test))
nb.save(os.path.join(MODEL_DIR, "nba.model.json"))