import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
categories = pd.read_csv("./classes.txt",header=None,names=["categories"])
categories.index = np.arange(1, len(categories)+1)
# print(categories)
data = pd.read_csv('train.csv',header=None,names=["class","question_title","question_body","answer"])
# print(data.head())
data.dropna(subset='question_title',inplace=True)
data.dropna(subset='answer',inplace=True)
data = data.reset_index()
# print(data.shape)

for idx,cs in enumerate(categories['categories'],start=1):
    data.loc[data['class']==idx,'class'] = cs

data = data.drop(["index", "question_body"], axis = 1)
