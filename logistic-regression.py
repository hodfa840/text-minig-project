#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns




# In[3]:


def random_subset(data, n):
    
    classes, counts = np.unique(data['class'], return_counts=True)

    # Set size of each subset to be the minimum number of samples in any class
    #subset_size = np.min(counts)
    subset_size = n
    # Initialize empty list to store subsets
    subsets = []

    # Loop over each class and select a random subset of samples
    for cls in classes:
        subset = data[data['class'] == cls].sample(n=subset_size, replace=False)
        subsets.append(subset)

    # Concatenate subsets into a single dataframe
    subset_df = pd.concat(subsets, axis=0)

    # Shuffle rows in the subset dataframe
    subset_df = subset_df.sample(frac=1)

    # Print number of samples in each class in the subset dataframe
   # print(subset_df['class'].value_counts())
    return(subset_df)


# In[18]:


from sklearn.model_selection import train_test_split

# Load the dataset
# Load the categories
categories = pd.read_csv("./classes/classes.txt",header=None,names=["categories"])
categories.index = np.arange(1, len(categories)+1)

# Load the training dataset
train_df = pd.read_csv('./data/train.csv',header=None,names=["class","question_title","question_body","answer"])

train_df.dropna(subset=['question_title','answer'],inplace=True)
train_df = train_df.reset_index()

train_df = train_df.drop(["index", "question_body"], axis = 1)

# Subtract 1 from the 'class' column to make the classes zero-indexed
train_df['class'] = train_df['class'] - 1

n = 30000
# Apply the random_subset function to each dataframe
train_df = random_subset(train_df, n)


# In[19]:


len(train_df)


# In[20]:


# Split the dataset into training, validation and testing sets
train_df, val_test_df = train_test_split(train_df, test_size=0.3, random_state=70)

# Split the validation/testing dataframe into separate validation and testing dataframes
val_df, test_df = train_test_split(val_test_df, test_size=0.5, random_state=70)


# In[13]:


val_df.shape


# In[14]:


train_df.shape


# In[15]:


test_df.shape


# In[21]:


# Unique classes in train_df
unique_classes_train = train_df['class'].unique()
print("Unique classes in train_df: ", unique_classes_train)

# Unique classes in val_df
unique_classes_val = val_df['class'].unique()
print("Unique classes in val_df: ", unique_classes_val)

# Unique classes in test_df
unique_classes_test = test_df['class'].unique()
print("Unique classes in test_df: ", unique_classes_test)


# In[22]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Combine question title and answer into a single field
train_df['text'] = train_df['question_title'] + ' ' + train_df['answer']
val_df['text'] = val_df['question_title'] + ' ' + val_df['answer']
test_df['text'] = test_df['question_title'] + ' ' + test_df['answer']

text_clf_lr = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression(solver='liblinear')),
])

# Fit the model to the training data
text_clf_lr.fit(train_df['text'], train_df['class'])

# Predict the class labels for the validation set
val_predictions_lr = text_clf_lr.predict(val_df['text'])

# Output the classification report for validation set
print(classification_report(val_df['class'], val_predictions_lr))

# Predict the class labels for the test set and output the classification report
test_predictions_lr = text_clf_lr.predict(test_df['text'])
print(classification_report(test_df['class'], test_predictions_lr))


# In[16]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# Combine question title and answer into a single field
train_df['text'] = train_df['question_title'] + ' ' + train_df['answer']
val_df['text'] = val_df['question_title'] + ' ' + val_df['answer']
test_df['text'] = test_df['question_title'] + ' ' + test_df['answer']

# Create a pipeline that transforms the data to a matrix of word counts and then applies the classifier
text_clf_nb = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])

# Fit the model to the training data
text_clf_nb.fit(train_df['text'], train_df['class'])

# Predict the class labels for the validation set
val_predictions_nb = text_clf_nb.predict(val_df['text'])

# Output the classification report for validation set
print(classification_report(val_df['class'], val_predictions_nb))

# Predict the class labels for the test set and output the classification report
test_predictions_nb = text_clf_nb.predict(test_df['text'])
print(classification_report(test_df['class'], test_predictions_nb))


# In[17]:


# inference engine

def predict_category(question_title, answer):
    # Combine the question title and answer into a single string
    text = question_title + ' ' + answer

    # Use the model to predict the class of the text
    prediction = text_clf_lr.predict([text])

    # Get the category name from the categories dataframe
    category_name = categories.loc[prediction[0] + 1, 'categories']

    # Return the category name
    return category_name

# Test the inference engine with a sample question title and answer
sample_question_title = ' Lab Workers or People With Drug TestingExperience;Will Herbal Liver Cleanser help me pass a drug screen?'
sample_answer = 'Well I sure hope you do not believe that.'

predicted_category = predict_category(sample_question_title, sample_answer)
print('Predicted category:', predicted_category)


# In[10]:


# Get the unique class numbers and counts
classes, counts = np.unique(train_df['class'], return_counts=True)

# Print each class number and its corresponding name
for cls, count in zip(classes, counts):
    class_name = categories.loc[cls + 1, 'categories']
    print('Class number:', cls, 'Class name:', class_name, 'Count:', count)


# In[11]:


from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

# Compute confusion matrix for validation set
plt.figure(figsize=(10, 8))
plot_confusion_matrix(text_clf_lr, val_df['text'], val_df['class'], cmap=plt.cm.Blues, normalize='true')
plt.title('Confusion Matrix for Validation Set')
plt.show()

# Compute confusion matrix for test set
plt.figure(figsize=(10, 8))
plot_confusion_matrix(text_clf_lr, test_df['text'], test_df['class'], cmap=plt.cm.Blues, normalize='true')
plt.title('Confusion Matrix for Test Set')
plt.show()


# In[12]:


from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Compute the confusion matrix for validation set
val_predictions = text_clf_lr.predict(val_df['text'])
conf_matrix = confusion_matrix(val_df['class'], val_predictions)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', xticklabels=categories['categories'], yticklabels=categories['categories'])
plt.title('Confusion Matrix for Validation Set')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




