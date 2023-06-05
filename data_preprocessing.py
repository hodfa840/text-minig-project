from data_loading import *
import spacy
import pandas as pd
import pickle
import os
nlp = spacy.load('en_core_web_sm')


# function to preprocess the data
def preprocess_data(texts):
    clean_texts = []
    i = 0 
    for text in texts:
        document = nlp(text)
        current_text = [str(token.lemma_) for token in document if not (token.is_stop or  token.is_punct or len(token) < 3)]

        clean_texts.append(current_text)
        i += 1
        if i % 10000 == 0:
            print(f"Processed {i} texts")
    print(f"Processed {i} texts")

    return clean_texts


# choosing subset of the data

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
    print(subset_df['class'].value_counts())
    return(subset_df)
n = 30000
sampled_data = random_subset(data,n )
sampled_data = data
# Preprocess the data

# Check if file exists
if not os.path.exists("cleaned_question_title.pkl"):
    # If file doesn't exist, preprocess data and pickle to file
    clean_texts = preprocess_data(sampled_data['question_title'])
    cleaned_question_title_pkl = open("cleaned_question_title.pkl", "wb")
    pickle.dump(clean_texts, cleaned_question_title_pkl)
    cleaned_question_title_pkl.close()
else:
    # If file exists, load pickled data from file
    with open("cleaned_question_title.pkl", "rb") as f:
        clean_texts = pickle.load(f)




