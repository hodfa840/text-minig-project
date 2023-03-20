from data_loading import *
from data_preprocessing import *
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
TF_ENABLE_ONEDNN_OPTS=0

def build_dic(df, cleaned_text, classes):
    # Use a dictionary comprehension to create the dictionary
    dic = {cls: [] for cls in classes}

    # Use iterrows to iterate over the rows of the dataframe
    for i, row in df.iterrows():
        cls = row['class']  
        text = cleaned_text[i]
        answer = row['answer'] 
        dic[cls].append((text, answer))
        # if i==10:
        #     break

    return dic

if os.path.exists("dic_train.pkl"):
    with open("dic_train.pkl", "rb") as f:
        dic = pickle.load(f)
else:
    dic = build_dic(data, clean_texts, classes=categories["categories"])
    with open("dic_train.pkl", "wb") as f:
        pickle.dump(dic, f)





def encoding_data(cleaned_text, labels, classes):

    tokenizer_file = "tokenizer_train.pkl"
    label_encoder_file = "label_encoder_train.pkl"
    data_sequences_file = "data_sequences_train.pkl"
    data_label_file = "data_label_train.pkl"
    word_index_file = "word_index_train.pkl"

    if os.path.isfile(tokenizer_file):
        with open(tokenizer_file, "rb") as file:
            tokenizer = pickle.load(file)
    else:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(cleaned_text)
        with open(tokenizer_file, "wb") as file:
            pickle.dump(tokenizer, file)

    # word_index is a dictionary contains indeces for the words like {'for': 8, 'is': 9, 'me': 10, ...}
    word_index = tokenizer.word_index

    if os.path.isfile(data_sequences_file):
        with open(data_sequences_file, "rb") as file:
            data_sequences = pickle.load(file)
    else:
        # train_sequences is vectors where each vector represents a sentence
        data_sequences = tokenizer.texts_to_sequences(cleaned_text)
        data_sequences = pad_sequences(data_sequences, maxlen=10, padding="post")
        with open(data_sequences_file, "wb") as file:
            pickle.dump(data_sequences, file)

    if os.path.isfile(label_encoder_file):
        with open(label_encoder_file, "rb") as file:
            label_encoder = pickle.load(file)
    else:
        # Convert our labels into one-hot encoded
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(classes)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

        onehot_encoder = OneHotEncoder(sparse=False)
        onehot_encoder.fit(integer_encoded)

        with open(label_encoder_file, "wb") as file:
            pickle.dump(label_encoder, file)

    if os.path.isfile(data_label_file):
        with open(data_label_file, "rb") as file:
            data_label = pickle.load(file)
    else:
        data_label_encoded = label_encoder.transform(labels)
        data_label_encoded = data_label_encoded.reshape(len(data_label_encoded), 1)
        data_label = onehot_encoder.transform(data_label_encoded)

        with open(data_label_file, "wb") as file:
            pickle.dump(data_label, file)

    if os.path.isfile(word_index_file):
        with open(word_index_file, "rb") as file:
            word_index = pickle.load(file)
    else:
        word_index = tokenizer.word_index
        with open(word_index_file, "wb") as file:
            pickle.dump(word_index, file)

    return data_sequences, data_label, word_index


# Usage:
data_sequences, data_class, word_index = encoding_data(clean_texts, data["class"], classes=categories["categories"])

def GloVe(data = 'glove.6B.100d.txt'):

  embeddings_index={}
  with open(data, encoding='utf8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
  return embeddings_index
#--------------------------------------------------------------------------------------------#
def embeddingWights(word_index, max_num_words=500000):
  embeddings_index = GloVe()
  all_embs = np.stack(embeddings_index.values())
  emb_mean,emb_std = all_embs.mean(), all_embs.std()

  num_words = min(max_num_words, len(word_index))+1

  embedding_dim=len(embeddings_index['the'])

  embedding_wights = np.random.normal(emb_mean, emb_std, (num_words, embedding_dim))

  for word, i in word_index.items():
      if i >= max_num_words:
          break
      embedding_vector = embeddings_index.get(word)
      if embedding_vector is not None:
          embedding_wights[i] = embedding_vector
  
  return embedding_wights, num_words


if not os.path.exists("embedding_wights_train.pkl") or not os.path.exists("num_words_train.pkl"):
    embedding_wights, num_words = embeddingWights(word_index)

    with open("embedding_wights_train.pkl", "wb") as a_file:
        pickle.dump(embedding_wights, a_file)
    with open("num_words.pkl_train", "wb") as a_file:
        pickle.dump(num_words, a_file)
else:
    with open("embedding_wights.pkl_train", "rb") as a_file:
        embedding_wights = pickle.load(a_file)
    with open("num_words.pkl_train", "rb") as a_file:
        num_words = pickle.load(a_file)


