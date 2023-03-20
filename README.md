# Comparative Study of Deep Learning Models for Text Classification: LSTM, CNN with GloVe Embedding, and BERT


 This project explores the use of various deep learning models for text classification on the Yahoo Answers dataset. The models considered are Long Short-Term Memory (LSTM), Convolutional Neural Network (CNN) with GloVe embedding, and Bidirectional Encoder Representations from Transformers (BERT). For LSTM and CNN models, pre-trained GloVe embeddings are used, while for BERT, pre-trained BERT embeddings are used. The performance of each model is evaluated using accuracy and ROC AUC score metrics.
 
 ## Dataset
 
 The dataset can be downloaded from: https://drive.google.com/drive/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M?resourcekey=0-TLwzfR2O-D2aPitmn5o9VQ
 
 ## Embeddings
The GloVe embeddings can be downloaded from: https://nlp.stanford.edu/data/glove.6B.zip (version glove.6B.100d)

Insert both files in the project folder.

### Running the Project
1. Run data_loading.py to load the dataset.
2. Run data_preprocessing.py to preprocess the data.
3. Run tokenizing.py to tokenize the text data.

Then use the respective notebooks to run and visualize the models.

### Notebooks
The project contains the following notebooks:

- lstm.ipynb: LSTM model.
- cnn.ipynb: CNN model.
- bert.ipynb: BERT model.

Each notebook contains the necessary code to train and evaluate the respective models.
