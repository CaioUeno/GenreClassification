import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, load_model
from keras.layers import (
    LSTM,
    Activation,
    Dense,
    Dropout,
    Input,
    Embedding,
    GRU,
    LocallyConnected1D,
)
from keras.layers import (
    RNN,
    LSTMCell,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Bidirectional,
    concatenate,
)
from keras.optimizers import RMSprop, Adadelta
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import layers

from sklearn import metrics
import itertools
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
import gc

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 200000

# Load Dataset
data = pd.read_csv("Data/Dataset4.csv")


SIZE = len(data)

# Pre-processing data - can choose one of the following, both (?)
ps = PorterStemmer()
lemma = WordNetLemmatizer()  # recommended

for i in range(len(data)):
    data["song"][i] = data["song"][i].lower()
    words = word_tokenize(data["song"][i])
    # data['song'][i] = ' '.join(ps.stem(w) for w in words)
    data["song"][i] = " ".join(lemma.lemmatize(w) for w in words)


# Tokenizer data and make sequences
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(data["song"])
sequences = tokenizer.texts_to_sequences(data["song"])
dat = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
word_index = tokenizer.word_index

print("Found %s unique tokens." % len(word_index))

encoder = LabelEncoder()
encoder = encoder.fit(["Rock", "Pop", "Country", "Metal", "Jazz", "Indie", "Folk"])

# Size train data
SIZE_TRAIN = int(SIZE * 80 / 100)

# Plot how many examples by genre in train
sns.countplot(data.genre[:SIZE_TRAIN])
plt.xlabel("Label")
plt.title("Train: Number of songs by genres")
plt.savefig("Results/DataTrain.png")
plt.clf()

# Plot how many examples by genre in test
sns.countplot(data.genre[SIZE_TRAIN:])
plt.xlabel("Label")
plt.title("Test: Number of songs by genres")
plt.savefig("Results/DataTest.png")
plt.clf()

# Transforms genre in a number and then make a vector ~~~ Rock -> 3 -> [0 0 1 0 0 0 0]
Genre_Target = encoder.transform(data.genre)
Genre_Target = to_categorical(Genre_Target, num_classes=7, dtype="int32")
class_names = encoder.classes_

# genre list
head = " ".join(str(e) for e in class_names)

# Split dataset into training and test
Songs_train, Songs_test, Genre_train, Genre_test = train_test_split(
    dat, Genre_Target, test_size=0.2, shuffle=False, random_state=4
)

model = load_model("Models/WELSTM.h5")

accr = model.evaluate(Songs_train, Genre_train)
print("LSTM\n Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(accr[0], accr[1]))

PScoreLSTM = model.predict(Songs_train)
matrix = metrics.confusion_matrix(
    Genre_train.argmax(axis=1), PScoreLSTM.argmax(axis=1), labels=range(7)
)

Scores = np.matrix(PScoreLSTM)
np.savetxt("Results/TrainScoresLSTM.txt", Scores, fmt="%.5f", header=head)

##CNN
model = load_model("Models/WECNN.h5")

accr = model.evaluate(Songs_train, Genre_train)
print("CNN\n Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(accr[0], accr[1]))

PScoreCNN = model.predict(Songs_train)
matrix = metrics.confusion_matrix(
    Genre_train.argmax(axis=1), PScoreCNN.argmax(axis=1), labels=range(7)
)

Scores = np.matrix(PScoreCNN)
np.savetxt("Results/TrainScoresCNN.txt", Scores, fmt="%.5f", header=head)
