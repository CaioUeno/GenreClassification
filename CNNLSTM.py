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


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Reds
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    else:
        print("Confusion matrix")

    # print(cm)

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()


np.set_printoptions(precision=2)

# Some useful flags
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 350
DIM_GLOVE = 50
EPOCHS = 20
BATCH_SIZE = 128

# Accuracies dictionary
lista_acc = {}

# Load Dataset
data = pd.read_csv("D1.csv")


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
print("MAX_SEQUENCE_LENGTH :::: %s." % MAX_SEQUENCE_LENGTH)

encoder = LabelEncoder()
encoder = encoder.fit(["Rock", "Pop", "Country", "Metal", "Jazz", "Indie", "Folk"])

# Size train data
SIZE_TRAIN = int(SIZE * 80 / 100)

# Plot how many examples by genre in train
sns.countplot(data.genre[:SIZE_TRAIN])
plt.xlabel("Label")
plt.title("Train: Number of songs by genres")
plt.savefig("DataTrain.png")
plt.clf()

# Plot how many examples by genre in test
sns.countplot(data.genre[SIZE_TRAIN:])
plt.xlabel("Label")
plt.title("Test: Number of songs by genres")
plt.savefig("DataTest.png")
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

# free memory
del data, dat, Genre_Target

# Makes a dictionary 'word': [vector]
embeddings_index = {}
f = open("glove.6B.50d.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype="float32")
    embeddings_index[word] = coefs
f.close()

print("Found %s word vectors." % len(embeddings_index))

# Makes a matrix, setting for each word in text its word-embedding vector
embedding_matrix = np.random.random((len(word_index) + 1, DIM_GLOVE))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# Callback Flag
EarlyStopping = EarlyStopping(monitor="val_acc", patience=8, mode="max")

# CNNLSTM
def CNNLSTM():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype="int32")
    embedded_sequences = Embedding(
        len(word_index) + 1,
        DIM_GLOVE,
        weights=[embedding_matrix],
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=True,
    )(
        sequence_input
    )  # False, but can try with True
    x = Conv1D(128, 5, activation="relu")(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = MaxPooling1D(15)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    preds = Dense(7, activation="softmax")(x)
    model = Model(sequence_input, outputs=preds)
    return model


model = CNNLSTM()
model.summary()
model.compile(
    loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
)
plot_model(model, to_file="CNNLSTM.png", show_shapes=True)

ModelCheckpoints = ModelCheckpoint(
    "WECNNLSTM.h5",
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    period=1,
)

history = model.fit(
    Songs_train,
    Genre_train,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[EarlyStopping, ModelCheckpoints],
)
model.save("WECNNLSTM.h5")

# Plot training & validation accuracy values
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Accuracy & loss CNNLSTM")
plt.ylabel("Values")
plt.ylim(0, 3)
plt.xlabel("Epoch")
plt.legend(["Train acc", "Val_acc", "Train loss", "Val_loss"], loc="upper right")
plt.grid(True)
plt.savefig("AccLossCNNLSTM.png")
plt.clf()

accr = model.evaluate(Songs_test, Genre_test)
print(
    "CNNLSTM\n Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(accr[0], accr[1])
)
lista_acc["CNNLSTM"] = accr[1]

PCNN = model.predict(Songs_test)
matrix = metrics.confusion_matrix(
    Genre_test.argmax(axis=1), PCNN.argmax(axis=1), labels=range(7)
)

Scores = np.matrix(PCNN)
np.savetxt("ScoresCNNLSTM.txt", Scores, fmt="%.5f", header=head)


# Plot confusion matrix
plt.figure()
plot_confusion_matrix(matrix, classes=class_names, title="Confusion matrix")
plt.savefig("MatrixCNNLSTM.png")
plt.clf()

gc.collect()
del model
