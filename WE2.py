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


EPOCHS = 25
BATCH_SIZE = 128

# Accuracies dictionary
lista_acc = {}

# Load Dataset
data = pd.read_csv("Data/Dataset4.csv")

encoder = LabelEncoder()
# Transforms genre in a number and then make a vector ~~~ Rock -> 3 -> [0 0 1 0 0 0 0]
Genre_Target = encoder.fit_transform(data.genre)
Genre_Target = to_categorical(Genre_Target, num_classes=7, dtype="int32")
class_names = encoder.classes_

# genre list
head = " ".join(str(e) for e in class_names)

ScoresLSTM = np.loadtxt("Results/TrainScoresLSTM.txt")
ScoresCNN = np.loadtxt("Results/TrainScoresCNN.txt")
Scores = []
print(ScoresCNN.shape)
print(ScoresLSTM.shape)
print(Genre_Target.shape)
for i in range(len(ScoresCNN)):
    Scores.append(np.concatenate((ScoresLSTM[i], ScoresCNN[i]), axis=None))

Scores = np.asarray(Scores)
# Split dataset into training and test
Scores_train, Scores_test, Genre_train, Genre_test = train_test_split(
    data, Genre_Target, test_size=0.2, shuffle=False, random_state=4
)
# Split dataset into training and test
Scores_train, Scores_test, Genre_train, Genre_test = train_test_split(
    Scores, Genre_train, test_size=0.2, shuffle=False, random_state=4
)

# free memory


# Callback Flag
EarlyStopping = EarlyStopping(monitor="val_acc", patience=8, mode="max")


def ScoreLSTMCNN():
    sequence_input = Input(shape=(14,), dtype="float32")
    x = Dense(258)(sequence_input)
    x = Dense(512)(x)
    preds = Dense(7, activation="sigmoid")(x)
    model = Model(sequence_input, outputs=preds)
    return model


model = ScoreLSTMCNN()
model.summary()
model.compile(
    loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
)

ModelCheckpoints = ModelCheckpoint(
    "ScoreLSTMCNN.h5",
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    period=1,
)

history = model.fit(
    Scores_train,
    Genre_train,
    validation_split=0.2,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    callbacks=[EarlyStopping],
)

# Plot training & validation accuracy values
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Accuracy & loss ScoreLSTMCNN")
plt.ylabel("Values")
plt.ylim(0, 3)
plt.xlabel("Epoch")
plt.legend(["Train acc", "Val_acc", "Train loss", "Val_loss"], loc="upper right")
plt.grid(True)
plt.savefig("Results/AccScoreLSTMCNN.png")
plt.clf()

accr = model.evaluate(Scores_test, Genre_test)
print("CNN\n Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(accr[0], accr[1]))

PCNN = model.predict(Scores_test)
matrix = metrics.confusion_matrix(
    Genre_test.argmax(axis=1), PCNN.argmax(axis=1), labels=range(7)
)

Scores = np.matrix(PCNN)
np.savetxt("Results/ScoreLSTMCNN.txt", Scores, fmt="%.5f", header=head)


# Plot confusion matrix
plt.figure()
plot_confusion_matrix(matrix, classes=class_names, title="Confusion matrix")
plt.savefig("Results/MatrixScoreLSTMCNN.png")
plt.clf()

gc.collect()
del model
