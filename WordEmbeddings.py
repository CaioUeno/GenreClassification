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
from keras.optimizers import RMSprop, Adadelta, Adam
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


# function to choose Major voting
def MVotes(l):
    m = 0
    for item in l:
        q = l.count(item)
        if q >= m:
            a = item
    return a


np.set_printoptions(precision=2)

# Some useful flags
MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 1000
DIM_GLOVE = 50
EPOCHS = 15
BATCH_SIZE = 128

# Accuracies dictionary
lista_acc = {}

# Load Dataset
data = pd.read_csv("Dataset1.csv")


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


# CNN
def CNN():
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
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation="relu")(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.1)(x)
    preds = Dense(7, activation="softmax")(x)
    model = Model(sequence_input, outputs=preds)
    return model


model = CNN()
model.summary()
model.compile(
    loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
)
plot_model(model, to_file="Results/CNN.png", show_shapes=True)

ModelCheckpoints = ModelCheckpoint(
    "Models/WECNN.h5",
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
model.save("Models/WECNN.h5")

# Plot training & validation accuracy values
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Accuracy & loss CNN")
plt.ylabel("Values")
plt.ylim(0, 3)
plt.xlabel("Epoch")
plt.legend(["Train acc", "Val_acc", "Train loss", "Val_loss"], loc="upper right")
plt.grid(True)
plt.savefig("Results/AccLossCNN.png")
plt.clf()

accr = model.evaluate(Songs_test, Genre_test)
print("CNN\n Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(accr[0], accr[1]))
lista_acc["CNN"] = accr[1]

PCNN = model.predict(Songs_test)
matrix = metrics.confusion_matrix(
    Genre_test.argmax(axis=1), PCNN.argmax(axis=1), labels=range(7)
)

Scores = np.matrix(PCNN)
np.savetxt("Results/ScoresCNN.txt", Scores, fmt="%.5f", header=head)


# Plot confusion matrix
plt.figure()
plot_confusion_matrix(matrix, classes=class_names, title="Confusion matrix")
plt.savefig("Results/MatrixCNN.png")
plt.clf()

gc.collect()
del model

# LSTM


def LSTMModel():
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
    l_lstm = LSTM(128, return_sequences=True)(embedded_sequences)
    x = Dense(64, activation="relu")(l_lstm)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    preds = Dense(7, activation="softmax")(x)
    model = Model(sequence_input, outputs=preds)
    return model


model = LSTMModel()
model.summary()
model.compile(
    loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
)
plot_model(model, to_file="Results/LSTM.png", show_shapes=True)

ModelCheckpoints = ModelCheckpoint(
    "Models/WELSTM.h5",
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


# Plot training & validation accuracy values
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Accuracy & loss LSTM")
plt.ylabel("Values")
plt.ylim(0, 3)
plt.xlabel("Epoch")
plt.legend(["Train acc", "Val_acc", "Train loss", "Val_loss"], loc="upper right")
plt.grid(True)
plt.savefig("Results/AccLossLSTM.png")
plt.clf()


accr = model.evaluate(Songs_test, Genre_test)
print("LSTM\n Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(accr[0], accr[1]))
lista_acc["LSTM"] = accr[1]

PLSTM = model.predict(Songs_test)
matrix = metrics.confusion_matrix(
    Genre_test.argmax(axis=1), PLSTM.argmax(axis=1), labels=range(7)
)

Scores = np.matrix(PLSTM)
np.savetxt("Results/ScoresLSTM.txt", Scores, fmt="%.5f", header=head)


# Plot confusion matrix
plt.figure()
plot_confusion_matrix(
    matrix, classes=class_names, title="Confusion matrix, without normalization"
)

plt.savefig("Results/MatrixLSTM.png")
plt.clf()


gc.collect()
del model


# LSTM & CNN


def LSTM_CNN():
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
    x = LSTM(128, return_sequences=True)(embedded_sequences)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Conv1D(128, 5, activation="relu")(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation="relu")(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.5)(x)
    preds = Dense(7, activation="sigmoid")(x)
    model = Model(sequence_input, outputs=preds)
    return model


model = LSTM_CNN()
model.summary()
model.compile(
    loss="categorical_crossentropy", optimizer=Adam(lr=0.0005), metrics=["accuracy"]
)
plot_model(model, to_file="Results/LSTMCNN.png", show_shapes=True)

ModelCheckpoints = ModelCheckpoint(
    "Models/WELSTMCNN.h5",
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


# Plot training & validation accuracy values
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Accuracy & loss LSTM&CNN")
plt.ylabel("Values")
plt.ylim(0, 3)
plt.xlabel("Epoch")
plt.legend(["Train acc", "Val_acc", "Train loss", "Val_loss"], loc="upper right")
plt.grid(True)
plt.savefig("Results/AccLossLSTM&CNN.png")
plt.clf()


accr = model.evaluate(Songs_test, Genre_test)
print(
    "LSTM&CNN\n Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(accr[0], accr[1])
)
lista_acc["LSTM&CNN"] = accr[1]


PLSTMCNN = model.predict(Songs_test)
matrix = metrics.confusion_matrix(
    Genre_test.argmax(axis=1), PLSTMCNN.argmax(axis=1), labels=range(7)
)

Scores = np.matrix(PLSTMCNN)
np.savetxt("Results/ScoresLSTMCNN.txt", Scores, fmt="%.5f", header=head)

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(
    matrix, classes=class_names, title="Confusion matrix, without normalization"
)
plt.savefig("Results/MatrixLSTM&CNN.png")
plt.clf()

gc.collect()
del model

"""
# Locally-connected

def LonCon():
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = Embedding(len(word_index) + 1,DIM_GLOVE,
                            weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                    trainable=True)(sequence_input) #False, but can try with True

    x = LocallyConnected1D(128, 5, activation='relu',dtype='int32')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = LocallyConnected1D(64, 5, activation='relu',dtype='int32')(x)
    x = MaxPooling1D(5)(x)
    x = LocallyConnected1D(32, 5, activation='relu',dtype='int32')(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    preds = Dense(7, activation='sigmoid',dtype='int32')(x)
    model = Model(sequence_input,outputs=preds)
    return model


# In[ ]:


model = LonCon()
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
plot_model(model,to_file='Results/LonCon.png',show_shapes=True)

# In[ ]:


history = model.fit(Songs_train, Genre_train, validation_split=0.2,
            batch_size=BATCH_SIZE,epochs=EPOCHS,
            callbacks=[EarlyStopping])
model.save('Models/LonCon.h5')

# In[ ]:


# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Accuracy & loss LonCon')
plt.ylabel('Values')
plt.ylim(0, 3)
plt.xlabel('Epoch')
plt.legend(['Train acc', 'Val_acc','Train loss', 'Val_loss'], loc='upper right')
plt.grid(True)
plt.savefig('Results/AccLossLonCon.png')
plt.clf()
# In[ ]:


accr = model.evaluate(Songs_test,Genre_test)
print('LonCon\n Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))
lista_acc['LonCon'] = accr[1]

# In[ ]:


PLonCon = model.predict(Songs_test)
matrix = metrics.confusion_matrix(Genre_test.argmax(axis=1), PLonCon.argmax(axis=1),labels=range(0,7))

Scores = np.matrix(PLonCon)
np.savetxt('Results/ScoresLonCon.txt',Scores,fmt='%.5f',header=head)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(matrix, classes=class_names,
                      title='Confusion matrix')
plt.savefig('Results/MatrixLonCon.png')
plt.clf()

gc.collect()
del model
"""
# Merge CNN + LSTM


def MergeCNNLSTM():
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
    x = Conv1D(64, 5, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(64, 5, activation="relu")(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(7, activation="sigmoid")(x)

    l_lstm = LSTM(128, return_sequences=True)(embedded_sequences)
    y = Dense(128, activation="relu")(l_lstm)
    y = Dropout(0.5)(y)
    y = Dense(64, activation="relu")(y)
    y = Dropout(0.5)(y)
    y = Flatten()(y)
    y = Dense(7, activation="relu")(y)

    z = concatenate([x, y])
    z = Dense(128, activation="relu")(z)
    z = Dropout(0.3)(z)
    z = Dense(64, activation="relu")(z)
    z = Dropout(0.3)(z)
    preds = Dense(7, activation="sigmoid")(z)
    model = Model(sequence_input, outputs=preds)
    return model


model = MergeCNNLSTM()
model.summary()
model.compile(
    loss="categorical_crossentropy", optimizer=Adam(lr=0.0005), metrics=["accuracy"]
)
plot_model(model, to_file="Results/MergeCNNLSTM.png", show_shapes=True)

ModelCheckpoints = ModelCheckpoint(
    "Models/WEMergeCNNLSTM.h5",
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


# Plot training & validation accuracy values
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Accuracy & loss MergeCNNLSTM")
plt.ylabel("Values")
plt.ylim(0, 3)
plt.xlabel("Epoch")
plt.legend(["Train acc", "Val_acc", "Train loss", "Val_loss"], loc="upper right")
plt.grid(True)
plt.savefig("Results/AccLossMergeCNNLSTM.png")
plt.clf()


accr = model.evaluate(Songs_test, Genre_test)
print(
    "MergeCNNLSTM\n Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(
        accr[0], accr[1]
    )
)
lista_acc["MergeCNNLSTM"] = accr[1]


PMergeLSTMCNN = model.predict(Songs_test)
matrix = metrics.confusion_matrix(
    Genre_test.argmax(axis=1), PMergeLSTMCNN.argmax(axis=1), labels=range(0, 7)
)

Scores = np.matrix(PMergeLSTMCNN)
np.savetxt("Results/ScoresMergeLSTMCNN.txt", Scores, fmt="%.5f", header=head)

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(matrix, classes=class_names, title="Confusion matrix")
plt.savefig("Results/MatrixMergeCNNLSTM.png")
plt.clf()

gc.collect()
del model


def MergeCNNLSTM2():
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
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation="relu")(x)
    x = MaxPooling1D(35)(x)  # global max pooling
    x = Dense(64, activation="relu")(x)
    x = Flatten()(x)

    l_lstm = LSTM(128, return_sequences=True)(embedded_sequences)
    y = Dense(128, activation="relu")(l_lstm)
    y = Dropout(0.5)(y)
    y = Flatten()(y)

    z = concatenate([x, y])
    z = Dense(256, activation="relu")(z)
    z = Dropout(0.3)(z)
    z = Dense(128, activation="relu")(z)
    z = Dropout(0.3)(z)
    preds = Dense(7, activation="sigmoid")(z)
    model = Model(sequence_input, outputs=preds)
    return model


model = MergeCNNLSTM2()
model.summary()
model.compile(
    loss="categorical_crossentropy", optimizer=Adam(lr=0.0005), metrics=["accuracy"]
)
plot_model(model, to_file="Results/MergeCNNLSTM2.png", show_shapes=True)

ModelCheckpoints = ModelCheckpoint(
    "Models/WEMergeCNNLSTM2.h5",
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


# Plot training & validation accuracy values
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Accuracy & loss MergeCNNLSTM2")
plt.ylabel("Values")
plt.ylim(0, 3)
plt.xlabel("Epoch")
plt.legend(["Train acc", "Val_acc", "Train loss", "Val_loss"], loc="upper right")
plt.grid(True)
plt.savefig("Results/AccLossMergeCNNLSTM2.png")
plt.clf()


accr = model.evaluate(Songs_test, Genre_test)
print(
    "PMergeLSTMCNN2\n Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(
        accr[0], accr[1]
    )
)
lista_acc["MergeCNNLSTM2"] = accr[1]


PMergeLSTMCNN2 = model.predict(Songs_test)
matrix = metrics.confusion_matrix(
    Genre_test.argmax(axis=1), PMergeLSTMCNN2.argmax(axis=1), labels=range(0, 7)
)

Scores = np.matrix(PMergeLSTMCNN2)
np.savetxt("Results/ScoresMergeLSTMCNN2.txt", Scores, fmt="%.5f", header=head)

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(matrix, classes=class_names, title="Confusion matrix")
plt.savefig("Results/MatrixMergeCNNLSTM2.png")
plt.clf()

gc.collect()
print(lista_acc)
