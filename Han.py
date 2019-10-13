import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from keras import initializers, layers
from keras import backend as K

from keras.engine.topology import Layer
from keras.models import Model, load_model
from keras.layers import TimeDistributed, Bidirectional, Embedding, LSTM, GRU, Conv1D
from keras.layers import MaxPooling1D, Dropout, Dense, Input, Reshape, Flatten
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import metrics

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# from nltk.corpus import stopwords


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Oranges
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

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


MAX_NB_WORDS = 30000
MAX_SEQUENCE_LENGTH = 1000
DIM_GLOVE = 50
EPOCHS = 30
BATCH_SIZE = 64
MAX_WORDS = 20  # max number of words in line
MAX_LINES = 50  # max number of lines in songs

data = pd.read_csv("Dataset1.csv")

# Pre-processing data - can choose one of the following, both (?)
ps = PorterStemmer()
lemma = WordNetLemmatizer()

# stopwords
# stopwords = set(stopwords.words('english'))

for i in range(len(data)):
    data["song"][i] = data["song"][i].lower()
    words = word_tokenize(data["song"][i])
    # data['song'][i] = ' '.join(ps.stem(w) for w in words)
    data["song"][i] = " ".join(lemma.lemmatize(w) for w in words)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(data["song"])
sequences = tokenizer.texts_to_sequences(data["song"])
dat = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

Trainset = []
for i in range(len(dat)):
    dat[i] = list(dat[i])
    Trainset.append(list(np.reshape(dat[i], (-1, MAX_WORDS))))
Trainset = np.asarray(Trainset)

word_index = tokenizer.word_index
print("Found %s unique tokens." % len(word_index))

encoder = LabelEncoder()
encoder.fit(["Rock", "Pop", "Country", "Metal", "Jazz", "Indie", "Folk"])
class_names = list(encoder.classes_)
Genre_Target = data.genre
Genre_Target = encoder.transform(Genre_Target)
Genre_Target = to_categorical(Genre_Target, num_classes=7, dtype="int32")

# genre list
head = " ".join(str(e) for e in class_names)

Songs_train, Songs_test, Genre_train, Genre_test = train_test_split(
    Trainset, Genre_Target, test_size=0.20, shuffle=False, random_state=4
)

embeddings_index = {}
f = open("glove.6B.50d.txt")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype="float32")
    embeddings_index[word] = coefs
f.close()

print("Found %s word vectors." % len(embeddings_index))

embedding_matrix = np.random.random((len(word_index) + 1, DIM_GLOVE))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be random numbers.
        embedding_matrix[i] = embedding_vector


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get("normal")
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


sentence_input = Input(shape=(MAX_WORDS,), dtype="int32")
embedded_sequences = Embedding(
    len(word_index) + 1,
    DIM_GLOVE,
    weights=[embedding_matrix],
    # input_length=MAX_SENT_LENGTH,
    trainable=True,
)(sentence_input)
l_lstm = Bidirectional(LSTM(64, return_sequences=True))(embedded_sequences)
l_att = AttLayer(64)(l_lstm)
sentEncoder = Model(sentence_input, l_att)

review_input = Input(shape=(MAX_LINES, MAX_WORDS), dtype="int32")
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(LSTM(64, return_sequences=True))(review_encoder)
l_att_sent = AttLayer(64)(l_lstm_sent)
preds = Dense(7, activation="softmax")(l_att_sent)
model = Model(review_input, preds)

model.compile(
    loss="categorical_crossentropy", optimizer=RMSprop(), metrics=["accuracy"]
)

model.summary()

ModelCheckpoints = ModelCheckpoint(
    "HAN.h5",
    monitor="val_acc",
    save_best_only=True,
    save_weights_only=False,
    mode="max",
    period=1,
)
EarlyStopping = EarlyStopping(monitor="val_acc", patience=7, mode="max")


plot_model(model, to_file="HAN.png", show_shapes=True)
history = model.fit(
    Songs_train,
    Genre_train,
    validation_split=0.2,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping, ModelCheckpoints],
)

# Plot training & validation accuracy values
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Accuracy & loss HAN")
plt.ylabel("Values")
plt.ylim(0, 3)
plt.xlabel("Epoch")
plt.legend(["Train acc", "Val_acc", "Train loss", "Val_loss"], loc="upper right")
plt.grid(True)
plt.savefig("Results/AccLossHAN.png")
plt.clf()

accr = model.evaluate(Songs_test, Genre_test)
print("HAN\n Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(accr[0], accr[1]))


PHAN = model.predict(Songs_test)
matrix = metrics.confusion_matrix(
    Genre_test.argmax(axis=1), PHAN.argmax(axis=1), labels=range(7)
)
Scores = np.matrix(PHAN)
np.savetxt("Results/ScoresHAN.txt", Scores, fmt="%.5f", header=head)

# Plot confusion matrix
plt.figure()
plot_confusion_matrix(matrix, classes=class_names, title="Confusion matrix")
plt.savefig("Results/MatrixHAN.png")
plt.clf()
