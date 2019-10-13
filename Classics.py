from sklearn import (
    model_selection,
    preprocessing,
    linear_model,
    naive_bayes,
    metrics,
    svm,
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble

import xgboost, textblob, string
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.text import Tokenizer

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import itertools
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
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


np.set_printoptions(precision=2)

# load the dataset
data = pd.read_csv("Dataset1.csv")


# Pre-processing data - can choose one of the following, both (?)
ps = PorterStemmer()
lemma = WordNetLemmatizer()

for i in range(len(data)):
    data["song"][i] = data["song"][i].lower()
    words = word_tokenize(data["song"][i])
    # data['song'][i] = ' '.join(ps.stem(w) for w in words)
    data["song"][i] = " ".join(lemma.lemmatize(w) for w in words)

MAX_FEATURES = 5000

# split the dataset into training and validation datasets
Song_train, Song_valid, Genre_train, Genre_valid = model_selection.train_test_split(
    data["song"], data["genre"], test_size=0.2, shuffle=False, random_state=4
)

# label encode the target variable
encoder = preprocessing.LabelEncoder()
encoder.fit(["Rock", "Pop", "Country", "Metal", "Jazz", "Indie", "Folk"])
class_names = list(encoder.classes_)
Genre_train = encoder.transform(Genre_train)
Genre_valid = encoder.transform(Genre_valid)

# genre list
head = " ".join(str(e) for e in class_names)

# create a count vectorizer object
count_vect = CountVectorizer(
    analyzer="word", token_pattern=r"\w{1,}", stop_words="english"
)
count_vect.fit(data["song"])

# transform the training and validation data using count vectorizer object
Song_train_count = count_vect.transform(Song_train)
Song_valid_count = count_vect.transform(Song_valid)


# word level tf-idf
tfidf_vect = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"\w{1,}",
    max_features=MAX_FEATURES,
    stop_words="english",
    use_idf=True,
)
tfidf_vect.fit(data["song"])
Song_train_tfidf = tfidf_vect.transform(Song_train)
Song_valid_tfidf = tfidf_vect.transform(Song_valid)

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(
    analyzer="word",
    token_pattern=r"\w{1,}",
    ngram_range=(2, 3),
    max_features=MAX_FEATURES,
    stop_words="english",
)
tfidf_vect_ngram.fit(data["song"])
Song_train_tfidf_ngram = tfidf_vect_ngram.transform(Song_train)
Song_valid_tfidf_ngram = tfidf_vect_ngram.transform(Song_valid)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(
    analyzer="char",
    token_pattern=r"\w{1,}",
    ngram_range=(1, 2),
    max_features=MAX_FEATURES,
    stop_words="english",
)
tfidf_vect_ngram_chars.fit(data["song"])
Song_train_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(Song_train)
Song_valid_tfidf_ngram_chars = tfidf_vect_ngram_chars.transform(Song_valid)

# Naive Bayes model

NB_results = {}
# Naive Bayes on Count Vectors
modelCV = naive_bayes.MultinomialNB()
modelCV.fit(Song_train_count, Genre_train)
accuracy = metrics.accuracy_score(modelCV.predict(Song_valid_count), Genre_valid)
print("NB, Count Vectors: ", accuracy)
NB_results["Count Vectors"] = accuracy

# Naive Bayes on Word Level TF IDF Vectors
modelTF = naive_bayes.MultinomialNB()
modelTF.fit(Song_train_tfidf, Genre_train)
accuracy = metrics.accuracy_score(modelTF.predict(Song_valid_tfidf), Genre_valid)
print("NB, WordLevel TF-IDF: ", accuracy)
NB_results["WordLevel TF-IDF"] = accuracy

# Naive Bayes on Ngram Level TF IDF Vectors
modelTFN = naive_bayes.MultinomialNB()
modelTFN.fit(Song_train_tfidf_ngram, Genre_train)
accuracy = metrics.accuracy_score(modelTFN.predict(Song_valid_tfidf_ngram), Genre_valid)
print("NB, N-Gram Vectors: ", accuracy)
NB_results["N-Gram Vectors"] = accuracy

# Naive Bayes on Character Level TF IDF Vectors
modelTFC = naive_bayes.MultinomialNB()
modelTFC.fit(Song_train_tfidf_ngram_chars, Genre_train)
accuracy = metrics.accuracy_score(
    modelTFC.predict(Song_valid_tfidf_ngram_chars), Genre_valid
)
print("NB, N-CharLevel Vectors: ", accuracy)
NB_results["N-CharLevel Vectors"] = accuracy

print(
    "Best result:",
    max(NB_results, key=NB_results.get),
    "= {:0.2f}".format(NB_results[max(NB_results, key=NB_results.get)]),
)

# Choose best level to save
with open("Models/NB_model.pkl", "wb") as file:
    if max(NB_results, key=NB_results.get) == "Count Vectors":
        pickle.dump(modelCV, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelCV.predict(Song_valid_count), labels=range(7)
        )
        scores = np.matrix(modelCV.predict_proba(Song_valid_count))

    elif max(NB_results, key=NB_results.get) == "WordLevel TF-IDF":
        pickle.dump(modelTF, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTF.predict(Song_valid_tfidf), labels=range(7)
        )
        scores = np.matrix(modelTF.predict_proba(Song_valid_tfidf))

    elif max(NB_results, key=NB_results.get) == "N-Gram Vectors":
        pickle.dump(modelTFN, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTFN.predict(Song_valid_tfidf_ngram), labels=range(7)
        )
        scores = np.matrix(modelTFN.predict_proba(Song_valid_tfidf_ngram))

    elif max(NB_results, key=NB_results.get) == "N-CharLevel Vectors":
        pickle.dump(modelTFC, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTFC.predict(Song_valid_tfidf_ngram_chars), labels=range(7)
        )
        scores = np.matrix(modelTFC.predict_proba(Song_valid_tfidf_ngram_chars))

    # plot confusion matrix
    plt.figure()
    plot_confusion_matrix(matrix, classes=class_names, title="Confusion matrix")
    plt.savefig("Results/MatrixNaiveBayes.png")
    plt.clf()

    np.savetxt("Results/ScoresNB.txt", scores, fmt="%.5f", header=head)
print("Scores saved in Results/ScoresNB.txt...")
print("Best model saved as NB_model.pkl...")

# Linear Classifier model

LC_results = {}
# Linear Classifier on Count Vectors
modelCV = linear_model.LogisticRegression(
    max_iter=100, multi_class="multinomial", n_jobs=-1, solver="sag"
)
modelCV.fit(Song_train_count, Genre_train)
accuracy = metrics.accuracy_score(modelCV.predict(Song_valid_count), Genre_valid)
print("LC, Count Vectors: ", accuracy)
LC_results["Count Vectors"] = accuracy

# Linear Classifier on Word Level TF IDF Vectors
modelTF = linear_model.LogisticRegression(
    max_iter=200, multi_class="multinomial", n_jobs=-1, solver="sag"
)
modelTF.fit(Song_train_tfidf, Genre_train)
accuracy = metrics.accuracy_score(modelTF.predict(Song_valid_tfidf), Genre_valid)
print("LC, WordLevel TF-IDF: ", accuracy)
LC_results["WordLevel TF-IDF"] = accuracy

# Linear Classifier on Ngram Level TF IDF Vectors
modelTFN = linear_model.LogisticRegression(
    max_iter=100, multi_class="multinomial", n_jobs=-1, solver="sag"
)
modelTFN.fit(Song_train_tfidf_ngram, Genre_train)
accuracy = metrics.accuracy_score(modelTFN.predict(Song_valid_tfidf_ngram), Genre_valid)
print("LC, N-Gram Vectors: ", accuracy)
LC_results["N-Gram Vectors"] = accuracy

# Linear Classifier on Character Level TF IDF Vectors
modelTFC = linear_model.LogisticRegression(
    max_iter=100, multi_class="multinomial", n_jobs=-1, solver="sag"
)
modelTFC.fit(Song_train_tfidf_ngram_chars, Genre_train)
accuracy = metrics.accuracy_score(
    modelTFC.predict(Song_valid_tfidf_ngram_chars), Genre_valid
)
print("LC, N-CharLevel Vectors: ", accuracy)
LC_results["N-CharLevel Vectors"] = accuracy


print(
    "Best result:",
    max(LC_results, key=LC_results.get),
    "= {:0.2f}".format(LC_results[max(LC_results, key=LC_results.get)]),
)

# Choose best level to save
with open("Models/LC_model.pkl", "wb") as file:
    if max(LC_results, key=LC_results.get) == "Count Vectors":
        pickle.dump(modelCV, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelCV.predict(Song_valid_count), labels=range(7)
        )
        scores = np.matrix(modelCV.predict_proba(Song_valid_count))

    elif max(LC_results, key=LC_results.get) == "WordLevel TF-IDF":
        pickle.dump(modelTF, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTF.predict(Song_valid_tfidf), labels=range(7)
        )
        scores = np.matrix(modelTF.predict_proba(Song_valid_tfidf))

    elif max(LC_results, key=LC_results.get) == "N-Gram Vectors":
        pickle.dump(modelTFN, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTFN.predict(Song_valid_tfidf_ngram), labels=range(7)
        )
        scores = np.matrix(modelTFN.predict_proba(Song_valid_tfidf_ngram))

    elif max(LC_results, key=LC_results.get) == "N-CharLevel Vectors":
        pickle.dump(modelTFC, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTFC.predict(Song_valid_tfidf_ngram_chars), labels=range(7)
        )
        scores = np.matrix(modelTFC.predict_proba(Song_valid_tfidf_ngram_chars))

    # plot confusion matrix
    plt.figure()
    plot_confusion_matrix(matrix, classes=class_names, title="Confusion matrix")
    plt.savefig("Results/MatrixLRegression.png")
    plt.clf()

    np.savetxt("Results/ScoresLC.txt", scores, fmt="%.5f", header=head)
print("Scores saved in Results/ScoresLC.txt...")
print("Best model saved as LC_model.pkl...")

# SVM model

SVM_results = {}
# SVM on on Count Vectors
modelCV = svm.SVC(max_iter=10, gamma="scale", probability=True)
modelCV.fit(Song_train_count, Genre_train)
accuracy = metrics.accuracy_score(modelCV.predict(Song_valid_count), Genre_valid)
print("SVM, Count Vectors: ", accuracy)
SVM_results["Count Vectors"] = accuracy

# SVM on Word Level TF IDF Vectors
modelTF = svm.SVC(max_iter=10, gamma="scale", probability=True)
modelTF.fit(Song_train_tfidf, Genre_train)
accuracy = metrics.accuracy_score(modelTF.predict(Song_valid_tfidf_ngram), Genre_valid)
print("SVM, WordLevel TF-IDF: ", accuracy)
SVM_results["WordLevel TF-IDF"] = accuracy

# SVM on Ngram Level TF IDF Vectors
modelTFN = svm.SVC(max_iter=10, gamma="scale", probability=True)
modelTFN.fit(Song_train_tfidf_ngram, Genre_train)
accuracy = metrics.accuracy_score(modelTFN.predict(Song_valid_tfidf), Genre_valid)
print("SVM, N-Gram Vectors: ", accuracy)
SVM_results["N-Gram Vectors"] = accuracy

# SVM on Character Level TF IDF Vectors
modelTFC = svm.SVC(max_iter=10, gamma="scale", probability=True)
modelTFC.fit(Song_train_tfidf_ngram_chars, Genre_train)
accuracy = metrics.accuracy_score(
    modelTFC.predict(Song_valid_tfidf_ngram_chars), Genre_valid
)
print("SVM, N-CharLevel Vectors: ", accuracy)
SVM_results["N-CharLevel Vectors"] = accuracy


print(
    "Best result:",
    max(SVM_results, key=SVM_results.get),
    "= {:0.2f}".format(SVM_results[max(SVM_results, key=SVM_results.get)]),
)

# Choose best level to save
with open("Models/SVM_model.pkl", "wb") as file:
    if max(SVM_results, key=SVM_results.get) == "Count Vectors":
        pickle.dump(modelCV, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelCV.predict(Song_valid_count), labels=range(7)
        )
        scores = np.matrix(modelCV.predict_proba(Song_valid_count))

    elif max(SVM_results, key=SVM_results.get) == "WordLevel TF-IDF":
        pickle.dump(modelTF, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTF.predict(Song_valid_tfidf), labels=range(7)
        )
        scores = np.matrix(modelTF.predict_proba(Song_valid_tfidf))

    elif max(SVM_results, key=SVM_results.get) == "N-Gram Vectors":
        pickle.dump(modelTFN, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTFN.predict(Song_valid_tfidf_ngram), labels=range(7)
        )
        scores = np.matrix(modelTFN.predict_proba(Song_valid_tfidf_ngram))

    elif max(SVM_results, key=SVM_results.get) == "N-CharLevel Vectors":
        pickle.dump(modelTFC, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTFC.predict(Song_valid_tfidf_ngram_chars), labels=range(7)
        )
        scores = np.matrix(modelTFC.predict_proba(Song_valid_tfidf_ngram_chars))

    # plot confusion matrix
    plt.figure()
    plot_confusion_matrix(matrix, classes=class_names, title="Confusion matrix")
    plt.savefig("Results/MatrixSVM.png")
    plt.clf()

    np.savetxt("Results/ScoresSVM.txt", scores, fmt="%.5f", header=head)
print("Scores saved in Results/ScoresSVM.txt...")
print("Best model saved as SVM_model.pkl...")

# RandomForest model

RF_results = {}
# RF on Count Vectors
modelCV = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100)
modelCV.fit(Song_train_count, Genre_train)
accuracy = metrics.accuracy_score(modelCV.predict(Song_valid_count), Genre_valid)
print("RF, Count Vectors: ", accuracy)
RF_results["Count Vectors"] = accuracy

# RF on Word Level TF IDF Vectors
modelTF = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100)
modelTF.fit(Song_train_tfidf, Genre_train)
accuracy = metrics.accuracy_score(modelTF.predict(Song_valid_tfidf), Genre_valid)
print("RF, WordLevel TF-IDF: ", accuracy)
RF_results["WordLevel TF-IDF"] = accuracy

# RF on Ngram Level TF IDF Vectors
modelTFN = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100)
modelTFN.fit(Song_train_tfidf_ngram, Genre_train)
accuracy = metrics.accuracy_score(modelTFN.predict(Song_valid_tfidf_ngram), Genre_valid)
print("RF, N-Gram Vectors: ", accuracy)
RF_results["N-Gram Vectors"] = accuracy

# RF on Character Level TF IDF Vectors
modelTFC = ensemble.RandomForestClassifier(n_jobs=-1, n_estimators=100)
modelTFC.fit(Song_train_tfidf_ngram_chars, Genre_train)
accuracy = metrics.accuracy_score(
    modelTFC.predict(Song_valid_tfidf_ngram_chars), Genre_valid
)
print("RF, CharLevel Vectors: ", accuracy)
RF_results["N-CharLevel Vectors"] = accuracy


print(
    "Best result:",
    max(RF_results, key=RF_results.get),
    "= {:0.2f}".format(RF_results[max(RF_results, key=RF_results.get)]),
)

# Choose best level to save
with open("Models/RF_model.pkl", "wb") as file:
    if max(RF_results, key=RF_results.get) == "Count Vectors":
        pickle.dump(modelCV, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelCV.predict(Song_valid_count), labels=range(7)
        )
        scores = np.matrix(modelCV.predict_proba(Song_valid_count))

    elif max(RF_results, key=RF_results.get) == "WordLevel TF-IDF":
        pickle.dump(modelTF, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTF.predict(Song_valid_tfidf), labels=range(7)
        )
        scores = np.matrix(modelTF.predict_proba(Song_valid_tfidf))

    elif max(RF_results, key=RF_results.get) == "N-Gram Vectors":
        pickle.dump(modelTFN, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTFN.predict(Song_valid_tfidf_ngram), labels=range(7)
        )
        scores = np.matrix(modelTFN.predict_proba(Song_valid_tfidf_ngram))

    elif max(RF_results, key=RF_results.get) == "N-CharLevel Vectors":
        pickle.dump(modelTFC, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTFC.predict(Song_valid_tfidf_ngram_chars), labels=range(7)
        )
        scores = np.matrix(modelTFC.predict_proba(Song_valid_tfidf_ngram_chars))

    # plot confusion matrix
    plt.figure()
    plot_confusion_matrix(matrix, classes=class_names, title="Confusion matrix")
    plt.savefig("Results/MatrixRForest.png")
    plt.clf()
    np.savetxt("Results/ScoresRF.txt", scores, fmt="%.5f", header=head)
print("Scores saved in Results/ScoresRF.txt...")
print("Best model saved as RF_model.pkl...")

# XGBoost model
GB_results = {}
# Extereme Gradient Boosting on Count Vectors
modelCV = xgboost.XGBClassifier(n_estimators=400)
modelCV.fit(Song_train_count, Genre_train)
accuracy = metrics.accuracy_score(modelCV.predict(Song_valid_count), Genre_valid)
print("Xgb, Count Vectors: ", accuracy)
GB_results["Count Vectors"] = accuracy

# Extereme Gradient Boosting on Word Level TF IDF Vectors
modelTF = xgboost.XGBClassifier(n_estimators=400)
modelTF.fit(Song_train_tfidf, Genre_train)
accuracy = metrics.accuracy_score(modelTF.predict(Song_valid_tfidf), Genre_valid)
print("Xgb, WordLevel TF-IDF: ", accuracy)
GB_results["WordLevel TF-IDF"] = accuracy

# Extereme Gradient Boosting on Ngram Level TF IDF Vectors
modelTFN = xgboost.XGBClassifier(n_estimators=400)
modelTFN.fit(Song_train_tfidf_ngram, Genre_train)
accuracy = metrics.accuracy_score(modelTFN.predict(Song_valid_tfidf_ngram), Genre_valid)
print("Xgb, N-Gram Vectors: ", accuracy)
GB_results["N-Gram Vectors"] = accuracy

# Extereme Gradient Boosting on Character Level TF IDF Vectors
modelTFC = xgboost.XGBClassifier(n_estimators=400)
modelTFC.fit(Song_train_tfidf_ngram_chars, Genre_train)
accuracy = metrics.accuracy_score(
    modelTFC.predict(Song_valid_tfidf_ngram_chars), Genre_valid
)
print("Xgb, N-CharLevel Vectors: ", accuracy)
GB_results["N-CharLevel Vectors"] = accuracy


print(
    "Best result:",
    max(GB_results, key=GB_results.get),
    "= {:0.2f}".format(GB_results[max(GB_results, key=GB_results.get)]),
)

# Choose best level to save
with open("Models/GB_model.pkl", "wb") as file:
    if max(GB_results, key=GB_results.get) == "Count Vectors":
        pickle.dump(modelCV, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelCV.predict(Song_valid_count), labels=range(7)
        )
        scores = np.matrix(modelCV.predict_proba(Song_valid_count))

    elif max(GB_results, key=GB_results.get) == "WordLevel TF-IDF":
        pickle.dump(modelTF, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTF.predict(Song_valid_tfidf), labels=range(7)
        )
        scores = np.matrix(modelTF.predict_proba(Song_valid_tfidf))

    elif max(GB_results, key=GB_results.get) == "N-Gram Vectors":
        pickle.dump(modelTFN, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTFN.predict(Song_valid_tfidf_ngram), labels=range(7)
        )
        scores = np.matrix(modelTFN.predict_proba(Song_valid_tfidf_ngram))

    elif max(GB_results, key=GB_results.get) == "N-CharLevel Vectors":
        pickle.dump(modelTFC, file)
        matrix = metrics.confusion_matrix(
            Genre_valid, modelTFC.predict(Song_valid_tfidf_ngram_chars), labels=range(7)
        )
        scores = np.matrix(modelTFC.predict_proba(Song_valid_tfidf_ngram_chars))

    # plot confusion matrix
    plt.figure()
    plot_confusion_matrix(matrix, classes=class_names, title="Confusion matrix")
    plt.savefig("Results/MatrixXGBoost.png")
    plt.clf()

    np.savetxt("Results/ScoresGB.txt", scores, fmt="%.5f", header=head)
print("Scores saved in Results/ScoresGB.txt...")
print("Best model saved as GB_model.pkl...")
