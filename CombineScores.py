import numpy as np
from scipy import stats


def voting(labels_matrix, expected_labels):

    ensembled_labels, count_mode = stats.mode(labels_matrix, axis=0)
    accuracy = np.sum(ensembled_labels == expected_labels) / expected_labels.shape[0]

    return ensembled_labels, accuracy


def voting_by_scores(scores_matrix, expected_labels):
    labels_matrix = np.argmax(scores_matrix, axis=2)
    return voting(labels_matrix, expected_labels)


def scores_average(scores_matrix, expected_labels):

    averages_scores = np.mean(scores_matrix, axis=0)
    ensembled_labels = np.argmax(averages_scores, axis=1)

    accuracy = np.sum(ensembled_labels == expected_labels) / expected_labels.shape[0]

    return ensembled_labels, accuracy


def scores_multiply(scores_matrix, expected_labels):

    n_models = scores_matrix.shape[0]
    multiplied_scores = scores_matrix[0, :, :]

    for i_model in range(1, n_models):
        multiplied_scores = multiplied_scores * scores_matrix[i_model, :, :]

    ensembled_labels = np.argmax(multiplied_scores, axis=1)
    accuracy = np.sum(ensembled_labels == expected_labels) / expected_labels.shape[0]

    return ensembled_labels, accuracy


# exemplo com 5 classes, 100 exemplos
# expected_labels = np.random.randint(5, size = 100) # 1 linha contendo o id da classe, com 100 colunas (uma para cada exemplo)

expected_labels = np.loadtxt("D1.txt", skiprows=0)
print(("Labels esperados: ", expected_labels))
"""
labels1 = np.random.randint(5, size = 100) # 1 linha contendo o id da classe, com 100 colunas (uma para cada exemplo)
labels2 = np.random.randint(5, size = 100)
labels3 = np.random.randint(5, size = 100)

labels_matrix = np.vstack((labels1,labels2,labels3))
print(('Labels para os metodos: ',labels_matrix))

ensembled_labels, accuracy = voting(labels_matrix, expected_labels)
print(('Labels combinados: ',ensembled_labels))
print(('Acuracia: ',accuracy))


# exemplo com scores
scores1 = np.random.rand(100, 5) # 100 linhas (uma para cada exemplo), com 5 colunas (score para cada classe)
scores2 = np.random.rand(100, 5)
scores3 = np.random.rand(100, 5)
"""
scoresNB = np.loadtxt("BoW/D1/Results/ScoresNB.txt", skiprows=0)
scoresLC = np.loadtxt("BoW/D1/Results/ScoresLC.txt", skiprows=0)
scoresRF = np.loadtxt("BoW/D1/Results/ScoresRF.txt", skiprows=0)
scoresCNN = np.loadtxt("RNP/D1/Results/ScoresCNN.txt", skiprows=0)
scoresLSTM = np.loadtxt("RNP/D1/Results/ScoresLSTM.txt", skiprows=0)
scoresLSTMCNN = np.loadtxt("RNP/D1/Results/ScoresLSTMCNN.txt", skiprows=0)
scoresMergeLSTMCNN1 = np.loadtxt("RNP/D1/Results/ScoresMergeLSTMCNN.txt", skiprows=0)
scoresMergeLSTMCNN2 = np.loadtxt("RNP/D1/Results/ScoresMergeLSTMCNN.txt", skiprows=0)
scoresHAN = np.loadtxt("HAN/D1/Results/ScoresHAN.txt", skiprows=0)

# scores_matrix = np.stack((scoresNB,scoresLC,scoresRF,scoresCNN,scoresLSTM,scoresLSTMCNN,scoresMergeLSTMCNN1,scoresMergeLSTMCNN2,scoresHAN), axis=0)

scores_matrix = np.stack((scoresCNN, scoresLSTM), axis=0)


ensembled_labels, accuracy = voting_by_scores(scores_matrix, expected_labels)
print(("Labels combinados: ", ensembled_labels))
print(("Acuracia: ", accuracy))

ensembled_labels, accuracy = scores_average(scores_matrix, expected_labels)
print(("Labels combinados: ", ensembled_labels))
print(("Acuracia: ", accuracy))

ensembled_labels, accuracy = scores_multiply(scores_matrix, expected_labels)
print(("Labels combinados: ", ensembled_labels))
print(("Acuracia: ", accuracy))
