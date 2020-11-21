import sys
import pickle
import os.path as path
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

np.set_printoptions(threshold=sys.maxsize)
np.warnings.filterwarnings('ignore')

OVERWRITE_XHAT_OVA = False
OVERWRITE_XHAT_OVO = False
OVERWRITE_KMEANS = False

def load_data(filename):
    data = sio.loadmat(filename)
    testX = data["testX"]
    testY = data["testY"]
    trainX = data["trainX"]
    trainY = data["trainY"]
    print("Data loaded. Recieved {} training examples and {} test examples".format(len(trainX), len(testX)))
    return testX, testY, trainX, trainY

def clean_data(testX, trainX):
    rows_to_remove = []

    # determine which pixels are mostly blank by iterating rows of trainX and removing all rows with less than min_nonzero nonzero entries
    min_nonzero = 600 # threshold for minimum number of images with nonzero values for the pixel
    for i, row in enumerate(trainX.T): # examples in trainX are rows not columns, so to iterate over pixel indeces, we transpose trainX since enumerate returns the rows
        if(np.count_nonzero(row) < min_nonzero):
            rows_to_remove.append(i)
    print("Removed {} rows for having too few nonzero entries".format(len(rows_to_remove)))
    clean_testX = np.delete(testX, rows_to_remove, 1)
    clean_trainX = np.delete(trainX, rows_to_remove, 1)
    return clean_testX, clean_trainX, rows_to_remove

def add_bias(testX, trainX):
    biased_testX = np.concatenate((testX, np.ones((testX.shape[0], 1))), axis=1)
    biased_trainX = np.concatenate((trainX, np.ones((trainX.shape[0], 1))), axis=1)
    return biased_testX, biased_trainX


def evaluate_model(testX, testY, xhat):
    ybar = np.dot(testX, xhat).reshape((1, testY.shape[1]))
    return ybar

def evaluate_confusian(testY, yhat, save_as=""):
    # print("Evaluating confusion matrix")
    confusion = np.zeros((10, 10))
    for i in range(testY.shape[1]):
        confusion[testY[0, i],yhat[0, i]] += 1
    if(not(save_as == "")):
        np.savetxt(save_as, confusion, fmt="%d", delimiter=",")
    return confusion

def train_one_v_all(trainX, trainY, overwrite=False):
    print("Running one vs all")
    print("Received {}x{} training data".format(trainX.shape[0], trainX.shape[1]))
    XHat = np.zeros((trainX.shape[1], 10))
    for digit in range(10):
        # print("Getting weights for digit {}".format(digit))
        labels = (np.where(trainY == digit, 1, -1)).T
        xhat = np.linalg.lstsq(trainX, labels)[0]
        XHat[:,digit] = xhat.reshape(xhat.shape[0])
    if(overwrite):
        np.savetxt("XHat_OVA.csv", XHat, delimiter=",")
    return XHat

def test_one_v_all(testX, testY, XHat):
    YBar = np.zeros((10, testY.shape[1]))
    for digit in range(10):
        # print("Evaluating model for digit {}".format(digit))
        ybar = evaluate_model(testX, testY, XHat[:,digit])
        YBar[digit,:] = ybar.reshape(ybar.shape[1])
    yhat = np.argmax(YBar, axis=0).reshape((1, YBar.shape[1]))
    return yhat

def train_one_v_one(trainX, trainY, overwrite=False):
    print("Running one vs one")
    print("Received {}x{} training data".format(trainX.shape[0], trainX.shape[1]))
    XHat = np.zeros((trainX.shape[1], 45))
    XHat_idx = 0
    for i in range(10):
        for j in range(i + 1, 10):
            # print("Getting weights for digit pair ({},{})".format(i, j))
            indeces_to_remove = []
            for idx, y in np.ndenumerate(trainY):
                if(not(y == i or y == j)):
                    indeces_to_remove.append(idx)
            only_ij_X = np.delete(trainX, indeces_to_remove, axis=0)
            only_ij_Y = np.delete(trainY, indeces_to_remove)
            labels = (np.where(only_ij_Y == i, 1, -1)).T
            xhat = np.linalg.lstsq(only_ij_X, labels)[0]
            XHat[:,XHat_idx] = xhat.reshape(xhat.shape[0])
            XHat_idx += 1
    if(overwrite):
        np.savetxt("XHat_OVO.csv", XHat, delimiter=",")
    return XHat

def test_one_v_one(testX, testY, XHat):
    YBar = np.zeros((10, testY.shape[1]))
    XHat_idx = 0
    for i in range(10):
        for j in range(i + 1, 10):
            # print("Evaluating model for digit pair ({},{})".format(i, j))
            ybar = evaluate_model(testX, testY, XHat[:,XHat_idx])
            i_row =  np.where(ybar >= 0, YBar[i,:] + 1,  YBar[i])
            j_row =  np.where(ybar < 0, YBar[j,:] + 1,  YBar[j])
            YBar[i,:] = i_row.reshape(i_row.shape[1])
            YBar[j,:] = j_row.reshape(j_row.shape[1])
            XHat_idx += 1

    yhat = np.argmax(YBar, axis=0).reshape((1, YBar.shape[1]))
    return yhat

# def train_kmeans(trainX, K, P):
#     C = np.random.randint(K, size=(1, trainX.shape[0]))
#     Z = np.zeros((K, trainX.shape[1]))
#     Jclust = np.zeros(P)
#     for j in range(P):
#         print("Starting iteration {}".format(j + 1))
#         L2 = np.zeros((trainX.shape[0], K))
#         for i in range(K):
#             Xi = trainX[C[0,:] == i,:]
#             z = (1 / Xi.shape[0]) * np.sum(Xi, axis=0)
#             Z[i,:] = z

#             e = np.linalg.norm(trainX - Z[i,:], axis=1)
#             L2[:,i] = e ** 2
#         Jclust[j] = np.sum(np.argmin(L2, axis=1))
#         if(not(j == P - 1)):
#             C = np.argmin(L2, axis=1).reshape(1, L2.shape[0])
#     return C, Z, Jclust

def train_kmeans(trainX, K, P):
    C = np.random.randint(K, size=(1, trainX.shape[0]))
    Z = np.zeros((K, trainX.shape[1]))
    Jclust = np.zeros(P)
    # Add one iteration because iteration 0 is initializing z0
    for j in range(P + 1):
        print("Starting iteration {}".format(j + 1))
        L2 = np.zeros((trainX.shape[0], K))
        for i in range(K):
            Xi = trainX[C[0,:] == i,:]
            z = (1 / Xi.shape[0]) * np.sum(Xi, axis=0)
            Z[i,:] = z

            L2[:,i] = np.square(np.linalg.norm(trainX - Z[i,:], axis=1))
            if(not(j == 0)):
                # Skip the first iteration because that is just initiazing z0
                Ji =  (1 / trainX.shape[0]) * np.sum(np.square(np.linalg.norm(Xi - Z[i,:], axis=1)))
                Jclust[j - 1] += Ji
        if(not(j == P)):
            # Skip the last iteration so that C and Z are not out of sync (C would be ahead by 1 iteration). This is likely entirely unnecessary
            C = np.argmin(L2, axis=1).reshape(1, L2.shape[0])
    kmeans = {"C": C, "Z": Z, "Jclust": Jclust}
    return kmeans


def cycle_test_data(testX, testY, yhat, offset=0):
    cycle = True
    index = offset
    plt.figure()
    plt.imshow(testX[index,:].reshape(28, 28), cmap="binary")
    print("Predicted: {}, Actual: {}".format(yhat[0, index], testY[0, index]))
    plt.show(block=False)
    while(cycle):
        value = input("Press 'Enter' to show next digit. Press any other key (then 'Enter') to exit.")
        if(value != ""):
            cycle = False
        else:
            plt.imshow(testX[index,:].reshape(28, 28), cmap="binary")
            print("Predicted: {}, Actual: {}".format(yhat[0, index], testY[0, index]))
            plt.show(block=False)
            index += 1

def print_stats(confusion, alg_name):
    total_correct = 0
    for i in range(10):
        total_correct += confusion[i, i]
    
    print("Percent Error for ${}: {}%".format(alg_name, (1 - total_correct / np.sum(confusion)) * 100))
    
def main():
    raw_testX, testY, raw_trainX, trainY = load_data("mnist")
    clean_testX, clean_trainX, rows_removed = clean_data(raw_testX, raw_trainX)
    testX, trainX = add_bias(clean_testX, clean_trainX)

    # XHat_OVA = train_one_v_all(trainX, trainY, OVERWRITE_XHAT_OVA) if not(path.exists("XHat_OVA.csv")) or OVERWRITE_XHAT_OVA else np.loadtxt("XHat_OVA.csv", delimiter=",")
    # yhat_OVA = test_one_v_all(testX, testY, XHat_OVA)
    # confusion_OVA = evaluate_confusian(testY, yhat_OVA, "confusion_OVA.csv")
    # print_stats(confusion_OVA, "One Vs All")

    # XHat_OVO = train_one_v_one(trainX, trainY, OVERWRITE_XHAT_OVO) if not(path.exists("XHat_OVO.csv")) or OVERWRITE_XHAT_OVO else np.loadtxt("XHat_OVO.csv", delimiter=",")
    # yhat_OVO = test_one_v_one(testX, testY, XHat_OVO)
    # confusion_OVO = evaluate_confusian(testY, yhat_OVO, "confusion_OVO.csv")
    # print_stats(confusion_OVO, "One Vs One")

    KMEANS_PATH = "kmeans.p"
    kmeans = train_kmeans(trainX, 10, 20) if not(path.exists(KMEANS_PATH)) or OVERWRITE_KMEANS else pickle.load(open(KMEANS_PATH, "rb"))
    C = kmeans["C"]
    Z = kmeans["Z"]
    Jclust = kmeans["Jclust"]
    plt.plot(Jclust)
    plt.show()
    
    v = input("Would you like to save this result? [Y/n]")
    if(v == "Y"):
        kmeans = {"C": C, "Z": Z, "Jclust": Jclust}
        pickle.dump(kmeans, open(KMEANS_PATH, "wb"))

    paddedZ = np.zeros((Z.shape[0], raw_trainX.shape[1]))
    for i in range(Z.shape[0]):
        zpad = np.zeros((1, raw_trainX.shape[1]))
        offset = 0
        for j in range(raw_trainX.shape[1]):
            if(j in rows_removed):
                zpad[0, j] = 0
                offset += 1
            else: 
                zpad[0, j] = Z[i,j - offset]
        
        plt.figure()
        plt.imshow(zpad.reshape(28, 28), cmap="binary")
        plt.show()
        

    print("Exiting program successfully")
    
main()