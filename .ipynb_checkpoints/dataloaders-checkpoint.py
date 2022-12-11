import pandas as pd
import numpy as np
import random
import cv2

def load_mnist(n=5000):
    df = pd.read_csv("datasets/mnist_train.csv").sample(n)
    labels = df.label.to_numpy()
    labels_one_hot = np.zeros((labels.size, 10), dtype=int)
    labels_one_hot[np.arange(labels.size),labels] = 1 
    images = np.array([row.to_numpy().reshape(28,28) for _, row in df.drop(columns=["label"]).iterrows()])
    dataset = list(zip(images, labels_one_hot))
    random.shuffle(dataset)
    images, labels_one_hot = zip(*dataset)
    return images, labels_one_hot

def load_sat_rep():
    return cv2.imread("plots/small.jpg"), cv2.imread("plots/large.jpg")