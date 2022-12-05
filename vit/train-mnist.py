import os
import random
from model import ViT
from dataloaders import load_mnist
from tqdm import tqdm
from itertools import cycle
from tinygrad.nn.optim import get_parameters
import numpy as np
from sklearn.metrics import accuracy_score
import cv2
from PIL import Image
import tinygrad.nn.optim as optim
from tinygrad.tensor import Tensor
import numpy as np

np.seterr(all='warn', over='raise')

def sparse_categorical_crossentropy(out, y):
    return out.log().mul(y).abs().mean()

def grey_to_rgb(img):
    return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)

def one_to_batch(arr):
    return np.expand_dims(arr, axis=0)
        

vit = ViT(out_dim=10, layers=6, num_heads=8, embed_dim=512)

lossfn = lambda out,y: sparse_categorical_crossentropy(out, y)

optim = optim.Adam(get_parameters(vit), lr=0.01)

examples = 60000
images, labels = load_mnist(n=examples)
images, labels = iter(cycle(images)), iter(cycle(labels))

Tensor.training = True
batch_size = 512

batches = examples//batch_size
epochs = 10

for example in range(epochs*batches):
   
    image = np.stack([grey_to_rgb(next(images)) for _ in range(batch_size)], axis=0).astype(np.float32) / 255.0
    label = np.stack([next(labels) for _ in range(batch_size)], axis=0).astype(np.float32)
    
    image, label = Tensor(image).permute(0,3,1,2), Tensor(label)

    image.gpu()
    label.gpu()
    
    image.requires_grad=True
        
    out = vit.forward(image).softmax()
        
    loss = lossfn(out, label)
               
    optim.zero_grad()
    loss.backward()
    optim.step()
        
    preds = np.argmax(out.data, axis=1)
    label = np.argmax(label.data, axis=1)
    print("batch acc: ", accuracy_score(y_pred=preds, y_true=label))
    print("batch loss: ", loss.detach().cpu().data[0], "\n")
    
    del loss, image, label, out