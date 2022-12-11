# standard
import copy

# local
from dataloaders import load_sat_rep
from models import ViT

# external
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import get_parameters
import tinygrad.nn.optim as optim
import numpy as np
import cv2
from patchify import patchify

# raise error on overflow caused by gelu
np.seterr(all="warn", over="raise")

# hyperparameters
EPOCHS = 10000
BATCH_SIZE = 8
LR = 1e-3

# batch img
def batchify(img, bs=8):
    return np.array([img for _ in range(bs)])

# patch img
def patch(batch, bs):
    return np.array([patchify(img, (20,20,3), step=20) for img in batch]).reshape(bs,-1,20,20,3)

# normalize img
def norm(batch):
    return batch/255.0

# mean squared error
def lossfn(pred, target):
    return target.sub(pred).square().mean()

# reconstruct image patches
def jigsaw_to_image(x, grid_size=(20, 20)):
    batch_size, num_patches, c, jigsaw_h, jigsaw_w = x.shape
    assert num_patches == grid_size[0] * grid_size[1]
    x_image = x.reshape(batch_size, grid_size[0], grid_size[1], c, jigsaw_h, jigsaw_w)
    output_h = grid_size[0] * jigsaw_h
    output_w = grid_size[1] * jigsaw_w
    x_image = x_image.permute(0, 3, 1, 4, 2, 5).contiguous()
    x_image = x_image.reshape(batch_size, c, output_h, output_w)
    return x_image

# load dataset
small, _ = load_sat_rep()

# initialize model
vit = ViT(
    layers=3,
    embed_dim=400,
    num_heads=8,
    img_dim=(400,400),
    out_dim=None,
    hid_dim=512,
    patch_size=20,
    batch=BATCH_SIZE
)
optim = optim.Adam(get_parameters(vit), lr=LR)

Tensor.traning = True

# training loop
for i in range(EPOCHS):
    
    # build input tensor
    x = copy.deepcopy(small) #h, w, c
    x = batchify(x, bs=BATCH_SIZE) # batch, h, w, c
    x = patch(x, BATCH_SIZE) # batch, patches, p_h, p_w, c
    x = norm(x) # batch, patches, p_h, p_w, c
    x = Tensor(x) # batch, patches, p_h, p_w, c
    x.gpu()
    x.requires_grad=True
    
    # build target tensor
    target = copy.deepcopy(small)
    target = cv2.cvtColor(target.astype(np.uint8), cv2.COLOR_BGR2GRAY) # h, w, c
    target = batchify(target, bs=BATCH_SIZE) # batch, h, w, c
    target = norm(target) # batch, h, w, c 
    target = Tensor(target) # batch, h, w, c
    target.gpu()
    target.requires_grad=True
    
    # forward pass
    pred = vit.forward(x) # batch, patches, features
    pred = pred.reshape(BATCH_SIZE, 400, 1, 20, 20) # batch, patches, c, p_h, p_w
    pred = jigsaw_to_image(pred) # batch, c, h, w
    pred = pred.reshape(BATCH_SIZE, 400, 400) # batch, h, w
    
    # backward pass
    loss = lossfn(pred, target)
    optim.zero_grad()
    loss.backward()
    optim.step()
    
    # print loss to stdout
    print("Iter: {}, Loss: {}".format(i, loss.data[0]))
    
    # save predicted image every 10 iterations
    if i % 100 == 0:
        pred = pred.detach().cpu() # batch, h, w
        img = pred[0].data # h, w
        img = np.expand_dims(img, axis=-1) # h, w, c
        img = img.astype(np.uint8)*255 # h, w, c
        cv2.imwrite("plots/iter_{}.jpg".format(i), img)
        del img
        
    # clear cuda memory
    del x, target, pred, loss
    