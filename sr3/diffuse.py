import numpy as np
import cv2


def diffuse(img, beta, t):

    a_hat = 1-np.prod(beta[:t])

    return np.sqrt(a_hat)*img + np.random.normal(size=img.shape)*np.sqrt(1-a_hat)

img = cv2.imread("images/small.jpg")
beta_start = 1e-4
beta_end = 2e-2
t = 2000
beta = np.linspace(start=beta_start, stop=beta_end, num=t)

out = diffuse(img, beta, 500)

cv2.imwrite("images/out.jpg", out)

print("finished")