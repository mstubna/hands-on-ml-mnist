import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml

# load the data set
mnist = fetch_openml('mnist_784', version=1)

X, y = mnist['data'], mnist['target']
y = y.astype(np.uint8)

# visualize some digits in the data set
fig, axs = plt.subplots(nrows=10, ncols=20, figsize=(8, 4))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
for row in range(10):
  digits = X[y == row]
  for col in range(20):
    digit_image = digits[col].reshape(28, 28)
    axis = axs[row, col]
    axis.imshow(
      digit_image,
      cmap=mpl.cm.get_cmap('binary'),
      interpolation='nearest'
    )
    axis.axis('off')

plt.show()