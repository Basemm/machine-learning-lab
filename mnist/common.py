import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml


def load_data():
  return fetch_openml('mnist_784', as_frame=False)

def get_dataset():
  mnist = load_data()

  train_set_cutoff = 60000

  X_train, y_train = mnist.data[:train_set_cutoff], mnist.target[:train_set_cutoff]
  X_test, y_test = mnist.data[train_set_cutoff:], mnist.target[train_set_cutoff:]


  return X_train, y_train, X_test, y_test

def plot_image(pixel_data, w=28, h=28):
  pixel_data_reshaped = pixel_data.reshape(w, h)
  plt.imshow(pixel_data_reshaped, cmap="binary")
  plt.gcf().set_size_inches(1, 1)
  plt.axis("off")
