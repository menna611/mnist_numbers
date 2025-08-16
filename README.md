# MNIST Digit Classification

This project trains a neural network to classify handwritten digits using the MNIST dataset.  
It uses TensorFlow, Keras, Pandas, and Matplotlib.

## Project Structure

- `mnist.ipynb`: Main Jupyter notebook for data loading, preprocessing, model building, training, and evaluation.
- `mnist_train.csv`, `mnist_test.csv`: Training and test datasets (CSV format).
- `requirements.txt`: Python dependencies.


## Model Overview

- **Input:** 784 features (flattened 28x28 images)
- **Architecture:**  
  - Dense layer (128 units, ReLU)
  - Dense layer (10 units, Softmax)
- **Loss:** Categorical Crossentropy
- **Optimizer:** Adam
- **Early Stopping:** Monitors validation accuracy

## Results

The model achieves high accuracy on the MNIST test set.  
You can visualize sample predictions and training curves in the notebook.



