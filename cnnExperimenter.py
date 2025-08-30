import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from keras import layers
from scipy.io import loadmat

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

from datasets.deepgrooveball import DeepGrooveBall

if tf.config.list_physical_devices('GPU'):
    print("GPU detectada e será utilizada:", tf.config.list_physical_devices('GPU'))
else:
    print("Nenhuma GPU detectada. Usando CPU.")
    
def KerasCNN1D(input_shape, num_classes):
  inputs = keras.Input(shape=input_shape)
  x = layers.Conv1D(32, kernel_size=4, activation="relu")(inputs)
  x = layers.Flatten()(x)
  x = layers.Dense(128, activation="relu")(x)
  outputs = layers.Dense(num_classes, activation="softmax")(x)
  return keras.Model(inputs=inputs, outputs=outputs)

def KerasCNN2D(input_shape, num_classes):
  inputs = keras.Input(shape=input_shape)
  x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(inputs)
  x = layers.MaxPooling2D(pool_size=2)(x)
  x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
  x = layers.MaxPooling2D(pool_size=2)(x)
  x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
  x = layers.Flatten()(x)
  outputs = layers.Dense(num_classes, activation="softmax")(x)
  return keras.Model(inputs=inputs, outputs=outputs)

class Divide255(BaseEstimator, TransformerMixin):
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    return (X / 255.0).astype("float32")

class Shape2Keras(BaseEstimator, TransformerMixin):
  def __init__(self, window_size=1024):
        self.window_size = window_size
  def fit(self, X, y=None):
    return self
  def transform(self, X):
    return X.reshape(-1, self.window_size, 1)

class KerasWrappedNN(BaseEstimator, ClassifierMixin):
  def __init__(self, epochs=5, batch_size=128, model_fabric=KerasCNN1D):
    self.epochs = epochs
    self.batch_size = batch_size
    self.model_fabric = model_fabric

  def fit(self, X, y):
    self.labels, ids = np.unique(y, return_inverse=True)
    yhot = keras.utils.to_categorical(ids, len(self.labels))
    self.model = self.model_fabric(X.shape[1:], len(self.labels))
    self.model.compile(optimizer="rmsprop",
                       loss="categorical_crossentropy",
                       metrics=["accuracy"])
    self.model.fit(X, yhot, epochs=self.epochs, batch_size=self.batch_size, verbose=1)
    return self

  def predict(self, X):
    probabilities = self.model.predict(X)
    return self.labels[np.argmax(probabilities, axis=1)]

rootpath = os.path.dirname(__file__)

def load_preprocess_data_multidomain(root_dir, window_size=1024, overlap=0.5):
    X_all = []
    y_all = []

    #print(f"Root: {root_dir}")
    for class_SamplingRate in os.listdir(root_dir):
        class_path_SamplingRate = os.path.join(root_dir, class_SamplingRate)
        if not os.path.isdir(class_path_SamplingRate):
            continue

        print(f"Processando classe: {class_SamplingRate}")

        #for class_RotatingSpeed  in os.listdir(class_path_SamplingRate):
        for class_RotatingSpeed  in ['RotatingSpeed_1000', 'RotatingSpeed_600']:
            class_path_RotatingSpeed = os.path.join(class_path_SamplingRate, class_RotatingSpeed)
            if not os.path.isdir(class_path_RotatingSpeed):
                continue
            
            print(f"Processando classe: {class_RotatingSpeed}")
            
            for file_name in os.listdir(class_path_RotatingSpeed):
                #print(f"Arquivo: {file_name}")
                if not file_name.endswith(".mat"):
                    continue
                
                file_path = os.path.join(class_path_RotatingSpeed, file_name)
                try:
                    mat_data = loadmat(file_path)
                    #signal_key = [k for k in mat_data.keys() if not k.startswith("__")][0]
                    raw_data = np.squeeze(mat_data['Data'])
                except Exception as e:
                    print(f"Erro ao carregar {file_name}: {e}")
                    continue

                raw_data = (raw_data - np.mean(raw_data)) / np.std(raw_data)

                step = int(window_size * (1 - overlap))
                windows = [
                    raw_data[i:i + window_size]
                    for i in range(0, len(raw_data) - window_size, step)
                ]

                X_all.extend(windows)
                y_all.extend([class_RotatingSpeed] * len(windows))

                #print(f"{class_SamplingRate}_{class_RotatingSpeed}: {len([w for w in windows])} janelas extraídas")

    X_all = np.array(X_all)
    y_all = np.array(y_all)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_all)

    print(f"\nDados preparados:")
    print(f"Shape de X: {X_all.shape}")
    print(f"Labels únicos: {label_encoder.classes_}")
    print(f"Distribuição: {np.bincount(y_encoded)}")

    return X_all, y_encoded, len(label_encoder.classes_)

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de Confusão")
    plt.show()
    
X, y, num_classes = load_preprocess_data_multidomain(root_dir=f'{rootpath}/datasets/{DeepGrooveBall().rawpath}')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = Pipeline([
    ("scaler", Divide255()),
    ("reshaper", Shape2Keras(window_size=1024)),
    ("model", KerasWrappedNN(model_fabric=KerasCNN1D))
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Acurácia de teste: {accuracy_score(y_test, y_pred):.4f}")

labels = np.unique(y)
plot_confusion_matrix(y_test, y_pred, labels)