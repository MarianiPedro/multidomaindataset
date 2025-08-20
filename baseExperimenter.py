import os
import sys
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

from keras import layers, models, applications
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from basebearing import BaseBearing

from datasets.deepgrooveball import DeepGrooveBall
from datasets.cylindricalroller import CylindricalRoller
from datasets.taperedroller import TaperedRoller

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class SimpleCNN1D:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()
        
    def build_model(self):
        model = tf.keras.Sequential([
            layers.Input(shape=self.input_shape),
            
            layers.Conv1D(32, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Dropout(0.3),
            
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            
            layers.Dense(self.num_classes, activation='softmax')
        ])
        return model
    
    def compile_model(self, learning_rate=0.001):
        self.model.compile(
            optimizer= tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        print("Iniciando treinamento...")
        start_time = time.time()
        
        history = self.model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        training_time = time.time() - start_time
        print(f"Tempo de treinamento: {training_time:.2f} segundos")
        
        return history, training_time
    
    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"Acurácia no teste: {accuracy:.4f}")
        return accuracy
    
def load_preprocess_data():
    
    # conditions = BearingConditions().condition
    # faults = BearingConditions().fault
    # sampling_rate = BearingConditions().samplingRate
    # speed = BearingConditions().speed
    
    
    X_all = []
    y_all = []
    
    base = BaseBearing()
    base.rawpath = r"D:\PEDRO\Mestrado\Proejto_BearingFaultDiagnosis\Multi-domain vibration dataset\multidomaindataset\datasets\raw_datasets\BearingType_DeepGrooveBall"
    conditions = ["H", "U1"]
    faults = ["H", "B"]
    sampling_rate = 8000
    speed = 1000
    base.model = DeepGrooveBall().model
    
    if not os.path.exists(base.rawpath):
        print(f"Diretório não encontrado: {base.rawpath}")
        print("Diretórios existentes:")
        for item in os.listdir('.'):
            if os.path.isdir(item):
                print(f"  - {item}/")
        return np.array([]), np.array([]), 0
    
    for condition in conditions:
        for fault in faults:
            try:
                print(f"Processando: {condition}_{fault}_{sampling_rate//1000}_{base.model}_{speed}")

                num_windows = 50  
                window_size = 1024
                
                raw_data = base.LoadData(condition, fault, sampling_rate, speed)
                if raw_data is None:
                    print(f"Arquivo não encontrado ou erro no carregamento")
                    continue
                print(f"Dados brutos carregados: shape {raw_data.shape}")
                windows = base.PreProcessData(raw_data, window_size=window_size, overlap=0.5)
                print(f"Janelas processadas: {windows.shape}")
                
                X_all.extend(windows)
                
                label = f"{condition}_{fault}"
                y_all.extend([label] * num_windows)
                
                print(f" {label}: {num_windows} janelas")
                
            except Exception as e:
                print(f" Erro em {condition}_{fault}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
            
    if len(X_all) != len(y_all):
        print(f"INCONSISTÊNCIA: X({len(X_all)}) != y({len(y_all)})")
        min_length = min(len(X_all), len(y_all))
        X_all = X_all[:min_length]
        y_all = y_all[:min_length]
        print(f"Corrigido: X={len(X_all)}, y={len(y_all)}")
    
    if len(X_all) == 0:
        return np.array([]), np.array([]), 0
    
    X = np.array(X_all)
    y = np.array(y_all)
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nDados preparados:")
    print(f"Shape de X: {X.shape}")
    print(f"Labels únicos: {label_encoder.classes_}")
    print(f"Distribuição: {np.bincount(y_encoded)}")
    
    return X, y_encoded, len(label_encoder.classes_)
    
def run_cnn_test():
    print("=== TESTE CNN 1D - DIAGNÓSTICO DE FALHAS ===")
    
    EPOCHS = 5
    BATCH_SIZE = 8
    
    X, y, num_classes = load_preprocess_data()
    
    if len(X) == 0:
        print("Nenhum dado foi carregado. Verifique os caminhos dos arquivos.")
        return
    
    X = X[..., np.newaxis]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        print("Inconsistência após split!")
        # Corrige
        min_train = min(len(X_train), len(y_train))
        X_train, y_train = X_train[:min_train], y_train[:min_train]
        
        min_test = min(len(X_test), len(y_test))
        X_test, y_test = X_test[:min_test], y_test[:min_test]
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    # )
    
    
    
    print(f"Treino: {X_train.shape}")
    #print(f"Validação: {X_val.shape}")
    print(f"Teste: {X_test.shape}")
    
    cnn = SimpleCNN1D(input_shape=(X_train.shape[1], X_train.shape[2]), num_classes=num_classes)
    cnn.compile_model(learning_rate=0.001)
    
    cnn.model.summary()
    
    print("\nIniciando treinamento...")
    history, training_time = cnn.train(
        X_train, y_train, X_test, y_test, 
        epochs=EPOCHS, batch_size=BATCH_SIZE
    )
    
    test_accuracy = cnn.evaluate(X_test, y_test)
    
    print(f"\n=== RESULTADOS ===")
    print(f"Tempo total de treinamento: {training_time:.2f}s")
    print(f"Acurácia final: {test_accuracy:.4f}")
    print(f"Épocas: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Amostras totais: {len(X)}")
    
    return history, training_time, test_accuracy

if __name__ == "__main__":
    print("GPUs disponíveis:", tf.config.list_physical_devices('GPU'))
    history, training_time, accuracy = run_cnn_test()