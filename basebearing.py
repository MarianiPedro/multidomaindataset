import os
import scipy.io
import numpy as np
from bearingconditions import BearingConditions

class BaseBearing(BearingConditions):
    def __init__(self):
        super().__init__()
        self.rawpath = None
        self.model = None
        
    def LoadData(self, condition, fault,  samplingRate, speed):
        filename = self.generate_filename(condition, fault, samplingRate, self.model, speed)
        filepath = os.path.join(self.rawpath, f"SamplingRate_{samplingRate}", f"RotatingSpeed_{speed}", filename)
    
        try:
            if not os.path.exists(filepath):
                print(f"Arquivo não encontrado: {filepath}")
                dir_path = os.path.dirname(filepath)
                if os.path.exists(dir_path):
                    print("Arquivos no diretório:")
                    for f in os.listdir(dir_path):
                        if f.endswith('.mat'):
                            print(f"   - {f}")
                return None
            mat_data = scipy.io.loadmat(filepath)
            
            # --=== CHAVES DISPONÍVEIS ===--
            # - Data: <class 'numpy.ndarray'>
            # - STFTFreq: <class 'numpy.ndarray'>
            # - STFTTime: <class 'numpy.ndarray'>
            # - Spectrogram: <class 'numpy.ndarray'>
            # --===--===--===--===--===--===--===--===
            # print(f"Chaves disponíveis:")
            # for key in mat_data.keys():
            #     if not key.startswith('__'):
            #         print(f"   - {key}: {type(mat_data[key])}")
            # return None
            
            return mat_data['Data']
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    
    def PreProcessData(self, raw_data, window_size=1024,  overlap=0.5):
        if raw_data is None:
            return np.array([])
            
        if len(raw_data.shape) > 1:
            raw_data = raw_data.flatten()
            
        normalized = (raw_data - np.mean(raw_data)) / np.std(raw_data)
        step = int(window_size * (1 - overlap))
        windows = []
        for i in range(0, len(normalized) - window_size + 1, step):
            window = normalized[i:i+window_size]
            windows.append(window)
        
        return np.array(windows)