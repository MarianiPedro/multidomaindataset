class BearingConditions():
    #format: {condition}_{fault}_{samplingrate}_{model}_{speed}.mat.
    def __init__(self):
        self.samplingRate = [8000, 16000] # kHz
        self.speed = [600, 800, 1000, 1200, 1400, 1600] # RPM
        self.condition = ["H", "L", "U1", "U2", "U3", "M1", "M2", "M3"] # healthy (H), looseness (L), unbalance (U), misalignment (M) ~ severity 1-low, 2-medium, 3-high
        self.fault = ["H", "B", "OR", "IR"] # healthy (H), bearing (B), outter rail (OR), inner rail (IR)
        pass
    
    def generate_filename(self, condition, fault, samplimgRate, model, speed):
        return f"{condition}_{fault}_{samplimgRate//1000}_{model}_{speed}.mat"