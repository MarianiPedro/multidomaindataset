class CylindricalRoller():
    def __init__(self):
        super().__init__()
        self.rawpath = "raw_datasets/BearingType_CylindricalRoller"
        #format: {condition}_{fault}_{samplingrate}_{model}_{speed}.mat.
        self.model = ["N204", "NJ204"]
        pass