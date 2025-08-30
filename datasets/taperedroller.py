class TaperedRoller():
    def __init__(self):
        super().__init__()
        self.rawpath = "raw_datasets/BearingType_TaperedRoller"
        #format: {condition}_{fault}_{samplingrate}_{model}_{speed}.mat.
        self.model = ["30204"]
        pass