class DeepGrooveBall() :
    def __init__(self):
        super().__init__()
        self.rawpath = "raw_datasets/BearingType_DeepGrooveBall"
        #format: {condition}_{fault}_{samplingrate}_{model}_{speed}.mat.
        self.model = ["6204"]
        pass