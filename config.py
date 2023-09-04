class Config(object):
    def __init__(self):
        self.epochs = 10
        self.batch_size = 8
        self.num_classes = 10
        # -----------------
        self.lambda_u = 30 # Hyper-parameter weighting the contribution of the unlabeled examples to the training loss
        self.alpha = 0.75  # Hyper-parameter for the Beta distribution used in MixU
        self.T = 0.5       # Temperature parameter for sharpening used in MixMatch
        self.K = 3         # Number of augmentations used when guessing labels in MixMatch
        # -----------------
        self.lr = 10e-6
        self.optim = "ADAM"  # type of optimizer [Adam, SGD]
        self.ema_decay = 0.999


