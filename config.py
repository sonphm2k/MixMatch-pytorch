class Config(object):
    def
configs = {
    "epochs": 1,
    "batch_size": 2,
    "num_classes": 10,
    # -----------------
    "lr": 10e-6,
    "optim": "ADAM",  # type of optimizer [Adam, SGD]
    "lambda_u": 30,
    # 75       Hyper-parameter weighting the contribution of the unlabeled examples to the training loss
    "alpha": 0.75,  # 0.75     Hyperparameter for the Beta distribution used in MixU
    "T": 0.5,  # 0.5      Temperature parameter for sharpening used in MixMatch
    "K": 3,  # 3        Number of augmentations used when guessing labels in MixMatch
    "ema_decay": 0.999,
}