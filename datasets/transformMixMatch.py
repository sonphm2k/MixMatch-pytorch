class TransformMixMatch(object):
    def __init__(self):
        self.transform1 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=188,
                                padding=int(112*0.125),
                                padding_mode='reflect'),
            transforms.Resize(size = (224,224)),
        ])

        self.transform2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=200,
                                padding=int(112*0.125),
                                padding_mode='reflect'),
            transforms.Resize(size = (224,224)),
            RandAugmentMC(n=2, m=10),
        ])
        self.transform3 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=224,
                                padding=int(112*0.125),
                                padding_mode='reflect'),
            transforms.Resize(size = (224,224)),
            RandAugmentMC(n=4, m=10),
        ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    def __call__(self, x):
        transform1 = self.transform1(x)
        transform2 = self.transform2(x)
        transform3 = self.transform3(x)

        return self.normalize(transform1), self.normalize(transform2), self.normalize(transform3)