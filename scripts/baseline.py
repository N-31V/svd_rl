"""Model training module to get baseline metric value."""
import os
import logging
from functools import partial
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor, Compose, RandomCrop, Normalize, RandomHorizontalFlip
from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter, FitParameters

logging.basicConfig(level=logging.INFO)
DATASETS_ROOT = '/media/n31v/data/datasets/'


if __name__ == "__main__":
    bs = 64
    lr = 0.05
    transform_train = Compose([
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    train_ds = CIFAR10(root=os.path.join(DATASETS_ROOT, 'CIFAR10'), transform=transform_train)
    val_ds = CIFAR10(root=os.path.join(DATASETS_ROOT, 'CIFAR10'), train=False, transform=transform_test)
    train_dl: DataLoader = DataLoader(dataset=train_ds, batch_size=bs, shuffle=True, num_workers=8)
    val_dl: DataLoader = DataLoader(dataset=val_ds, batch_size=bs, shuffle=False, num_workers=8)
    model = resnet18(num_classes=10)
    exp = ClassificationExperimenter(model=model)
    params = FitParameters(
        dataset_name='CIFAR10',
        train_dl=train_dl,
        val_dl=val_dl,
        num_epochs=250,
        optimizer=partial(SGD, lr=lr, momentum=0.9, weight_decay=5e-4),
        lr_scheduler=partial(CosineAnnealingLR, T_max=200),
        description=f'Augmented_lr{lr}_bs{bs}'
    )
    exp.fit(p=params)
