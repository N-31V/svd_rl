import os

from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter, FitParameters


DATASETS_ROOT = '/media/n31v/data/datasets/'


train_ds = CIFAR10(root=os.path.join(DATASETS_ROOT, 'CIFAR10'), transform=ToTensor())
val_ds = CIFAR10(root=os.path.join(DATASETS_ROOT, 'CIFAR10'), train=False, transform=ToTensor())
train_dl: DataLoader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True, num_workers=8)
val_dl: DataLoader = DataLoader(dataset=val_ds, batch_size=32, shuffle=False, num_workers=8)
model = resnet18(num_classes=10)
exp = ClassificationExperimenter(model=model)
params = FitParameters(
    dataset_name='CIFAR10',
    train_dl=train_dl,
    val_dl=val_dl,
    num_epochs=30
)
exp.fit(p=params)
