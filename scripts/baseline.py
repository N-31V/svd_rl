"""Model training module to get baseline metric value."""
import logging
from functools import partial
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter, FitParameters

from svdtrainer.config import Config
from cv_models.resnet import resnet20

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    config = Config('baseline')
    train_dl: DataLoader = DataLoader(dataset=config.train_ds, shuffle=True, **config.dataloader_params)
    val_dl: DataLoader = DataLoader(dataset=config.val_ds, shuffle=False, **config.dataloader_params)
    model = resnet18(num_classes=10)
    exp = ClassificationExperimenter(model=model)
    params = FitParameters(
        dataset_name='CIFAR10',
        train_dl=train_dl,
        val_dl=val_dl,
        num_epochs=200,
        optimizer=partial(SGD, lr=0.05, momentum=0.9, weight_decay=5e-4),
        lr_scheduler=partial(CosineAnnealingLR, T_max=200),
        description='18_CosineAnnealingLR'
    )
    exp.fit(p=params)
