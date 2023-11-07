"""Model training module to get baseline metric value."""
import logging
from torch.utils.data import DataLoader
from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter, FitParameters
from fedot_ind.core.operation.optimization.structure_optimization import SVDOptimization

from svdtrainer.config import Config

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    config = Config('baseline', decomposing_mode='channel')
    train_dl: DataLoader = DataLoader(dataset=config.train_ds, shuffle=True, **config.dataloader_params)
    val_dl: DataLoader = DataLoader(dataset=config.val_ds, shuffle=False, **config.dataloader_params)
    model = config.model()
    exp = ClassificationExperimenter(model=model)
    optim = SVDOptimization(energy_thresholds=[0.9], decomposing_mode='spatial', forward_mode='two_layers')
    params = FitParameters(
        dataset_name='CIFAR10',
        train_dl=train_dl,
        val_dl=val_dl,
        num_epochs=config.epochs,
        optimizer=config.svd_optimizer,
        lr_scheduler=config.lr_scheduler,
        description='channel'
    )
    # exp.fit(p=params)
    optim.fit(exp=exp, params=params)
