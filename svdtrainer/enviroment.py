from typing import Type, Dict, Tuple
import os
import enum
from functools import partial

import torch.nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter
from fedot_ind.core.architecture.datasets.splitters import train_test_split
from fedot_ind.core.operation.optimization.svd_tools import decompose_module, energy_svd_pruning
from fedot_ind.core.operation.decomposition.decomposed_conv import DecomposedConv2d
from fedot_ind.core.metrics.loss.svd_loss import HoyerLoss, OrthogonalLoss


DATASETS_ROOT = '/media/n31v/data/datasets/'


class Actions(enum.Enum):
    train_compose = 0
    train_decompose = 1
    prune_99 = 2
    prune_9 = 3
    prune_7 = 4
    prune_5 = 5
    increase_hoer = 6
    decrease_hoer = 7


class SVDEnv:
    def __init__(
            self,
            f1_baseline: float,
            train_ds: Dataset = CIFAR10(root=os.path.join(DATASETS_ROOT, 'CIFAR10'), transform=ToTensor()),
            val_ds: Dataset = CIFAR10(root=os.path.join(DATASETS_ROOT, 'CIFAR10'), train=False, transform=ToTensor()),
            model: Type[torch.nn.Module] = resnet18,
            epochs: int = 30,
            device: str = 'cuda'
    ) -> None:
        self.device = device
        self.train_dl: DataLoader = DataLoader(dataset=train_ds, batch_size=32, shuffle=True, num_workers=8)
        self.val_dl: DataLoader = DataLoader(dataset=val_ds, batch_size=32, shuffle=False, num_workers=8)
        self.model: Type[torch.nn.Module] = model
        self.exp: ClassificationExperimenter = None
        self.epoch: int = 0
        self.epochs = epochs
        self.optimizer: torch.optim.Optimizer = None
        self.decomposition: bool = False
        self.hoer_loss: HoyerLoss = HoyerLoss(factor=0.01)
        self.orthogonal_loss: OrthogonalLoss = OrthogonalLoss(factor=10)
        self.base_f1 = f1_baseline
        self.base_params: int = 0
        self.last_f1: float = 0.
        self.last_params: float = 1

    def state(self):
        return torch.Tensor([float(self.decomposition), self.epoch / self.epochs, self.last_f1, self.last_params])

    def reset(self):
        num_classes = len(self.train_dl.dataset.class_to_idx)
        model = self.model(num_classes=num_classes)
        decompose_module(model=model, forward_mode='two_layers')
        self.decomposition = False
        self.exp = ClassificationExperimenter(model=model, device=self.device)
        self.epoch = 0
        self.optimizer = torch.optim.Adam(self.exp.model.parameters())
        self.base_params = self.exp.number_of_model_params()
        self.last_f1 = 0.
        self.last_params = 1
        return self.state()

    def step(self, action):
        action = Actions(action)

        if action == Actions.train_compose:
            if self.decomposition:
                self.exp._apply_function(
                    func=self.compose_layer,
                    condition=self.layer_filer
                )
                self.decomposition = False
            return self.do_step()

        if action == Actions.train_decompose:
            if not self.decomposition:
                self.exp._apply_function(
                    func=self.decompose_layer,
                    condition=self.layer_filer
                )
                self.decomposition = True
            return self.do_step(self.svd_loss)

        if self.decomposition:
            if action == Actions.prune_99:
                self.prune_model(e=0.99)
                return self.do_step(self.svd_loss)

            if action == Actions.prune_9:
                self.prune_model(e=0.9)
                return self.do_step(self.svd_loss)

            if action == Actions.prune_7:
                self.prune_model(e=0.7)
                return self.do_step(self.svd_loss)

            if action == Actions.prune_5:
                self.prune_model(e=0.5)
                return self.do_step(self.svd_loss)

            if action == Actions.increase_hoer:
                self.hoer_loss = HoyerLoss(factor=self.hoer_loss.factor*10)
                return self.do_step(self.svd_loss)

            if action == Actions.decrease_hoer:
                self.hoer_loss = HoyerLoss(factor=self.hoer_loss.factor/10)
                return self.do_step(self.svd_loss)

        else:
            print("skip")
            return self.state(), -0.001, False

    def do_step(self, svd_loss=None) -> Tuple[torch.Tensor, float, bool]:
        self.epoch += 1
        train_score = self.exp.train_loop(
            dataloader=self.train_dl,
            optimizer=self.optimizer,
            model_losses=svd_loss
        )
        val_scores = self.exp.val_loop(dataloader=self.val_dl)
        p_f1 = val_scores['f1'] / self.base_f1
        p_params = self.exp.number_of_model_params() / self.base_params
        d_f1 = p_f1 - self.last_f1
        d_params = self.last_params - p_params
        self.last_f1 = p_f1
        self.last_params = p_params
        reward = float(d_f1 + 0.1 * d_params)
        done = self.epoch >= self.epochs
        return self.state(), reward, done

    def prune_model(self, e: float):
        self.exp._apply_function(
            func=partial(energy_svd_pruning, energy_threshold=e),
            condition=self.layer_filer
        )

    def svd_loss(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        losses = {
            'orthogonal_loss': self.orthogonal_loss(model),
            'hoer_loss': self.hoer_loss(model)
        }
        return losses

    @staticmethod
    def n_actions():
        return len(Actions)

    @staticmethod
    def compose_layer(layer: DecomposedConv2d):
        layer.compose()

    @staticmethod
    def decompose_layer(layer: DecomposedConv2d, decomposing_mode='spatial'):
        layer.decompose(decomposing_mode=decomposing_mode)

    @staticmethod
    def layer_filer(layer: torch.nn.Module):
        return isinstance(layer, DecomposedConv2d)
