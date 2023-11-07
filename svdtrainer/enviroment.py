from typing import Type, Dict, Tuple, Optional, Union
from functools import partial
import logging

import torch.nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter
from fedot_ind.core.operation.optimization.svd_tools import decompose_module, energy_svd_pruning
from fedot_ind.core.operation.decomposition.decomposed_conv import DecomposedConv2d
from fedot_ind.core.metrics.loss.svd_loss import HoyerLoss, OrthogonalLoss

from svdtrainer.state import State
from svdtrainer.actions import Actions


def _compose_layer(layer: DecomposedConv2d):
    layer.compose()


def _decompose_layer(layer: DecomposedConv2d, decomposing_mode: str):
    layer.decompose(decomposing_mode=decomposing_mode)


def _layer_filer(layer: torch.nn.Module):
    return isinstance(layer, DecomposedConv2d)


class SVDEnv:
    def __init__(
            self,
            train_ds: Dataset,
            val_ds: Dataset,
            dataloader_params: Dict,
            model: Union[Type[torch.nn.Module], partial],
            weights: Optional[str],
            decomposing_mode: str,
            f1_baseline: float,
            epochs: int,
            start_epoch: int,
            train_compose: bool,
            optimizer: Union[Type[torch.optim.Optimizer], partial],
            lr_scheduler: Optional[Union[Type[LRScheduler], partial]],
            device: str = 'cuda'
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.train_ds: Dataset = train_ds
        self.val_ds: Dataset = val_ds
        self.dl_params: Dict = dataloader_params

        self.model: Type[torch.nn.Module] = model
        self.weights = weights
        self.decomposing_mode = decomposing_mode

        self.base_f1: float = f1_baseline
        self.epochs: int = epochs
        self.start_epoch: int = start_epoch
        self.train_compose: bool = train_compose

        self.optimizer_type: Type[torch.optim.Optimizer] = optimizer
        self.lr_scheduler_type: Optional[Type[LRScheduler]] = lr_scheduler

        self.device = device

        self.hoer_loss: HoyerLoss = HoyerLoss(factor=0.1)
        self.orthogonal_loss: OrthogonalLoss = OrthogonalLoss(factor=10)
        self.decomposition: bool = False
        self.epoch: int = 0
        self.base_params: int = 0
        self.last_f1: float = 0.
        self.last_params: float = 1

        self.train_dl: DataLoader = None
        self.val_dl: DataLoader = None
        self.exp: ClassificationExperimenter = None
        self.optimizer: torch.optim.Optimizer = None
        self.lr_scheduler: Optional[LRScheduler] = None
        self.logger.info('Environment configured.')

    def get_state(self) -> State:
        """Returns current state."""
        state = State(
            f1=self.last_f1,
            size=self.last_params,
            epoch=self.epoch / self.epochs,
            decomposition=float(self.decomposition),
            hoer_factor=self.hoer_loss.factor
        )
        return state

    def is_done(self) -> bool:
        return self.epoch >= self.epochs

    def reset(self) -> State:
        self.logger.info('Resetting environment...')
        self.train_dl = DataLoader(dataset=self.train_ds, shuffle=True, **self.dl_params)
        self.val_dl = DataLoader(dataset=self.val_ds, shuffle=False, **self.dl_params)

        model = self.model()
        decompose_module(model=model, forward_mode='two_layers')
        self.decomposition = False
        self.exp = ClassificationExperimenter(model=model, weights=self.weights, device=self.device)
        self.epoch = 0
        self.reset_optimizer()
        self.base_params = self.exp.number_of_model_params()
        self.update_state()
        while self.epoch < self.start_epoch:
            self._step()
        if not self.train_compose:
            self.decompose_model()
        return self.get_state()

    def reset_optimizer(self):
        self.optimizer = self.optimizer_type(self.exp.model.parameters())
        if self.lr_scheduler_type is not None:
            self.lr_scheduler = self.lr_scheduler_type(self.optimizer)

    def step(self, action: Actions) -> Tuple[State, bool]:
        if action == Actions.train_compose:
            if self.decomposition:
                self.compose_model()
            return self._step()

        if action == Actions.train_decompose:
            if not self.decomposition:
                self.decompose_model()
            return self._step()

        if self.decomposition:
            if action == Actions.prune_99:
                self.prune_model(e=0.99)
            elif action == Actions.prune_9:
                self.prune_model(e=0.9)
            elif action == Actions.prune_7:
                self.prune_model(e=0.7)
            elif action == Actions.prune_5:
                self.prune_model(e=0.5)
            elif action == Actions.increase_hoer:
                self.hoer_loss = HoyerLoss(factor=self.hoer_loss.factor*10)
            elif action == Actions.decrease_hoer:
                self.hoer_loss = HoyerLoss(factor=self.hoer_loss.factor/10)
            return self._step()

        else:
            self.epoch += 1
            self.logger.info('Impossible action, step lost.')
            return self.get_state(), self.is_done()

    def _step(self) -> Tuple[State, bool]:
        self.epoch += 1
        train_score = self.exp.train_loop(
            dataloader=self.train_dl,
            optimizer=self.optimizer,
            model_losses=self.svd_loss if self.decomposition else None
        )
        self.update_state()
        if self.lr_scheduler_type is not None:
            self.lr_scheduler.step()
        return self.get_state(), self.is_done()

    def update_state(self):
        val_scores = self.exp.val_loop(dataloader=self.val_dl)
        self.last_f1 = val_scores['f1'] / self.base_f1
        self.last_params = self.exp.number_of_model_params() / self.base_params

    def prune_model(self, e: float):
        self.exp._apply_function(
            func=partial(energy_svd_pruning, energy_threshold=e),
            condition=_layer_filer
        )
        # self.reset_optimizer()

    def decompose_model(self):
        self.exp._apply_function(
            func=partial(_decompose_layer, decomposing_mode=self.decomposing_mode),
            condition=_layer_filer
        )
        self.decomposition = True

    def compose_model(self):
        self.exp._apply_function(
            func=_compose_layer,
            condition=_layer_filer
        )
        self.decomposition = False

    def svd_loss(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        losses = {
            'orthogonal_loss': self.orthogonal_loss(model),
            'hoer_loss': self.hoer_loss(model)
        }
        return losses
