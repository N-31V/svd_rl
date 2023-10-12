from typing import Type, Dict, Tuple, Optional, Callable
import collections
import enum
from functools import partial
import logging

import torch.nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LRScheduler
from fedot_ind.core.architecture.experiment.nn_experimenter import ClassificationExperimenter
from fedot_ind.core.operation.optimization.svd_tools import decompose_module, energy_svd_pruning
from fedot_ind.core.operation.decomposition.decomposed_conv import DecomposedConv2d
from fedot_ind.core.metrics.loss.svd_loss import HoyerLoss, OrthogonalLoss


State = collections.namedtuple('State', ['f1', 'size', 'epoch', 'decomposition', 'hoer_factor'])


class Actions(enum.Enum):
    train_compose = 0
    train_decompose = 1
    prune_99 = 2
    prune_9 = 3
    prune_7 = 4
    prune_5 = 5
    increase_hoer = 6
    decrease_hoer = 7


def _compose_layer(layer: DecomposedConv2d):
    layer.compose()


def _decompose_layer(layer: DecomposedConv2d, decomposing_mode='spatial'):
    layer.decompose(decomposing_mode=decomposing_mode)


def _layer_filer(layer: torch.nn.Module):
    return isinstance(layer, DecomposedConv2d)


class SVDEnv:
    def __init__(
            self,
            f1_baseline: float,
            train_ds: Dataset,
            val_ds: Dataset,
            model: Type[torch.nn.Module],
            model_params: Dict,
            dataloader_params: Dict,
            decomposing_mode: str,
            epochs: int,
            start_epoch: int,
            train_compose: bool,
            skip_impossible_steps: bool,
            size_factor: float,
            lr_scheduler: Optional[Callable],
            device: str = 'cuda'
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_f1 = f1_baseline
        self.train_ds: Dataset = train_ds
        self.val_ds: Dataset = val_ds
        self.dl_params: Dict = dataloader_params
        self.model: Type[torch.nn.Module] = model
        self.model_params: Dict = model_params
        self.decomposing_mode = decomposing_mode
        self.epochs = epochs
        self.start_epoch: int = start_epoch
        self.train_compose = train_compose
        self.skip = skip_impossible_steps
        self.size_factor = size_factor
        self.lr_scheduler: Optional[Callable] = lr_scheduler
        self.scheduler: Optional[LRScheduler] = None
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
        self.logger.info('Environment configured.')

    def get_state(self) -> State:
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
        model = self.model(**self.model_params)
        decompose_module(model=model, forward_mode='two_layers')
        self.decomposition = False
        self.exp = ClassificationExperimenter(model=model, device=self.device)
        self.epoch = 0
        self.reset_lr()
        self.base_params = self.exp.number_of_model_params()
        self.last_f1 = 0.
        self.last_params = 1
        while self.epoch < self.start_epoch:
            self._step()
        if not self.train_compose:
            self.decompose_model()
        return self.get_state()

    def reset_lr(self):
        self.optimizer = torch.optim.Adam(self.exp.model.parameters())
        if self.lr_scheduler is not None:
            self.scheduler = self.lr_scheduler(self.optimizer)

    def step(self, action: Actions) -> Tuple[State, float, bool]:
        if action == Actions.train_compose:
            if self.decomposition:
                self.compose_model()
            return self._step()

        if action == Actions.train_decompose:
            if not self.decomposition:
                self.decompose_model()
            return self._step(self.svd_loss)

        if self.decomposition:
            if action == Actions.prune_99:
                self.prune_model(e=0.99)
                return self._step(self.svd_loss)

            if action == Actions.prune_9:
                self.prune_model(e=0.9)
                return self._step(self.svd_loss)

            if action == Actions.prune_7:
                self.prune_model(e=0.7)
                return self._step(self.svd_loss)

            if action == Actions.prune_5:
                self.prune_model(e=0.5)
                return self._step(self.svd_loss)

            if action == Actions.increase_hoer:
                self.hoer_loss = HoyerLoss(factor=self.hoer_loss.factor*10)
                return self._step(self.svd_loss)

            if action == Actions.decrease_hoer:
                self.hoer_loss = HoyerLoss(factor=self.hoer_loss.factor/10)
                return self._step(self.svd_loss)

        else:
            if self.skip:
                self.logger.info('Impossible action, skipping a step.')
            else:
                self.epoch += 1
                self.logger.info('Impossible action, step lost.')
            return self.get_state(), 0, self.is_done()

    def _step(self, svd_loss=None) -> Tuple[State, float, bool]:
        self.epoch += 1
        train_score = self.exp.train_loop(
            dataloader=self.train_dl,
            optimizer=self.optimizer,
            model_losses=svd_loss
        )
        val_scores = self.exp.val_loop(dataloader=self.val_dl)
        if self.lr_scheduler is not None:
            self.scheduler.step()
        p_f1 = val_scores['f1'] / self.base_f1
        p_params = self.exp.number_of_model_params() / self.base_params
        d_f1 = p_f1 - self.last_f1
        d_params = self.last_params - p_params
        reward = float(d_f1 + self.size_factor * d_params)
        self.last_f1 = p_f1
        self.last_params = p_params
        return self.get_state(), reward, self.is_done()

    def prune_model(self, e: float):
        self.exp._apply_function(
            func=partial(energy_svd_pruning, energy_threshold=e),
            condition=_layer_filer
        )
        self.reset_lr()

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
