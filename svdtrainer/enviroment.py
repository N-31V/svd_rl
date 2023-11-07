from typing import Type, Dict, Tuple, Optional, Union
from functools import partial
import enum
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


class DecomposingMods(enum.Enum):
    """List of decomposing mods."""
    composed = 0
    channel = 1
    spatial = 2


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
            f1_baseline: float,
            max_steps: int,
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

        self.base_f1: float = f1_baseline
        self.max_steps: int = max_steps

        self.optimizer_type: Type[torch.optim.Optimizer] = optimizer
        self.lr_scheduler_type: Optional[Type[LRScheduler]] = lr_scheduler

        self.device = device

        self.hoer_loss: HoyerLoss = HoyerLoss(factor=0.1)
        self.orthogonal_loss: OrthogonalLoss = OrthogonalLoss(factor=10)
        self.decomposition: DecomposingMods = DecomposingMods.composed

        self.step: int = 0
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
        val_scores = self.exp.val_loop(dataloader=self.val_dl)
        self.last_f1 = val_scores['f1']
        self.last_params = self.exp.number_of_model_params()
        state = State(
            f1=self.last_f1 / self.base_f1,
            size=self.last_params / self.base_params,
            step=self.step / self.max_steps,
            decomposition=float(self.decomposition.value),
            hoer_factor=self.hoer_loss.factor
        )
        return state

    def is_done(self) -> bool:
        return self.step >= self.max_steps

    def reset(self) -> State:
        self.logger.info('Resetting environment...')
        self.train_dl = DataLoader(dataset=self.train_ds, shuffle=True, **self.dl_params)
        self.val_dl = DataLoader(dataset=self.val_ds, shuffle=False, **self.dl_params)

        model = self.model()
        decompose_module(model=model, forward_mode='two_layers')
        self.decomposition = DecomposingMods.composed
        self.exp = ClassificationExperimenter(model=model, weights=self.weights, device=self.device)
        self.base_params = self.exp.number_of_model_params()
        self.step = 0

        self.optimizer = self.optimizer_type(self.exp.model.parameters())
        if self.lr_scheduler_type is not None:
            self.lr_scheduler = self.lr_scheduler_type(self.optimizer)

        return self.get_state()

    def do_step(self, action: Actions) -> Tuple[State, bool]:
        self.step += 1

        if action == Actions.stop:
            return self.get_state(), True

        elif action == Actions.train:
            self.train()

        elif self.decomposition == DecomposingMods.composed:
            if action == Actions.channel:
                self.decompose_model(DecomposingMods.channel)
            elif action == Actions.spatial:
                self.decompose_model(DecomposingMods.spatial)

        else:
            if action == Actions.prune_999:
                self.prune_model(e=0.999)
            elif action == Actions.prune_99:
                self.prune_model(e=0.99)
            elif action == Actions.prune_9:
                self.prune_model(e=0.9)
            elif action == Actions.prune_7:
                self.prune_model(e=0.7)

        return self.get_state(), self.is_done()

    def train(self) -> None:
        train_score = self.exp.train_loop(
            dataloader=self.train_dl,
            optimizer=self.optimizer,
            model_losses=self.svd_loss if self.decomposition.value > 0 else None
        )
        if self.lr_scheduler_type is not None:
            self.lr_scheduler.step()

    def prune_model(self, e: float):
        self.exp._apply_function(
            func=partial(energy_svd_pruning, energy_threshold=e),
            condition=_layer_filer
        )
        self.logger.info(f'Pruned size: {self.exp.size_of_model():.2f} MB.')

    def decompose_model(self, decomposing_mode: DecomposingMods):
        self.logger.info(f'Default size: {self.exp.size_of_model():.2f} MB.')
        self.exp._apply_function(
            func=partial(_decompose_layer, decomposing_mode=decomposing_mode.name),
            condition=_layer_filer
        )
        self.decomposition = decomposing_mode
        self.logger.info(f'Decomposed size: {self.exp.size_of_model():.2f} MB.')

    def svd_loss(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        losses = {
            'orthogonal_loss': self.orthogonal_loss(model),
            'hoer_loss': self.hoer_loss(model)
        }
        return losses
