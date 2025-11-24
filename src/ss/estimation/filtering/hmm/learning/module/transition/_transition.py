from typing import Generic

import torch
from torch import nn

from ss.estimation.filtering.hmm.learning.module.filter.config import (
    FilterConfig,
)
from ss.estimation.filtering.hmm.learning.module.transition.config import (
    TransitionConfig,
)
from ss.utility.descriptor import BatchTensorDescriptor
from ss.utility.learning.module import BaseLearningModule
from ss.utility.learning.parameter.probability import ProbabilityParameter
from ss.utility.learning.parameter.transformer import T
from ss.utility.learning.parameter.transformer.config import TC
from ss.utility.logging import Logging

logger = Logging.get_logger(__name__)


class TransitionModule(
    BaseLearningModule[TransitionConfig[TC]], Generic[T, TC]
):
    def __init__(
        self,
        config: TransitionConfig[TC],
        filter_config: FilterConfig,
    ) -> None:
        super().__init__(config)
        self._state_dim = filter_config.state_dim
        self._initial_state: ProbabilityParameter[T, TC] = (
            ProbabilityParameter[T, TC](
                self._config.initial_state.probability_parameter,
                (self._state_dim,),
            )
        )
        self._matrix1 = ProbabilityParameter[T, TC](
            self._config.matrix.probability_parameter,
            (self._state_dim, self._state_dim),
        )
        self._matrix2 = ProbabilityParameter[T, TC](
            self._config.matrix.probability_parameter,
            (self._state_dim, self._state_dim, self._state_dim),
        )
        # self._eps = ProbabilityParameter[T, TC](
        #     self._config.matrix.probability_parameter,
        #     (self._state_dim, self._state_dim, 2),
        # )

        self._init_batch_size(batch_size=1)

    def _init_batch_size(
        self, batch_size: int, is_initialized: bool = False
    ) -> None:
        self._is_initialized = is_initialized
        self._batch_size = batch_size
        with self.evaluation_mode():
            self._estimated_state = self.initial_state.repeat(
                self._batch_size, 1
            )

    def _check_batch_size(self, batch_size: int) -> None:
        if self._is_initialized:
            assert batch_size == self._batch_size, (
                f"batch_size must be the same as the initialized batch_size. "
                f"batch_size given is {batch_size} while the "
                f"initialized batch_size is {self._batch_size}."
            )
            return
        self._init_batch_size(batch_size, is_initialized=True)

    estimated_state = BatchTensorDescriptor(
        "_batch_size",
        "_state_dim",
    )

    @property
    def initial_state_parameter(
        self,
    ) -> ProbabilityParameter[T, TC]:
        return self._initial_state

    @property
    def initial_state(self) -> torch.Tensor:
        initial_state: torch.Tensor = self._initial_state()
        return initial_state

    @initial_state.setter
    def initial_state(self, initial_state: torch.Tensor) -> None:
        self._initial_state.set_value(initial_state)

    # @property
    # def eps(
    #     self,
    # ) -> ProbabilityParameter[T, TC]:
    #     return self._eps

    # @property
    # def eps(self) -> torch.Tensor:
    #     eps: torch.Tensor = self._eps()
    #     return eps

    # @eps.setter
    # def eps(self, eps: torch.Tensor) -> None:
    #     self._eps.set_value(eps)

    @property
    def matrix_parameter1(
        self,
    ) -> ProbabilityParameter[T, TC]:
        return self._matrix1

    @property
    def matrix1(self) -> torch.Tensor:
        matrix1: torch.Tensor = self._matrix1()
        return matrix1

    @matrix1.setter
    def matrix1(self, matrix1: torch.Tensor) -> None:
        self._matrix1.set_value(matrix1)
    
    @property
    def matrix_parameter2(
        self,
    ) -> ProbabilityParameter[T, TC]:
        return self._matrix2

    @property
    def matrix2(self) -> torch.Tensor:
        matrix2: torch.Tensor = self._matrix2()
        return matrix2

    @matrix2.setter
    def matrix2(self, matrix2: torch.Tensor) -> None:
        self._matrix2.set_value(matrix2)

    @torch.compile
    def _prediction(
        self,
        estimated_state: torch.Tensor,
        transition_matrix1: torch.Tensor,
        transition_matrix2: torch.Tensor,
        # eps,
        k,
        state_dim,
        batch_size
    ) -> torch.Tensor:
        if k == 0:
            estimated_state = estimated_state.unsqueeze(2).expand(batch_size, state_dim, state_dim)
            predicted_next_state = estimated_state * transition_matrix1
        else:
            # f1 = torch.matmul(eps[:, :, 0], transition_matrix1.transpose(0, 1))
            # f2 = torch.diag(torch.matmul(eps[:, :, 1], transition_matrix2.transpose(0, 1))).unsqueeze(1).expand(state_dim, state_dim)
            # f = f1 + f2
            # estimated_state = estimated_state
            predicted_next_state = torch.sum(estimated_state.unsqueeze(2).expand(batch_size, state_dim, state_dim, state_dim).transpose(2, 3) * transition_matrix2, dim=1)
            # predicted_next_state = torch.diagonal(predicted_next_state, dim1=1, dim2=2)
        # print(torch.sum(predicted_next_state, dim=(1, 2)))
        return predicted_next_state

    @torch.compile
    def _update(
        self,
        prior_state: torch.Tensor,
        likelihood_state: torch.Tensor,
        k,
        state_dim,
        batch_size
    ) -> torch.Tensor:
        # update step based on likelihood_state (conditional probability)
        if k == 0:
            posterior_state = nn.functional.normalize(
                prior_state * likelihood_state,
                p=1,
                dim=1,
            )  # (batch_size, state_dim)
        else:
            likelihood_state_expanded = likelihood_state.unsqueeze(2).expand(batch_size, state_dim, state_dim).transpose(1, 2)
            posterior_state = nn.functional.normalize(
                prior_state * likelihood_state_expanded,
                p=1,
                dim=(1, 2),
            )  # (batch_size, state_dim, state_dim)
        return posterior_state

    def _process(
        self,
        estimated_state: torch.Tensor,
        likelihood_state: torch.Tensor,
        # eps,
        k,
        state_dim,
        batch_size
    ) -> torch.Tensor:

        # update step based on input_state (conditional probability)
        estimated_state = self._update(
            estimated_state, likelihood_state, k, state_dim, batch_size
        )  # (batch_size, state_dim, state_dim)

        # prediction step based on model process (predicted probability)
        estimated_state = self._prediction(
            estimated_state, self.matrix1, self.matrix2, k, state_dim, batch_size
        )  # (batch_size, state_dim, state_dim)

        return estimated_state

    def forward(self, emission_trajectory: torch.Tensor) -> torch.Tensor:

        batch_size, state_dim, horizon = emission_trajectory.shape
        # (batch_size, state_dim, horizon)

        estimated_state_trajectory = torch.empty(
            (batch_size, self._state_dim, horizon),
            device=emission_trajectory.device,
        )

        estimated_state = self.initial_state.repeat(batch_size, 1)
        # (batch_size, state_dim)

        # eps = self.eps

        for k in range(horizon):

            estimated_state = self._process(
                estimated_state,
                emission_trajectory[:, :, k],
                # eps,
                k,
                state_dim,
                batch_size
            )

            estimated_state_trajectory[:, :, k] = torch.sum(estimated_state, dim=1)

        return estimated_state_trajectory

    @torch.inference_mode()
    def at_inference(self, emission_trajectory: torch.Tensor) -> torch.Tensor:
        batch_size, state_dim, horizon = emission_trajectory.shape

        # eps = self.eps

        self._check_batch_size(batch_size)

        for k in range(horizon):

            self._estimated_state = torch.sum(self._process(
                self._estimated_state,
                emission_trajectory[:, :, k],
                # eps,
                k,
                state_dim,
                batch_size
            ), dim=1)

        return self.estimated_state

    def reset(self) -> None:
        self._init_batch_size(
            batch_size=self._batch_size, is_initialized=False
        )
