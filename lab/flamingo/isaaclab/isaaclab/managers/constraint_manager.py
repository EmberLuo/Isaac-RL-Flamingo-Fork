# constraint_manager.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Constraint manager for computing done signals for a given world."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from prettytable import PrettyTable
from typing import TYPE_CHECKING

from isaaclab.managers.manager_base import ManagerBase, ManagerTermBase
from .constraint_term_cfg import ConstraintTermCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class ConstraintManager(ManagerBase):
    """Manager for computing done signals for a given world.

    The constraint manager computes the termination signal (also called dones) as a combination
    of constraint terms. Each constraint term is a function which takes the environment as an
    argument and returns a boolean tensor of shape (num_envs,). The constraint manager
    computes the termination signal as the union (logical or) of all the constraint terms.

    Following the `Gymnasium API <https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/>`_,
    the termination signal is computed as the logical OR of the following signals:

    * **Time-out**: This signal is set to true if the environment has ended after an externally defined condition
      (that is outside the scope of a MDP). For example, the environment may be terminated if the episode has
      timed out (i.e. reached max episode length).
    * **Terminated**: This signal is set to true if the environment has reached a terminal state defined by the
      environment. This state may correspond to task success, task failure, robot falling, etc.

    These signals can be individually accessed using the :attr:`time_outs` and :attr:`terminated` properties.

    The constraint terms are parsed from a config class containing the manager's settings and each term's
    parameters. Each constraint term should instantiate the :class:`ConstraintTermCfg` class. The term's
    configuration :attr:`ConstraintTermCfg.time_out` decides whether the term is a timeout or a constraint term.
    """

    _env: ManagerBasedRLEnv
    """The environment instance."""

    def __init__(self, cfg: object, env: ManagerBasedRLEnv):
        """Initializes the constraint manager.

        Args:
            cfg: The configuration object or dictionary (``dict[str, ConstraintTermCfg]``).
            env: An environment object.
        """
        # create buffers to parse and store terms
        self._term_names: list[str] = list()
        self._term_cfgs: list[ConstraintTermCfg] = list()
        self._class_term_cfgs: list[ConstraintTermCfg] = list()

        # call the base class constructor (this will parse the terms config)
        super().__init__(cfg, env)
        self._term_name_to_term_idx = {name: i for i, name in enumerate(self._term_names)}
        # prepare extra info to store individual constraint term information
        self._term_dones = torch.zeros((self.num_envs, len(self._term_names)), device=self.device, dtype=torch.bool)
        # create buffer for managing termination per environment
        self._truncated_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._terminated_buf = torch.zeros_like(self._truncated_buf)

    def __str__(self) -> str:
        """Returns: A string representation for constraint manager."""
        msg = f"<ConstraintManager> contains {len(self._term_names)} active terms.\n"

        # create table for term information
        table = PrettyTable()
        table.title = "Active Constraint Terms"
        table.field_names = ["Index", "Name", "Time Out"]
        # set alignment of table columns
        table.align["Name"] = "l"
        # add info on each term
        for index, (name, term_cfg) in enumerate(zip(self._term_names, self._term_cfgs)):
            table.add_row([index, name, term_cfg.time_out])
        # convert table to string
        msg += table.get_string()
        msg += "\n"

        return msg

    """
    Properties.
    """

    @property
    def active_terms(self) -> list[str]:
        """Name of active constraint terms."""
        return self._term_names

    @property
    def dones(self) -> torch.Tensor:
        """The net termination signal. Shape is (num_envs,)."""
        return self._truncated_buf | self._terminated_buf

    @property
    def time_outs(self) -> torch.Tensor:
        """The timeout signal (reaching max episode length). Shape is (num_envs,).

        This signal is set to true if the environment has ended after an externally defined condition
        (that is outside the scope of a MDP). For example, the environment may be terminated if the episode has
        timed out (i.e. reached max episode length).
        """
        return self._truncated_buf

    @property
    def terminated(self) -> torch.Tensor:
        """The terminated signal (reaching a terminal state). Shape is (num_envs,).

        This signal is set to true if the environment has reached a terminal state defined by the environment.
        This state may correspond to task success, task failure, robot falling, etc.
        """
        return self._terminated_buf

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        """Returns the episodic counts of individual constraint terms.

        Args:
            env_ids: The environment ids. Defaults to None, in which case
                all environments are considered.

        Returns:
            Dictionary of episodic sum of individual reward terms.
        """
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        # add to episode dict
        extras = {}
        last_episode_done_stats = self._term_dones.float().mean(dim=0)
        for i, key in enumerate(self._term_names):
            # store information
            extras["Episode_Constraint/" + key] = last_episode_done_stats[i].item()
        # reset all the reward terms
        for term_cfg in self._class_term_cfgs:
            term_cfg.func.reset(env_ids=env_ids)
        # return logged information
        return extras

    def compute(self) -> torch.Tensor:
        """Computes the termination signal as union of individual terms.

        This function calls each constraint term managed by the class and performs a logical OR operation
        to compute the net termination signal.

        Returns:
            The combined termination signal of shape (num_envs,).
        """
        # reset computation
        self._truncated_buf[:] = False
        self._terminated_buf[:] = False
        # iterate over all the constraint terms
        for i, term_cfg in enumerate(self._term_cfgs):
            value = term_cfg.func(self._env, **term_cfg.params)
            
            # 确保value是tensor类型，如果它是布尔值，则转换为tensor
            if not hasattr(value, 'nonzero'):
                # 如果value不是tensor而是单个布尔值，将其转换为合适的tensor
                if isinstance(value, (bool, int, float)):
                    value = torch.full((self.num_envs,), value, device=self.device, dtype=torch.bool)
                elif not torch.is_tensor(value):
                    # 如果value是其他类型的tensor-like对象，转换为tensor
                    value = torch.tensor(value, device=self.device, dtype=torch.bool)
            
            # store timeout signal separately
            if term_cfg.time_out:
                self._truncated_buf |= value
            else:
                self._terminated_buf |= value
            # add to episode dones
            rows = value.nonzero(as_tuple=True)[0]  # indexing is cheaper than boolean advance indexing
            if rows.numel() > 0:
                self._term_dones[rows] = False
                self._term_dones[rows, i] = True
        # return combined termination signal
        return self._truncated_buf | self._terminated_buf

    def get_term(self, name: str) -> torch.Tensor:
        """Returns the constraint term with the specified name.

        Args:
            name: The name of the constraint term.

        Returns:
            The corresponding constraint term value. Shape is (num_envs,).
        """
        return self._term_dones[:, self._term_name_to_term_idx[name]]

    def get_active_iterable_terms(self, env_idx: int) -> Sequence[tuple[str, Sequence[float]]]:
        """Returns the active terms as iterable sequence of tuples.

        The first element of the tuple is the name of the term and the second element is the raw value(s) of the term.

        Args:
            env_idx: The specific environment to pull the active terms from.

        Returns:
            The active terms.
        """
        terms = []
        for i, key in enumerate(self._term_names):
            terms.append((key, [self._term_dones[env_idx, i].float().cpu().item()]))
        return terms

    """
    Operations - Term settings.
    """

    def set_term_cfg(self, term_name: str, cfg: ConstraintTermCfg):
        """Sets the configuration of the specified term into the manager.

        Args:
            term_name: The name of the constraint term.
            cfg: The configuration for the constraint term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Constraint term '{term_name}' not found.")
        # set the configuration
        self._term_cfgs[self._term_name_to_term_idx[term_name]] = cfg

    def get_term_cfg(self, term_name: str) -> ConstraintTermCfg:
        """Gets the configuration for the specified term.

        Args:
            term_name: The name of the constraint term.

        Returns:
            The configuration of the constraint term.

        Raises:
            ValueError: If the term name is not found.
        """
        if term_name not in self._term_names:
            raise ValueError(f"Constraint term '{term_name}' not found.")
        # return the configuration
        return self._term_cfgs[self._term_name_to_term_idx[term_name]]

    """
    Helper functions.
    """

    def _prepare_terms(self):
        # check if config is dict already
        if isinstance(self.cfg, dict):
            cfg_items = self.cfg.items()
        else:
            cfg_items = self.cfg.__dict__.items()
        # iterate over all the terms
        for term_name, term_cfg in cfg_items:
            # check for non config
            if term_cfg is None:
                continue
            # check for valid config type
            if not isinstance(term_cfg, ConstraintTermCfg):
                raise TypeError(
                    f"Configuration for the term '{term_name}' is not of type ConstraintTermCfg."
                    f" Received: '{type(term_cfg)}'."
                )
            # resolve common parameters
            self._resolve_common_term_cfg(term_name, term_cfg, min_argc=1)
            # add function to list
            self._term_names.append(term_name)
            self._term_cfgs.append(term_cfg)
            # check if the term is a class
            if isinstance(term_cfg.func, ManagerTermBase):
                self._class_term_cfgs.append(term_cfg)