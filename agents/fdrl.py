import copy
from functools import partial
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ValueVectorField, ActorVectorField


class FDRLAgent(flax.struct.PyTreeNode):
    """Flow distributional reinforcement learning (FDRL) agent."""

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    def critic_loss(self, batch, grad_params, rng):
        """Compute the flow critic loss. (Q-learning)"""
        if self.config['critic_loss_type'] == 'sarsa':
            batch_size = batch['actions'].shape[0]
            rng, actor_rng, noise_rng, time_rng, q_rng = jax.random.split(rng, 5)

            noises = jax.random.normal(noise_rng, (batch_size, 1))
            times = jax.random.uniform(time_rng, (batch_size, 1))
            next_returns1 = self.compute_flow_returns(
                noises, batch['next_observations'], batch['value_goals'], batch['next_actions'],
                flow_network_name='target_critic_flow1')
            next_returns2 = self.compute_flow_returns(
                noises, batch['next_observations'], batch['value_goals'], batch['next_actions'],
                flow_network_name='target_critic_flow2')
            next_returns = jnp.minimum(next_returns1, next_returns2)
            # The following returns will be bounded automatically
            returns = (jnp.expand_dims(batch['rewards'], axis=-1) +
                       self.config['discount'] * jnp.expand_dims(batch['masks'], axis=-1) * next_returns)
            noisy_returns = times * returns + (1 - times) * noises
            target_vector_field = returns - noises

            vector_field1 = self.network.select('critic_flow1')(
                noisy_returns, times,
                batch['observations'], batch['value_goals'], batch['actions'], params=grad_params)
            vector_field2 = self.network.select('critic_flow2')(
                noisy_returns, times,
                batch['observations'], batch['value_goals'], batch['actions'], params=grad_params)
            vector_field_loss = ((vector_field1 - target_vector_field) ** 2 +
                                 (vector_field2 - target_vector_field) ** 2).mean()

            noisy_next_returns1 = self.compute_flow_returns(
                noises, batch['next_observations'], batch['value_goals'], batch['next_actions'], end_times=times,
                flow_network_name='target_critic_flow1')
            noisy_next_returns2 = self.compute_flow_returns(
                noises, batch['next_observations'], batch['value_goals'], batch['next_actions'], end_times=times,
                flow_network_name='target_critic_flow2')
            noisy_next_returns = jnp.minimum(noisy_next_returns1, noisy_next_returns2)
            transformed_noisy_returns = (
                jnp.expand_dims(batch['rewards'], axis=-1) +
                self.config['discount'] * jnp.expand_dims(batch['masks'], axis=-1) * noisy_next_returns
            )
            bootstrapped_vector_field1 = self.network.select('critic_flow1')(
                transformed_noisy_returns, times,
                batch['observations'], batch['value_goals'], batch['actions'], params=grad_params)
            bootstrapped_vector_field2 = self.network.select('critic_flow2')(
                transformed_noisy_returns, times,
                batch['observations'], batch['value_goals'], batch['actions'], params=grad_params)
            target_bootstrapped_vector_field1 = self.network.select('target_critic_flow1')(
                noisy_next_returns, times,
                batch['next_observations'], batch['value_goals'], batch['next_actions'])
            target_bootstrapped_vector_field2 = self.network.select('target_critic_flow2')(
                noisy_next_returns, times,
                batch['next_observations'], batch['value_goals'], batch['next_actions'])
            target_bootstrapped_vector_field = jnp.minimum(target_bootstrapped_vector_field1,
                                                           target_bootstrapped_vector_field2)
            bootstrapped_vector_field_loss = ((bootstrapped_vector_field1 - target_bootstrapped_vector_field) ** 2 +
                                              (bootstrapped_vector_field2 - target_bootstrapped_vector_field) ** 2).mean()

            # Additional metrics for logging.
            q_noises = jax.random.normal(q_rng, (batch_size, 1))
            q1 = (q_noises + self.network.select('critic_flow1')(
                q_noises, jnp.zeros_like(q_noises),
                batch['observations'], batch['value_goals'], batch['actions'])).squeeze(-1)
            q2 = (q_noises + self.network.select('critic_flow2')(
                q_noises, jnp.zeros_like(q_noises),
                batch['observations'], batch['value_goals'], batch['actions'])).squeeze(-1)
            q = jnp.minimum(q1, q2)

            critic_loss = vector_field_loss + self.config['alpha_critic'] * bootstrapped_vector_field_loss

        return critic_loss, {
            'vector_field_loss': vector_field_loss,
            'bootstrapped_vector_field_loss': bootstrapped_vector_field_loss,
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def actor_loss(self, batch, grad_params, rng):
        """Compute the FQL actor loss."""
        batch_size, action_dim = batch['actions'].shape
        rng, x_rng, t_rng, q_rng, actor_rng = jax.random.split(rng, 5)

        # BC flow loss.
        x_0 = jax.random.normal(x_rng, (batch_size, action_dim))
        x_1 = batch['actions']
        t = jax.random.uniform(t_rng, (batch_size, 1))
        x_t = (1 - t) * x_0 + t * x_1
        vel = x_1 - x_0

        pred = self.network.select('actor_flow')(
            batch['observations'], goals=batch['actor_goals'], actions=x_t, times=t, params=grad_params)
        bc_flow_loss = jnp.mean((pred - vel) ** 2)

        # Distillation loss.
        rng, noise_rng = jax.random.split(rng)
        noises = jax.random.normal(noise_rng, (batch_size, action_dim))
        target_flow_actions = self.compute_flow_actions(
            noises, batch['observations'], batch['actor_goals'])
        actor_actions = self.network.select('actor_onestep_flow')(
            batch['observations'], batch['actor_goals'], noises, params=grad_params)
        actor_actions = jnp.clip(actor_actions, -1, 1)
        distill_loss = jnp.mean((actor_actions - target_flow_actions) ** 2)

        # Q loss.
        q_noises, q_noises = jax.random.normal(q_rng, (2, batch_size, 1))
        q1 = (q_noises + self.network.select('critic_flow1')(
            q_noises, jnp.zeros_like(q_noises),
            batch['observations'], batch['actor_goals'], actor_actions)).squeeze(-1)
        q2 = (q_noises + self.network.select('critic_flow2')(
            q_noises, jnp.zeros_like(q_noises),
            batch['observations'], batch['actor_goals'], actor_actions)).squeeze(-1)
        q = jnp.minimum(q1, q2)

        q_loss = -q.mean()
        if self.config['normalize_q_loss']:
            lam = jax.lax.stop_gradient(1 / jnp.abs(q).mean())
            q_loss = lam * q_loss

        # Total loss.
        actor_loss = bc_flow_loss + self.config['alpha_actor'] * distill_loss + q_loss

        # Additional metrics for logging.
        actor_noises = jax.random.normal(actor_rng, (batch_size, action_dim))
        actions = self.network.select('actor_onestep_flow')(
            batch['observations'], batch['actor_goals'], actor_noises)
        mse = jnp.mean((actions - batch['actions']) ** 2)

        return actor_loss, {
            'actor_loss': actor_loss,
            'bc_flow_loss': bc_flow_loss,
            'distill_loss': distill_loss,
            'q_loss': q_loss,
            'q': q.mean(),
            'mse': mse,
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng
        rng, value_rng, critic_rng, actor_rng = jax.random.split(rng, 4)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic_flow1')
        self.target_update(new_network, 'critic_flow2')

        return self.replace(network=new_network, rng=new_rng), info

    @partial(jax.jit, static_argnames=('flow_network_name',))
    def compute_flow_returns(
        self,
        noises,
        observations,
        goals,
        actions,
        init_times=None,
        end_times=None,
        flow_network_name='critic_flow',
    ):
        """Compute returns from the return flow model using the Euler method."""
        noisy_returns = noises
        if init_times is None:
            init_times = jnp.zeros((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        if end_times is None:
            end_times = jnp.ones((*noisy_returns.shape[:-1], 1), dtype=noisy_returns.dtype)
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def func(carry, i):
            """
            carry: (noisy_goals, )
            i: current step index
            """
            (noisy_returns, ) = carry

            times = i * step_size + init_times
            if self.config['ode_solver'] == 'euler':
                vector_field = self.network.select(flow_network_name)(
                    noisy_returns, times, observations, goals, actions)
                new_noisy_returns = noisy_returns + step_size * vector_field
            elif self.config['ode_solver'] == 'midpoint':
                vector_field = self.network.select(flow_network_name)(
                    noisy_returns, times, observations, goals, actions)

                mid_noisy_returns = noisy_returns + 0.5 * step_size * vector_field
                mid_times = times + 0.5 * step_size

                vector_field = self.network.select(flow_network_name)(
                    mid_noisy_returns, mid_times, observations, goals, actions)

                new_noisy_returns = noisy_returns + step_size * vector_field
            else:
                raise NotImplementedError
            new_noisy_returns = jnp.clip(
                new_noisy_returns,
                self.config['min_reward'] / (1 - self.config['discount']),
                self.config['max_reward'] / (1 - self.config['discount']),
            )

            return (new_noisy_returns, ), None

        # Use lax.scan to do the iteration
        (noisy_returns, ), _ = jax.lax.scan(
            func, (noisy_returns,), jnp.arange(self.config['num_flow_steps']))
        noisy_returns = jnp.clip(
            noisy_returns,
            self.config['min_reward'] / (1 - self.config['discount']),
            self.config['max_reward'] / (1 - self.config['discount']),
        )

        return noisy_returns

    @jax.jit
    def compute_flow_actions(
        self,
        noises,
        observations,
        goals,
        init_times=None,
        end_times=None,
    ):
        noisy_actions = noises
        if init_times is None:
            init_times = jnp.zeros((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype)
        if end_times is None:
            end_times = jnp.ones((*noisy_actions.shape[:-1], 1), dtype=noisy_actions.dtype)
        step_size = (end_times - init_times) / self.config['num_flow_steps']

        def func(carry, i):
            """
            carry: (noisy_goals, )
            i: current step index
            """
            (noisy_actions, ) = carry

            times = i * step_size + init_times
            vector_field = self.network.select('actor_flow')(
                observations, goals, noisy_actions, times)
            new_noisy_actions = noisy_actions + vector_field * step_size
            new_noisy_actions = jnp.clip(new_noisy_actions, -1, 1)

            return (new_noisy_actions, ), None

        # Use lax.scan to do the iteration
        (noisy_actions, ), _ = jax.lax.scan(
            func, (noisy_actions,), jnp.arange(self.config['num_flow_steps']))

        noisy_actions = jnp.clip(noisy_actions, -1, 1)

        return noisy_actions

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the one-step policy."""
        action_seed, noise_seed = jax.random.split(seed)
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                self.config['action_dim'],
            ),
        )
        actions = self.network.select('actor_onestep_flow')(observations, goals, noises)
        actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        example_batch,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            example_batch: Example batch.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_observations = example_batch['observations']
        ex_actions = example_batch['actions']
        ex_goals = example_batch['value_goals']
        ex_returns = ex_actions[..., :1]
        ex_times = ex_actions[..., :1]
        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        min_reward = example_batch['min_reward']
        max_reward = example_batch['max_reward']

        # Define networks.
        critic_flow1_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
        )
        critic_flow2_def = ValueVectorField(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['value_layer_norm'],
        )
        actor_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
        )
        actor_onestep_flow_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
        )

        network_info = dict(
            critic_flow1=(critic_flow1_def, (ex_returns, ex_times, ex_observations, ex_goals, ex_actions)),
            critic_flow2=(critic_flow2_def, (ex_returns, ex_times, ex_observations, ex_goals, ex_actions)),
            target_critic_flow1=(copy.deepcopy(critic_flow1_def),
                                 (ex_returns, ex_times, ex_observations, ex_goals, ex_actions)),
            target_critic_flow2=(copy.deepcopy(critic_flow2_def),
                                 (ex_returns, ex_times, ex_observations, ex_goals, ex_actions)),
            actor_flow=(actor_flow_def, (ex_observations, ex_goals, ex_actions, ex_times)),
            actor_onestep_flow=(actor_onestep_flow_def, (ex_observations, ex_goals, ex_actions)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_critic_flow1'] = params['modules_critic_flow1']
        params['modules_target_critic_flow2'] = params['modules_critic_flow2']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['min_reward'] = min_reward
        config['max_reward'] = max_reward
        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='fdrl',  # Agent name.
            ob_dims=ml_collections.config_dict.placeholder(list),  # Observation dimensions (will be set automatically).
            action_dim=ml_collections.config_dict.placeholder(int),  # Action dimension (will be set automatically).
            min_reward=ml_collections.config_dict.placeholder(float),  # Minimum reward (will be set automatically).
            max_reward=ml_collections.config_dict.placeholder(float),  # Maximum reward (will be set automatically).
            lr=3e-4,  # Learning rate.
            batch_size=1024,  # Batch size.
            actor_hidden_dims=(1024, 1024, 1024, 1024),  # Actor network hidden dimensions.
            value_hidden_dims=(1024, 1024, 1024, 1024),  # Value network hidden dimensions.
            actor_layer_norm=False,  # Whether to use layer normalization for the actor.
            value_layer_norm=True,  # Whether to use layer normalization for the value and the critic.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            expectile=0.9,  # IQL expectile.
            critic_loss_type='sarsa',  # Type of the critic loss ('sarsa', 'q-learning').
            alpha_critic=1.0,  # vector field bootstrapped regularization coefficient.
            alpha_actor=1.0,  # BC coefficient.
            num_samples=32,  # Number of action samples for rejection sampling.
            ode_solver='euler',  # Type of the ODE solver ('euler', 'midpoint').
            num_flow_steps=10,  # Number of flow steps.
            normalize_q_loss=False,  # Whether to normalize the Q loss.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='GCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=False,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.5,  # Probability of using a random state as the actor goal.
            actor_geom_sample=True,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
        )
    )
    return config
