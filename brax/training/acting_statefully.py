# Copyright 2025 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Brax training acting functions."""

import time
from typing import Callable, Sequence, Tuple

from brax import envs
from brax.training.distribution import NormalDistribution, NormalTanhDistribution
from brax.training.lp_distribution import LowPassNoise
from brax.training.types import Metrics
from brax.training.types import Policy
from brax.training.types import PolicyParams
from brax.training.types import PRNGKey
from brax.training.types import Transition
import jax
import jax.numpy as jnp
import numpy as np

State = envs.State
Env = envs.Env

def generate_stateful_unroll(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    unroll_length: int,
    cutoff_freq: float,
    order: int,
    extra_fields: Sequence[str] = (),
) -> Tuple[State, Transition]:
  """Collect trajectories of given unroll_length."""
  # TODO initialize the state of the policy distribution
  noise_dist = LowPassNoise(episode_length=unroll_length,
                            action_dim=env.action_size,
                            cutoff=cutoff_freq,
                            order=order,
                            fs=1./env.dt)
  init_noise_state = noise_dist.init_state(key, n_envs=env_state.done.shape[0])

  def f(carry, unused_t):
    env_state, noise_state, current_key = carry
    current_key, next_key = jax.random.split(current_key)
    actions, policy_extras = policy(env_state.obs, current_key)

    # MAN-IN-THE-MIDDLE attack on the policy distribution
    logits = policy_extras['distribution_params']
    loc, scale = jnp.split(logits, 2, axis=-1)
    dist = NormalTanhDistribution(event_size=loc.shape[-1])
    param_dist = dist.create_dist(logits)
    noise_sample, next_noise_state, _ = noise_dist.sample(noise_state)
    #jax.debug.print("Noise sample: {noise_sample}", noise_sample=noise_sample)
    #tanh_scale = (jax.nn.softplus(scale) + dist._min_std) * dist._var_scale
    #raw_actions = loc + tanh_scale * noise_sample
    raw_actions = param_dist.loc + param_dist.scale * noise_sample
    log_prob = dist.log_prob(logits, raw_actions)
    #jax.debug.print("Log probs: {log_prob}", log_prob=log_prob)
    #jax.debug.print("Raw actions: {raw_actions}", raw_actions=raw_actions)
    actions = dist.postprocess(raw_actions)
    policy_extras['log_prob'] = log_prob
    policy_extras['raw_action'] = raw_actions

    nstate = env.step(env_state, actions)
    state_extras = {x: nstate.info[x] for x in extra_fields}
    transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
        observation=env_state.obs,
        action=actions,
        reward=nstate.reward,
        discount=1 - nstate.done,
        next_observation=nstate.obs,
        extras={'policy_extras': policy_extras, 'state_extras': state_extras},
    )
    return (nstate, next_noise_state, next_key), transition

  f_jit = jax.jit(f, donate_argnums=(0,))
  (final_state, final_noise_state, _), data = jax.lax.scan(
      f_jit, (env_state, init_noise_state, key), (), length=unroll_length
  )
  return final_state, data


#def _agg_fn(metric, fn, to_aggregate, to_normalize, episode_lengths):
#  if not to_aggregate:
#    return metric
#  if to_normalize:
#    return fn(metric / episode_lengths)
#  return fn(metric)
#
#
## TODO(eorsini): Consider moving this to its own file.
#class Evaluator:
#  """Class to run evaluations."""
#
#  def __init__(
#      self,
#      eval_env: envs.Env,
#      eval_policy_fn: Callable[[PolicyParams], Policy],
#      num_eval_envs: int,
#      episode_length: int,
#      action_repeat: int,
#      key: PRNGKey,
#  ):
#    """Init.
#
#    Args:
#      eval_env: Batched environment to run evals on.
#      eval_policy_fn: Function returning the policy from the policy parameters.
#      num_eval_envs: Each env will run 1 episode in parallel for each eval.
#      episode_length: Maximum length of an episode.
#      action_repeat: Number of physics steps per env step.
#      key: RNG key.
#    """
#    self._key = key
#    self._eval_walltime = 0.0
#
#    eval_env = envs.training.EvalWrapper(eval_env)
#    self._eval_state_to_donate = jax.jit(eval_env.reset)(
#        jax.random.split(key, num_eval_envs)
#    )
#
#    def generate_eval_unroll(
#        eval_env_state_donated: State, policy_params: PolicyParams, key: PRNGKey
#    ) -> State:
#      reset_keys = jax.random.split(key, num_eval_envs)
#      eval_first_state = eval_env.reset(reset_keys)
#      return generate_unroll(
#          eval_env,
#          eval_first_state,
#          eval_policy_fn(policy_params),
#          key,
#          unroll_length=episode_length // action_repeat,
#      )[0]
#
#    self._generate_eval_unroll = jax.jit(
#        generate_eval_unroll, donate_argnums=(0,), keep_unused=True
#    )
#    self._steps_per_unroll = episode_length * num_eval_envs
#
#  def run_evaluation(
#      self,
#      policy_params: PolicyParams,
#      training_metrics: Metrics,
#      aggregate_episodes: bool = True,
#  ) -> Metrics:
#    """Run one epoch of evaluation."""
#    self._key, unroll_key = jax.random.split(self._key)
#
#    t = time.time()
#    eval_state = self._generate_eval_unroll(
#        self._eval_state_to_donate, policy_params, unroll_key
#    )
#    self._eval_state_to_donate = eval_state
#
#    eval_metrics = eval_state.info['eval_metrics']
#    eval_metrics.active_episodes.block_until_ready()
#    epoch_eval_time = time.time() - t
#    episode_lengths = np.maximum(eval_metrics.episode_steps, 1.0).astype(float)
#
#    metrics = {}
#    for fn in [np.mean, np.std]:
#      suffix = '_std' if fn == np.std else ''
#      for name, value in eval_metrics.episode_metrics.items():
#        metrics[f'eval/episode_{name}{suffix}'] = _agg_fn(
#            value,
#            fn,
#            aggregate_episodes,
#            name.endswith('per_step'),
#            episode_lengths,
#        )
#
#    metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
#    metrics['eval/std_episode_length'] = np.std(eval_metrics.episode_steps)
#    metrics['eval/epoch_eval_time'] = epoch_eval_time
#    metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
#    self._eval_walltime = self._eval_walltime + epoch_eval_time
#    metrics = {
#        'eval/walltime': self._eval_walltime,
#        **training_metrics,
#        **metrics,
#    }
#
#    return metrics  # pytype: disable=bad-return-type  # jax-ndarray
#