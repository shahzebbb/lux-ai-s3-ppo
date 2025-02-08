import unittest

import jax
import jax.numpy as jnp

from tests.utils import return_vmap_env, return_max_steps, sample_params
from train.basic_agent import take_basic_action


NUM_ENVS = 5
reset_fn, step_fn = return_vmap_env()
max_episode_steps = return_max_steps()


def run_episode_and_reset(rng_key, env_params, num_envs = NUM_ENVS):
    # check shape of rng_key
    rng_key, subkey = jax.random.split(rng_key)
    obs, state = reset_fn(jax.random.split(subkey, num_envs), env_params)
    discovered_relic_positions = jnp.full((num_envs,6,2), -1)
    discovered_relic_mask = jnp.full((num_envs,6), False)
    unit_explore_locations = jnp.full((num_envs,16,2), -1)
    step = 0 
    team_id = 0
    action_info = (obs, discovered_relic_positions, discovered_relic_mask, unit_explore_locations, step, team_id)
    
    def take_step(carry, _):
        # Unpack tuples
        rng_key, action_info, state = carry
        (obs, discovered_relic_positions, discovered_relic_mask, unit_explore_locations, step, team_id) = action_info

        # Take action
        rng_key, actions, discovered_relic_positions, discovered_relic_mask, unit_explore_locations = take_basic_action(rng_key, num_envs, action_info)
        # Hackery
        action = {
            'player_0': actions,
            'player_1': actions
        }
        
        rng_key, subkey = jax.random.split(rng_key)

        obs, state, reward, terminated_dict, truncated_dict, info = step_fn(
            jax.random.split(subkey, num_envs), 
            state, 
            action,
            env_params
        )

        step += 1
        
        action_info = (obs, discovered_relic_positions, discovered_relic_mask, unit_explore_locations, step, team_id)
        
        return (rng_key, action_info, state), (obs, state, reward, terminated_dict, truncated_dict, info)
        
    _, (obs, state, reward, terminated_dict, truncated_dict, info) = jax.lax.scan(take_step, (rng_key, action_info, state), length=max_episode_steps)
    
    return obs, state, reward, terminated_dict, truncated_dict, info

run_episode_and_reset_jit = jax.jit(run_episode_and_reset)

class TestBasicAgent(unittest.TestCase):
    def test_points(self):
        seed = 42
        rng_key = jax.random.key(seed)

        rng_key, subkey = jax.random.split(rng_key)
        env_params = jax.vmap(sample_params)(jax.random.split(subkey, NUM_ENVS))

        rng_key, subkey = jax.random.split(rng_key)
        obs, state, reward, terminated_dict, truncated_dict, info = run_episode_and_reset_jit(subkey, env_params)
        expected_points = jnp.array([9360,3789,26455,13040,12569], dtype=jnp.int32)
        returned_points = obs["player_0"].team_points.sum(axis=0)[:, 0]
        self.assertTrue(jnp.all(expected_points == returned_points), 
                       f"Points are not equal, something is wrong with the implementation of basic agent. Expected {expected_points}, got {returned_points}")


if __name__ == "__main__":
    unittest.main()