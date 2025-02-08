import jax
import jax.numpy as jnp

from luxai_s3.params import EnvParams
from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import env_params_ranges


env = LuxAIS3Env(auto_reset=False, fixed_env_params=EnvParams())

def return_vmap_env():
    reset_fn = jax.vmap(env.reset)
    step_fn = jax.vmap(env.step)
    return reset_fn, step_fn


def return_max_steps():
    return (env.fixed_env_params.max_steps_in_match + 1) * env.fixed_env_params.match_count_per_episode


def sample_params(rng_key):
    randomized_game_params = dict()
    for k, v in env_params_ranges.items():
        rng_key, subkey = jax.random.split(rng_key)
        if isinstance(v[0], int):
            randomized_game_params[k] = jax.random.choice(subkey, jax.numpy.array(v, dtype=jnp.int16))
        else:
            randomized_game_params[k] = jax.random.choice(subkey, jax.numpy.array(v, dtype=jnp.float32))
    params = EnvParams(**randomized_game_params)
    return params

def return_max_steps():
    return (env.fixed_env_params.max_steps_in_match + 1) * env.fixed_env_params.match_count_per_episode
