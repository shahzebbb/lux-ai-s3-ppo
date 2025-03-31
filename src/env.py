import jax
import jax.numpy as jnp

from luxai_s3.env import LuxAIS3Env
from luxai_s3.params import EnvParams
from luxai_s3.params import env_params_ranges

def make_env():
    ''' Setups env and vmaps the step and reset function for efficiency. '''
    env = LuxAIS3Env()
    reset_fn = jax.vmap(env.reset)
    step_fn = jax.vmap(env.step)

    return env, reset_fn, step_fn

def _sample_params(rng_key):
    ''' 
    Helper function to sample params when running new episodes.

    New params affect the board state and different random params.
    '''
    randomized_game_params = dict()
    for k, v in env_params_ranges.items():
        rng_key, sub_key = jax.random.split(rng_key)
        if isinstance(v[0], int):
            randomized_game_params[k] = jax.random.choice(sub_key, jax.numpy.array(v, dtype=jnp.int16))
        else:
            randomized_game_params[k] = jax.random.choice(sub_key, jax.numpy.array(v, dtype=jnp.float32))
    params = EnvParams(**randomized_game_params)
    return params
