import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from models.conv_network import ConvActorCritic
from models.utils import transform_obs
from env import make_env, _sample_params
from utils import _env_step, _calculate_gae, _update_epoch

def train(rng):
    # Init the network
    network = ConvActorCritic()

    # Init network parameters by passing a rng key and dummy value (init_x)
    rng, sub_key = jax.random.split(rng)
    init_x = jnp.zeros((config["NUM_ENVS"], 7, 24, 24))
    network_params = network.init(sub_key, init_x)

    # Check how this works
    tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

    # Check how this works
    train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

    # Init env
    rng, sub_key = jax.random.split(rng)
    env_params = jax.vmap(_sample_params)(jax.random.split(sub_key, config["NUM_ENVS"]))

    rng, sub_key = jax.random.split(rng)    
    obs, env_state = reset_fn(jax.random.split(sub_key, config["NUM_ENVS"]), env_params)

    # Define function for one iteration of training
    def _update_step(runner_state, unused, env_params, network, config):

        train_state, _, _, rng = runner_state

        # Init env again 
        # Repeated to make sure that there is a new env on every iteration
        rng, sub_key = jax.random.split(rng)
        env_params = jax.vmap(_sample_params)(jax.random.split(sub_key, config["NUM_ENVS"]))

        rng, sub_key = jax.random.split(rng)    
        obs, env_state = reset_fn(jax.random.split(sub_key, config["NUM_ENVS"]), env_params)

        runner_state = (train_state, env_state, obs, rng)

        # Run environment to get actions, values, etc.
        runner_state, traj_batch = jax.lax.scan(lambda carry, x: _env_step(carry, x, 
                                                            env_params, network, config), 
                                            runner_state, 
                                            None, 
                                            config["NUM_STEPS"])

        # Get value for final state
        train_state, env_state, last_obs, rng = runner_state
        transformed_obs = transform_obs(last_obs["player_0"]) 
        _, last_val = network.apply(train_state.params, transformed_obs)

        # Calculate advantages and targets via jax.lax.scan
        advantages, targets = _calculate_gae(traj_batch, last_val, config)

        # Perform some epochs of update over collected data
        # Also return metrics to be printed
        update_state = (train_state, traj_batch, advantages, targets, rng)
        update_state, loss_info = jax.lax.scan(lambda carry, x: _update_epoch(carry, x,
                                                            network, config),
                                               update_state,
                                               None,
                                               config["UPDATE_EPOCHS"])        
        train_state = update_state[0]
        metric = traj_batch.reward
        rng = update_state[-1]

        # Define callback to print info to screen
        def callback(reward):
            average_reward = reward.sum(axis=0).mean()
            print(f"average_episodic return={average_reward}")
        jax.debug.callback(callback, metric)

        runner_state = (train_state, env_state, last_obs, rng)
        return runner_state, metric

        
    # Perform training
    rng, _rng = jax.random.split(rng)
    runner_state = (train_state, env_state, obs, _rng)
    jax.debug.print("num_updates: {}",config["NUM_UPDATES"])
    runner_state, metric = jax.lax.scan(lambda carry, x: _update_step(carry, x, 
                                                            env_params, network,
                                                            config), 
                                        runner_state, 
                                        None, 
                                        config["NUM_UPDATES"])

    return {"runner_state": runner_state, "metrics": metric}

def make_train(config):
    # Update config to create some keys which will be used in training
    config["NUM_STEPS"] = (
        env.fixed_env_params.max_steps_in_match
        * env.fixed_env_params.match_count_per_episode - 1
    )

    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    return train


if __name__ == "__main__":
    config = {
    "LR": 2.5e-4,
    "NUM_ENVS": 4,
    "TOTAL_TIMESTEPS": 5e7,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 4,
    "GAMMA": 0.99,
    "GAE_LAMBDA": 0.95,
    "CLIP_EPS": 0.2,
    "ENT_COEF": 0.01,
    "VF_COEF": 0.5,
    "MAX_GRAD_NORM": 0.5,
    "ANNEAL_LR": True,
    "DEBUG": True,
    }
    env, reset_fn, step_fn = make_env()
    print("It takes around 3 minutes for jax to get initialized. Please wait...")
    rng = jax.random.key(30)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
