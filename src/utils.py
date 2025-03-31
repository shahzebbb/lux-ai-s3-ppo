from typing import NamedTuple 

import jax.numpy as jnp 
import jax

from models.utils import transform_obs
from env import make_env

_, reset_fn, step_fn = make_env()

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def _env_step(runner_state, unused, env_params, network, config):
    train_state, env_state, last_obs, rng = runner_state

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    transformed_obs = transform_obs(last_obs["player_0"]) 
    pi, value = network.apply(train_state.params, transformed_obs)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    action_expanded = action[..., jnp.newaxis].astype(jnp.int32)
    action_expanded = jnp.concatenate([action_expanded, jnp.zeros(action.shape + (2,), dtype=jnp.int32)], axis=-1)
    
    action_dict = {"player_0": action_expanded, "player_1": jnp.zeros((config["NUM_ENVS"],16,3), dtype=jnp.int32)}

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, config["NUM_ENVS"])
    obs, env_state, wins, terminated_dict, truncated_dict, info = step_fn(rng_step, env_state, action_dict, env_params)
    reward = jnp.maximum(obs["player_0"].team_points[:, 0] - last_obs["player_0"].team_points[:, 0], 0)
    
    # Perform element-wise OR operation
    done_terminated = jnp.logical_or(terminated_dict["player_0"], terminated_dict["player_1"])
    done_truncated = jnp.logical_or(truncated_dict["player_0"], terminated_dict["player_1"])
    
    done = jnp.logical_or(done_terminated, done_truncated).astype(jnp.int32)
    
    # Define return tuples
    transition = Transition(done, action, value, reward, log_prob, last_obs, info)
    runner_state = (train_state, env_state, obs, rng)

    return runner_state, transition

def _calculate_gae(traj_batch, last_val, config):
    
    def _get_advantages(gae_and_next_value, transition):
        # Unpack tuples into variables
        gae, next_value = gae_and_next_value
        done, value, reward = (
            transition.done,
            transition.value,
            transition.reward
        )

        # This is TD(1)
        delta = reward + config["GAMMA"] * next_value * (1-done) - value

        # Recursively update gae by adding to previous gae
        gae = (
            delta
            + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
        )
        
        return (gae, value), gae

    # Scan over _get_advantages to obtain advantages using a carry sum.
    # Notice how you go backwards, unroll is simply for efficiency
    # and doesn't change the final answer.
    _, advantages =jax.lax.scan(
        f=_get_advantages,
        init=(jnp.zeros_like(last_val), last_val), # (gae, value)
        xs=traj_batch,
        reverse=True,
        unroll=16
    )

    # I do not know why targets are advantages + traj_batch.value
    return advantages, advantages + traj_batch.value
            

def _update_epoch(update_state, unused, network, config):

    # Define some helper functions 
    def _update_minbatch(train_state, batch_info):
        traj_batch, advantages, targets = batch_info
        
        def _loss_fn(params, traj_batch, gae, targets):
            # RERUN NETWORK
            transformed_obs = transform_obs(traj_batch.obs["player_0"])
            pi, value = network.apply(params, transformed_obs)
            log_prob = pi.log_prob(traj_batch.action)

            # CALCULATE VALUE LOSS
            value_pred_clipped = traj_batch.value + (
                value - traj_batch.value
            ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = (
                0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            )

            # CALCULATE ACTOR LOSS
            ratio = jnp.exp(log_prob - traj_batch.log_prob)
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)
            gae = gae.reshape(-1,1)


            loss_actor1 = (ratio * gae).mean(axis=-1)
            loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - config["CLIP_EPS"],
                    1.0 + config["CLIP_EPS"],
                )
                * gae
            ).mean(axis=-1)
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()

            # CALCULATE ENTROPY
            entropy = pi.entropy().mean(axis=-1).mean()

            # Add up to create the final total loss
            total_loss = (
                loss_actor
                + config["VF_COEF"] * value_loss
                - config["ENT_COEF"] * entropy
            )

            return total_loss, (value_loss, loss_actor, entropy)

        # Calculate grads and update grads on weights
        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        total_loss, grads = grad_fn(
            train_state.params, traj_batch, advantages, targets
        )
        train_state = train_state.apply_gradients(grads=grads)
        
        return train_state, total_loss
    
    train_state, traj_batch, advantages, targets, rng = update_state
    rng, sub_key = jax.random.split(rng)

    # Perform batching and shuffling of data
    batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
    assert (batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]), (
        "batch size must be equal to number of steps * number of envs"
    )
    permutation = jax.random.permutation(sub_key, batch_size)

    batch = (traj_batch, advantages, targets)

    # Reshape data to merge timesteps and num_envs into batch_size
    batch = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
    )
    
    shuffled_batch = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch
    )

    # Reshape data to add mini batches
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(
            x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
        ),
        shuffled_batch,
    )

    train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

    update_state = (train_state, traj_batch, advantages, targets, rng)
    return update_state, total_loss
