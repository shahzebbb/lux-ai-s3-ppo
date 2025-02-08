'''This file contains code for the basic agent introduced in lux ai s3 remade in jax'''

import jax
import jax.numpy as jnp


################################# ACTION TAKING CODE ###################################################

def take_basic_action(rng_key, num_envs, action_info):
    rng_key, subkey = jax.random.split(rng_key)
    subkeys = jax.random.split(subkey, num_envs)
    
    obs, discovered_relic_positions, discovered_relic_mask, unit_explore_locations, step, team_id = action_info
    obs = obs['player_0']

    action_info = (obs, discovered_relic_positions, discovered_relic_mask, unit_explore_locations, step, team_id)
    return rng_key, *basic_action(subkeys, *action_info)


def basic_action(rng_keys, 
                 obs, 
                 discovered_relic_position,
                 discovered_relic_mask,
                 unit_explore_locations,
                 step,
                 team_id
                ):
    
    unit_masks = obs.units_mask[:, team_id, :] # (num_envs, 16)
    unit_positions = obs.units.position[: ,team_id, :, :] # (num_envs, 16, 2)
    observed_relic_position = obs.relic_nodes # (num_envs, max_relic_nodes, 2)
    observed_relic_mask = obs.relic_nodes_mask # (num_envs, max_relic_nodes)

    discovered_relic_mask, discovered_relic_position = update_relic_nodes(observed_relic_mask, 
                                                                          discovered_relic_mask, 
                                                                          observed_relic_position, 
                                                                          discovered_relic_position)

    actions, unit_explore_locations, rng_keys = jax.vmap(get_unit_action, in_axes=(0,0,0,0,None,0))(unit_positions, unit_masks,
                    discovered_relic_position, unit_explore_locations, 
                    step, rng_keys)

    return actions, discovered_relic_position, discovered_relic_mask, unit_explore_locations

#####################################################################################################

################################# HELPER FUNCTIONS ###################################################

def direction_to(src, target):
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    return jax.lax.cond((dx == 0) & (dy == 0),
        lambda _: 0,
        lambda _: jax.lax.cond(jnp.abs(dx) > jnp.abs(dy),
            lambda _: jax.lax.cond(dx > 0,
                lambda _: 2,
                lambda _: 4,
                operand=None),
            lambda _: jax.lax.cond(dy > 0,
                lambda _: 3,
                lambda _: 1,
                operand=None),
            operand=None),
        operand=None)

def update_relic_nodes(observed_relic_mask, discovered_relic_mask, 
                    observed_relic_position, discovered_relic_position):
    # Create a mask for relic_ids that are not yet discovered
    new_relic_mask = discovered_relic_mask | observed_relic_mask
    
    new_relic_positions = jnp.where(
    observed_relic_mask[..., None],  # Broadcast observed mask along positions
    observed_relic_position,      # Use observed positions where mask is True
    discovered_relic_position     # Retain existing positions otherwise
    )

    return new_relic_mask, new_relic_positions


################# For loop ##################

def get_unit_action(unit_positions, unit_masks,
                    discovered_relic_positions, unit_explore_locations, 
                    step, rng_key):

    actions = jnp.full((16,3), 2, dtype=jnp.int32)
    for i in range(len(unit_positions)):
        unit_pos = unit_positions[i]
        unit_mask = unit_masks[i]
        unit_explore = unit_explore_locations[i]
        action, unit_explore, rng_key = jax.lax.cond(unit_mask,
                                                     lambda args: calc_action(*args),
                                                     lambda args: no_action(*args),
                                                     (unit_pos, unit_mask, discovered_relic_positions, 
                                                      unit_explore, step, rng_key))

        # jax.debug.print("Inside get_unit_action: step = {}, unit_pos = {}, unit_mask = {}, action = {}", step, unit_pos, unit_mask, action)
        # jax.debug.print("--------------------------------------------------------")
        
        actions = actions.at[i].set(action) 
        unit_explore_locations = unit_explore_locations.at[i].set(unit_explore)

    # jax.debug.print("final_actions: actions = {}", actions)
    return actions, unit_explore_locations, rng_key


################# Outer if-statement ##################


def calc_action(unit_pos, unit_mask,
               discovered_relic_positions, 
               unit_explore, step, rng_key, 
              ):

    # jax.debug.print("Inside calc_action: step = {}, unit_pos = {}, unit_mask = {}", step, unit_pos, unit_mask)
    action, unit_explore, rng_key = jax.lax.cond(jnp.any(discovered_relic_positions[:, 0] != -1),
                                   lambda args: relic_action(*args),
                                   lambda args: explore_action(*args),
                                   (unit_pos, unit_mask, discovered_relic_positions, 
                                    unit_explore, step, rng_key))
    
    return action, unit_explore, rng_key


def no_action(unit_pos, unit_mask,
           discovered_relic_positions, 
           unit_explore, step, rng_key, 
          ):

    action = jnp.array([-1,0,0])

    # jax.debug.print("Inside no action: step = {}, unit_pos = {}, unit_mask = {}", step, unit_pos, unit_mask)
    
    return action, unit_explore, rng_key

################# Inner if-statement loop ##################

def relic_action(unit_pos, unit_mask,
                 discovered_relic_positions, 
                 unit_explore, step, rng_key, 
                ):
    mask = jnp.logical_not(jnp.all(discovered_relic_positions == jnp.array([-1, -1]), axis=1))
    valid_relics = jnp.where(mask[:, None], discovered_relic_positions, jnp.array([-1000, -1000]))
    distances = jnp.abs(unit_pos[0] - valid_relics[:, 0]) + jnp.abs(unit_pos[1] - valid_relics[:, 1])
    nearest_idx = jnp.argmin(distances)
    nearest_relic = valid_relics[nearest_idx]
    distance = jnp.abs(unit_pos[0] - nearest_relic[0]) + jnp.abs(unit_pos[1] - nearest_relic[1])

    # Random vs direct move
    new_key, dir_key = jax.random.split(rng_key)
    random_dir = jax.random.randint(dir_key, (), 0, 5)
    action_dir = jnp.where(distance <= 4,
                           random_dir,
                           direction_to(unit_pos, nearest_relic))

    
    # debug = jnp.any(discovered_relic_positions[:, 0] != -1)
    # jax.debug.print("Inside relic_action, step = {}, unit_pos = {}, nearest_relic = {}, action_dir = {}, debug = {}", step, unit_pos, nearest_relic, action_dir, debug)
    # # jax.debug.print("--------------------------------------------------------")
    return jnp.array([action_dir, 0, 0]), unit_explore, new_key
    

def explore_action(unit_pos, unit_mask,
                 discovered_relic_positions, 
                 unit_explore, step, rng_key, 
                ):
    
    unit_explore_mask = unit_explore[0] == -1
    need_new = (step % 20 == 0) | (unit_explore_mask)

    def take_random_action():
        new_key, x_key, y_key = jax.random.split(rng_key, 3)
        new_x = jax.random.randint(x_key, (), 0, 24)
        new_y = jax.random.randint(y_key, (), 0, 24)
        new_pos = jnp.array([new_x, new_y])
        action_dir = direction_to(unit_pos, new_pos)
        return action_dir, new_pos, new_key

    def take_action_to_explore():
        return direction_to(unit_pos, unit_explore), unit_explore, rng_key

    action_dir, new_pos, new_key = jax.lax.cond(need_new,
                             lambda _: take_random_action(),
                             lambda _: take_action_to_explore(),
                             operand=None)

    # jax.debug.print("Inside explore_action, step = {}, unit_pos = {}, unit_explore = {}, action_dir = {}", step, unit_pos, unit_explore, action_dir)

    return jnp.array([action_dir, 0, 0]), new_pos, new_key


#####################################################################################################