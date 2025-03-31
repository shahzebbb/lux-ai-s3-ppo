import jax
import jax.numpy as jnp

def transform_obs(obs, team_id = 0):
    ''' 
    
    Takes in observation from the environment and converts it to a jax array
    to be run through the model.

    Map Height = Map Width = 24

    Args:
        - obs: a dict of observations received from the environment
        - team_id: a tag to identify which team's observations we need to look at

    Returns:
        - transformed_obs: an jax array of shape 
                           (Batch, Channels, Map Height, Map Width)
    '''

    # Create a board of shape (B, 4, map_height, map_width) where the first axis
    # contains positions and energies of both teams on the board
    pos_energy_board = _fill_positions_and_energies(obs, team_id)

    # Create a board of shape (B, 1, map_height, map_width) where the first axis
    # contains the position of relics currently observed in th environment
    relic_board = _fill_relics(obs, team_id)

    # Create a board of shape (B, 2, map_height, map_width) where the first axis
    # contains the map tiles and the energy of each tile
    map_board = _fill_map(obs, team_id)

    # Concatenate everything together to output array of shape (B, 7, map_height, map_width)
    transformed_obs = jnp.concatenate([pos_energy_board, relic_board, map_board], axis=1)
    
    return transformed_obs

def _fill_positions_and_energies(obs, team_id):
    ''' Helper function to extract the positions and energies onto a map board. '''

    # Extract the coordinates and energies of the units of both the enemy and ally team
    ally_pos = obs.units.position[:, team_id, :, :]
    ally_energies = obs.units.energy[:, team_id, :]
    enemy_pos = obs.units.position[:, 1-team_id, :, :]
    enemy_energies = obs.units.energy[:, 1-team_id, :]

    def _process_batch(ally_pos, ally_energies, enemy_pos, enemy_energies):
        # Create a board of zeros for all pos and energies
        board = jnp.zeros((4, 24, 24), dtype=jnp.int32)
        
        # Determine which units are alive (i.e. not (-1, -1))
        ally_alive = jnp.all(ally_pos != -1, axis=-1) 
        enemy_alive = jnp.all(enemy_pos != -1, axis=-1)
        
        # Get the positions for alive units
        ally_alive_pos = jnp.where(ally_alive[:, None], ally_pos, jnp.array([0, 0]))
        enemy_alive_pos = jnp.where(enemy_alive[:, None], enemy_pos, jnp.array([0, 0]))
        
        # Update board to have positions of allies and enemies
        # If a unit exists at position (x,y) then that positions value on the board
        # will be 1
        ally_updates = ally_alive.astype(jnp.int32)
        enemy_updates = enemy_alive.astype(jnp.int32)
        board = board.at[0, ally_alive_pos[:, 0], ally_alive_pos[:, 1]].add(ally_updates)
        board = board.at[2, enemy_alive_pos[:, 0], enemy_alive_pos[:, 1]].add(enemy_updates)

        # Update board to energies of allies and enemies
        ally_energy_updates = ally_alive.astype(ally_energies.dtype) * ally_energies
        enemy_energy_updates = enemy_alive.astype(enemy_energies.dtype) * enemy_energies
        board = board.at[1, ally_alive_pos[:, 0], ally_alive_pos[:, 1]].add(ally_energy_updates)
        board = board.at[3, enemy_alive_pos[:, 0], enemy_alive_pos[:, 1]].add(enemy_energy_updates)

        return board

    # Vmap over the batch dimension for speed
    boards = jax.vmap(_process_batch)(ally_pos, 
                ally_energies, 
                enemy_pos, 
                enemy_energies)

    return boards


def _fill_relics(obs, team_id):
    ''' Helper function to extract the positions of relics onto a map board. '''
    relic_nodes = obs.relic_nodes
    
    def process_batch(relic_nodes):
        # Create a board of zeros
        board = jnp.zeros((1,24, 24), dtype=jnp.int32)
        
        # Determine which relics have been found (i.e. not (-1, -1))
        found_relics = jnp.all(relic_nodes != -1, axis=-1) 
        
        # Get the positions of the found relics
        relic_pos = jnp.where(found_relics[:, None], relic_nodes, jnp.array([0, 0]))
        
        # Update board to have a 1 at positions where the relic exists
        relic_updates = found_relics.astype(jnp.int32)
        board = board.at[0, relic_pos[:, 0], relic_pos[:, 1]].add(relic_updates)
        
        return board
    
    # Vmap over the batch dimension for speed
    boards = jax.vmap(process_batch)(relic_nodes)
    return boards
    

def _fill_map(obs, team_id):
    ''' Helper function to simply extrac the map and energy tiles from obs '''

    # Map and energy tiles are already in the board shape required so just
    # eed to add a batch dimension
    map_tiles = jnp.expand_dims(obs.map_features.tile_type, axis=1)
    map_energy = jnp.expand_dims(obs.map_features.energy, axis=1)

    return jnp.concatenate([map_tiles, map_energy], axis=1)
    