import jax.numpy as jnp
import distrax
from flax import linen as nn


class ConvActorCritic(nn.Module):
    ''' 
    Model class to create a conv model as per jax documentation.

    Contains two models: one actor and one critic model. 

    Both models input an array of shape (Batch, 7, 24, 24)

    In both models, the input runs through three convolution blocks, 
    then flattened and run through dense layers.
    
    The output of the actor is an array of shape (Batch, 16, 5) representing
    16 units each with 5 possible actions.

    The output of the critic is an array of shape (Batch,1) representing the scalar
    value.
    '''
    features_dim: int = 512

    @nn.compact
    def __call__(self, x):

        ###### ACTOR MODEL ######

        # First convolutional block
        actor_mean = nn.Conv(
            features=64,
            kernel_size=(8, 8),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            name="conv1"
        )(x)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.LayerNorm(reduction_axes=(1, 2, 3), name="ln1")(actor_mean)
        
        # Second convolutional block
        actor_mean = nn.Conv(
            features=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            name="conv2"
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.LayerNorm(reduction_axes=(1, 2, 3), name="ln2")(actor_mean)
        
        # Third convolutional block
        actor_mean = nn.Conv(
            features=256,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            name="conv3"
        )(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.LayerNorm(reduction_axes=(1, 2, 3), name="ln3")(actor_mean)
        
        # Flatten: combine spatial and channel dimensions
        actor_mean = x.reshape((x.shape[0], -1))
        
        # Fully connected layers
        actor_mean = nn.Dense(features=1024, name="dense1")(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.LayerNorm(reduction_axes=-1, name="ln_dense")(actor_mean)
        actor_mean = nn.Dense(features=self.features_dim, name="dense2")(actor_mean)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(features=16 * 5, name="dense3")(actor_mean)
        actor_mean = actor_mean.reshape(-1, 16, 5)
        pi = distrax.Categorical(logits=actor_mean)

        ##############################

        ###### CRITIC MODEL ######
        # First convolutional block
        critic = nn.Conv(
            features=64,
            kernel_size=(8, 8),
            strides=(1, 1),
            padding=((1, 1), (1, 1)),
            name="conv1_critic"
        )(x)
        critic = nn.relu(critic)
        critic = nn.LayerNorm(reduction_axes=(1, 2, 3), name="ln1_critic")(critic)
        
        # Second convolutional block
        critic = nn.Conv(
            features=128,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            name="conv2_critic"
        )(critic)
        critic = nn.relu(critic)
        critic = nn.LayerNorm(reduction_axes=(1, 2, 3), name="ln2_critic")(critic)
        
        # Third convolutional block
        critic = nn.Conv(
            features=256,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            name="conv3_critic"
        )(critic)
        critic = nn.relu(critic)
        critic = nn.LayerNorm(reduction_axes=(1, 2, 3), name="ln3_critic")(critic)
        
        # Flatten spatial and channel dimensions
        critic = critic.reshape((critic.shape[0], -1))
        
        # Fully connected layers
        critic = nn.Dense(features=1024, name="dense1_critic")(critic)
        critic = nn.relu(critic)
        critic = nn.LayerNorm(reduction_axes=-1, name="ln_dense_critic")(critic)
        critic = nn.Dense(features=512, name="dense2_crtiic")(critic)
        critic = nn.relu(critic)
        critic = nn.Dense(features=1, name="dense3_critic")(critic)
        
        # Squeeze the last dimension to produce a scalar value
        value = critic.squeeze(-1)

        ##############################
        
        return pi, value
