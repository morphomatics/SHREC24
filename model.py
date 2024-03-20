import jax
import jax.numpy as jnp

import jraph
import haiku as hk

from morphomatics.manifold import Euclidean
from morphomatics.graph import max_pooling, mean_pooling
from morphomatics.nn import FlowLayer, TangentMLP, MfdInvariant

def net_fn(G: jraph.GraphsTuple, n_flowLayer) -> jnp.ndarray:
    n_steps = 1

    # number of channels per node
    n_flow_channel = G.nodes.shape[1]

    # signal domain
    M = Euclidean(G.nodes.shape[2:])

    for i in range(n_flowLayer - 1):
        # diffusion layer
        G = FlowLayer(M, n_steps)(G)
        # node-wise MLP
        G = G._replace(nodes=TangentMLP(M, (n_flow_channel,))(G.nodes[None])[0])

    # final diffusion layer
    G = FlowLayer(M, n_steps)(G)
    # node-wise invariant layer
    z = MfdInvariant(M, n_flow_channel)(G.nodes[None])[0]
    z = jax.nn.leaky_relu(z)

    # global pooling
    z = jnp.concatenate((max_pooling(G, z), mean_pooling(G, z)), axis=1)

    ### MLP mapping to NUM_CLASSES channels per graph ###
    return hk.nets.MLP([n_flow_channel, n_flow_channel//2, 7], activation=jax.nn.leaky_relu)(z)