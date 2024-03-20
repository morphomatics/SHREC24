import sys
import pickle
import argparse
from glob import glob
from typing import Sequence

import numpy as np
import jax
import jax.numpy as jnp

import haiku as hk

from morphomatics.manifold import Sphere
from morphomatics.stats import ExponentialBarycenter as Mean
from morphomatics.stats import GeometricMedian as Median

from model import net_fn
from helpers import Motion, classes, read, iterate

class_idx_to_str = lambda idx: list(classes.keys())[list(classes.values()).index(idx)]

def predict(trjs: Sequence[Motion], model_pkl: str):

    # create model
    net = hk.transform(lambda G: net_fn(G, n_flowLayer=1))
    net = hk.without_apply_rng(net)

    # load model params from disk
    with open(model_pkl, 'rb') as f:
        models = pickle.load(f)
        params = [jax.tree.map(lambda x: x.astype(jnp.float64), p) for p in models]

    for i, trj in enumerate(iterate(trjs)):
        # predict
        p = [jax.nn.softmax(net.apply(p, trj)) for p in params]
        # fuse
        mu = Mean.compute(Sphere((6,)), jnp.sqrt(jnp.asarray(p)))
        # mu = Median.compute(Sphere((6,)), jnp.sqrt(jnp.asarray(p)))
        c = jnp.argmax(mu)
        print(i, class_idx_to_str(c), int(trj.globals[0]), c)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict hand motion.')
    
    parser.add_argument('model', type=str, nargs='?', default='model_10x.pkl',
                    help='path to trained model(s)')
    parser.add_argument('path', type=str, nargs='?', default='./data/Test-set/**/*.txt',
                    help='path to motion file (may be glob pattern)')

    args = parser.parse_args()

    files = sorted(glob(args.path, recursive=True))
    trjs = read(files)

    predict(trjs, args.model)