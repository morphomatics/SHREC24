import os
from glob import glob
from typing import NamedTuple, List, Generator, Sequence

import numpy as np
import jraph
import random

import jax.numpy as jnp

from morphomatics.nn.train import evaluate, update

data_dir = 'data/'

classes = ['Centering', 'MakingHole', 'Pressing', 'Raising', 'Smoothing', 'Sponge', 'Tightening']
classes = dict(zip(classes, np.arange(len(classes))))

MAX_NUM_NODES = 3721

class Motion(NamedTuple):
    x: np.ndarray
    y: int
    path: str


def read(files: Sequence[str]) -> Sequence[Motion]:
    trjs: List[Motion] = []
    for f in files:
        x = np.loadtxt(f, delimiter=';', usecols=np.arange(85))
        x = x[:, 1:].reshape(-1, 28, 3)
        try:
            y = f.split(os.sep)[-2]
            y = classes[y]
        except:
            y = -1
        trjs.append(Motion(x, y, f))
    return trjs

def create_data():
    trjs_train = read(glob(os.path.join(data_dir, 'Train-set/**/*.txt'), recursive=True))
    trjs_test = read(glob(os.path.join(data_dir, 'Test-set/**/*.txt'), recursive=True))

    print('classes: {0} / counts: {1}'.format(*np.unique([t.y for t in trjs_train], return_counts=True)))
    print('classes: {0} / counts: {1}'.format(*np.unique([t.y for t in trjs_test], return_counts=True)))

    weights = len(trjs_train) / np.unique([t.y for t in trjs_train], return_counts=True)[1]

    return trjs_train, trjs_test, weights, classes


def iterate(data: Sequence[Motion]) -> Generator[jraph.GraphsTuple, None, None]:
    for t in data:
        x = jnp.asarray(t.x, dtype=jnp.float64)
        n_e = len(x) - 1
        yield jraph.GraphsTuple(
            n_node=jnp.asarray([len(x)]),
            n_edge=jnp.asarray([2 * n_e]),
            nodes=x,
            edges=jnp.ones(2 * n_e, dtype=jnp.float64),
            globals=jnp.array([t.y]),
            senders=jnp.r_[jnp.arange(n_e), jnp.arange(1, n_e + 1)],
            receivers=jnp.r_[jnp.arange(1, n_e + 1), jnp.arange(n_e)])


def batch_iterate(data: Sequence[Motion], batch_size: int) -> Generator[jraph.GraphsTuple, None, None]:
    # upper bound to always fit batch_size graphs into a batch
    batch_max_num_nodes = batch_size * MAX_NUM_NODES
    # number of frames/nodes varies a lot -> reduce number of dummy computations
    batch_max_num_nodes = np.min([2 * MAX_NUM_NODES, batch_max_num_nodes])
    batch_iter = jraph.dynamically_batch(
        iterate(data),
        n_node=batch_max_num_nodes + 1,  # +1 for the extra padding node
        n_edge=2 * (batch_max_num_nodes - 1),  # fwd & bwd edges -> *2
        n_graph=batch_size + 1)
    for batch in batch_iter:
        # init padding nodes
        mask = jraph.get_node_padding_mask(batch)
        batch.nodes[~mask] = batch.nodes[0]
        yield batch


def train(data_train, data_validation, data_test, batch_size, net, optimizer, state, WEIGHTS, rng, n_epochs):
    def batch_eval(data):
        n = 0
        a = 0.
        for g in batch_iterate(data, batch_size):
            mask = jraph.get_graph_padding_mask(g)
            n_g = jnp.sum(mask)
            a += n_g * evaluate(state.avg_params, g, g.globals, len(classes), net, next(rng), mask)
            n += n_g
        return a / n if n > 0 else -1.

    opt_acc = 0.
    opt_test_acc = 0.
    opt_param = state.params

    # training loop
    for i in range(n_epochs):
        # do random assignment to batches in each epoch
        random.Random(i).shuffle(data_train)

        # train epoch
        for step, batch in enumerate(batch_iterate(data_train, batch_size)):
            mask = jraph.get_graph_padding_mask(batch)
            state = update(state, batch, batch.globals, optimizer, net, next(rng), mask, jnp.asarray(WEIGHTS))

        # evaluate accuracy
        train_acc = batch_eval(data_train)
        validation_acc = batch_eval(data_validation)
        _test_acc = batch_eval(data_test)

        # print validation results
        print({"epoch": f"{i}", "train_acc": f"{train_acc:.3f}", "validation_acc": f"{validation_acc:.3f}",
               "test_acc": f"{_test_acc:.3f}"})

        # save best model
        if train_acc == 1 and validation_acc > opt_acc:
            opt_param = state.avg_params
            opt_acc = validation_acc
            opt_test_acc = _test_acc

    return opt_test_acc, opt_param
