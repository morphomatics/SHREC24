import pickle

import jax

import haiku as hk
import optax
from sklearn import model_selection

from morphomatics.nn.train import TrainingState

from helpers import iterate, train, create_data

from model import net_fn

jax.config.update("jax_enable_x64", True)


def main(batch_size: int = 1, n_epochs: int = 100, seed: int = 0, learning_rate: float = 1e-3, n_flow: int = 3,):

    # hold back validation set from the training data
    data_train, data_validation = model_selection.train_test_split(
        trjs_train, test_size=0.25, random_state=seed, stratify=[t.y for t in trjs_train])

    # initialize network
    net = hk.transform(lambda G: net_fn(G, n_flowLayer=n_flow))

    rng = hk.PRNGSequence(jax.random.PRNGKey(13))
    params = net.init(next(rng), next(iterate(trjs_train)))
    flat_para, _ = jax.flatten_util.ravel_pytree(params)
    print(f"Number of network parameters: {len(flat_para)}")

    # initialize optimizer + state
    optimizer = optax.rmsprop(learning_rate)
    opt_state = optimizer.init(params)
    state = TrainingState(params, params, opt_state)

    return train(data_train, data_validation, trjs_test, batch_size, net, optimizer, state, WEIGHTS, rng, n_epochs)


if __name__ == '__main__':
    # get data
    trjs_train, trjs_test, WEIGHTS, classes = create_data()

    # train ensemble
    models = []
    for seed in range(10):
        acc, params = main(batch_size=1, n_epochs=300, seed=seed, learning_rate=1e-3, n_flow=1)
        models.append(jax.device_get(params))
        print("test_acc", f"{acc:.3f}")

    # write parameters of ensemble
    with open('model.pkl', 'wb') as f:
        pickle.dump(models, f)
