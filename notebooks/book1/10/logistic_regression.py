import jax
import jax.numpy as jnp
import optax


@jax.jit
def binary_loss_function(parameters, x, y, lambd):
    z = jnp.dot(x, parameters["weights"]) + parameters["bias"]
    hypothesis_x = jax.nn.sigmoid(z)
    regularizer = (lambd * jnp.sum(jnp.sum(parameters["weights"] ** 2))) / (2 * x.shape[0])

    return (
        -(
            (jnp.dot(y.T, jnp.log(hypothesis_x + 1e-7)) + jnp.dot((1 - y).T, jnp.log(1 - hypothesis_x + 1e-7)))
            / x.shape[0]
        )
        + regularizer
    )[0]


@jax.jit
def multi_loss_function(parameters, x, y, lambd):
    z = jnp.dot(parameters["weights"], x.T) + parameters["bias"]
    hypothesis_x = jax.nn.softmax(z, axis=1)
    regularizer = (lambd * jnp.sum(jnp.sum(parameters["weights"] ** 2, axis=1))) / (2 * x.shape[0])

    return -jnp.sum(y * jnp.log(hypothesis_x.T + 1e-7)) / x.shape[0] + regularizer


def init_weights(n_f, n_c, random_key):
    parameters = {}
    parameters["weights"] = jax.random.normal(key=jax.random.PRNGKey(random_key), shape=[n_c, n_f])
    parameters["bias"] = jnp.zeros((n_c, 1))
    return parameters


def fit(x, y, max_iter=1000, learning_rate=0.75, lambd=0.001, random_key=1):
    n_classes = len(jnp.unique(y))
    if n_classes > 2:
        y = jax.nn.one_hot(y, n_classes)
        parameters = init_weights(n_f=x.shape[1], n_c=y.shape[1], random_key=random_key)
        loss_and_grad_fn = jax.value_and_grad(multi_loss_function)
    elif n_classes == 2:
        parameters = {}
        parameters["weights"] = jax.random.normal(key=jax.random.PRNGKey(random_key), shape=[x.shape[1], 1])
        parameters["bias"] = jnp.zeros(1)
        loss_and_grad_fn = jax.value_and_grad(binary_loss_function)

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(parameters)

    def one_step(carry, loss):

        parameters = carry["parameters"]
        x, y = carry["x"], carry["y"]
        opt_state = carry["opt_state"]
        lambd = carry["lambd"]

        loss, grads = loss_and_grad_fn(parameters, x, y, lambd)

        updates, opt_state = optimizer.update(grads, opt_state, parameters)
        parameters = optax.apply_updates(parameters, updates)

        carry["parameters"] = parameters
        carry["x"], carry["y"] = x, y
        carry["opt_state"] = opt_state

        return carry, loss

    losses = None
    carry = {"parameters": parameters, "x": x, "y": y, "opt_state": opt_state, "lambd": lambd}
    last_carry, losses = jax.lax.scan(one_step, carry, losses, max_iter)
    return last_carry["parameters"], losses


def predict_prob(parameters, x):
    if parameters["weights"].shape[1] > 1:
        z = jnp.dot(parameters["weights"], x.T) + parameters["bias"]
        hypothesis_x = jax.nn.softmax(z, axis=1).T
    else:
        z = jnp.dot(x, parameters["weights"]) + parameters["bias"]
        hypothesis_x = jax.nn.sigmoid(z)
    return hypothesis_x


def score(x, y, parameters):
    n_classes = len(jnp.unique(y))
    if n_classes > 2:
        y_pred = predict_prob(parameters, x)
        y_pred = jnp.argmax(y_pred, axis=1)
        return jnp.sum(y_pred == y) / y.shape[0]
    else:
        y_pred = predict_prob(parameters, x)
        y_pred = (y_pred > 0.5).astype(int)
        return jnp.sum(y_pred[:, 0] == y) / y.shape[0]
