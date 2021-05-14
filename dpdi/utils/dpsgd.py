"""
Implements DP-SGD algorithm from the paper and related utility functions.
"""
import numpy as np
import torch
from math import sqrt
from math import log as ln
from numpy.random import default_rng
from sklearn.linear_model import LinearRegression

RANDOM_SEED = 983445


def get_wstar(df):
    """Compute the parameters via OLS (with no intercept term).

    If an intercept term is desired, add an 'intercept' column to the
        design matrix of all ones.
    """
    lr = LinearRegression(fit_intercept=False)\
    .fit(X=df.drop(['sensitive', 'target'],axis=1), y=df['target'])
    return lr.coef_


def compute_mse(X, y, w_hat):
    test_preds = X @ w_hat
    test_err = np.mean((y - test_preds) ** 2)
    return test_err


def build_loader(X, y, batch_size=64, shuffle=False):
    inputs = torch.from_numpy(X).double()
    targets = torch.from_numpy(y).double()
    train_ds = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle)
    return loader


def maybe_cast_scalar_to_square_ary(a: np.array) -> np.array:
    """Utility function to give a scalar a square shape."""
    if len(a) == 1 and np.ndim(a) == 1:
        return a.reshape(1, 1)
    else:
        return a


def build_dataset(H_min: np.array, H_maj: np.array, mu_min: np.array, mu_maj: np.array,
                  n: int, alpha: float, w_star: np.ndarray,
                  sd_eta: float = 1., verbose=True):
    """
    Build a dataset according to the parameters.
    H_0:
    mu: vector of group means [mu_min, mu_maj].
    n: total number of samples.
    alpha: Fraction of samples from the majority group.
    sd_eta: sd of the noise in the model, such that y = x^T w_star + eta.
    """
    assert 0 < alpha < 1
    assert (H_min.shape == H_maj.shape) and (mu_min.shape == mu_maj.shape)
    n_maj = int(alpha * n)
    n_min = n - n_maj
    if verbose:
        print(f"[INFO] built dataset with alpha={alpha}, n_maj={n_maj}, n_min={n_min}")
    rng = default_rng()
    eta = rng.normal(0., sd_eta, n)

    H_min = maybe_cast_scalar_to_square_ary(H_min)
    H_maj = maybe_cast_scalar_to_square_ary(H_maj)
    X_min = rng.multivariate_normal(mu_min, H_min, n_min)
    X_maj = rng.multivariate_normal(mu_maj, H_maj, n_maj)
    X = np.vstack((X_min, X_maj))
    # The ith entry in g is zero if the ith element of X is from minority; otherwise one.
    g = np.concatenate((np.zeros(n_min), np.ones(n_maj)))
    y = (X @ w_star) + eta
    return X, g, y


def compute_L_and_k(X, y, w_star, n, T, delta):
    c = 2  # TODO:confirm value of c.
    L_1 = np.linalg.norm(X, axis=1, ord=2).max()
    L_2 = np.abs(y).max()
    L_3 = np.linalg.norm(w_star, ord=2)
    k = float(T) / n + c * sqrt((T / n) * ln(2 * n / delta))
    return L_1, L_2, L_3, k


def compute_sigma_dp(L_1, L_2, L_3, k, delta, eps: float, n: int):
    """Compute sigma_DP."""
    sigma_dp = (2 * (L_2 * L_3 + L_1 * L_3 ** 2)
                * sqrt(2 * ln(1.25 * 2 * k / delta))
                * sqrt(k * ln(2 * n / delta))
                / eps)
    return sigma_dp


def compute_disparity(X: np.array, g: np.array, y: np.array, sgd_w_hat: np.array,
                      dpsgd_w_hat: np.array):
    """Compute the quantities defined as \rho and \hi in the paper, along with their
    constituents."""
    loss_sgd_0 = np.mean(((X[g == 0, :] @ sgd_w_hat) - y[g == 0]) ** 2)
    loss_sgd_1 = np.mean(((X[g == 1, :] @ sgd_w_hat) - y[g == 1]) ** 2)
    loss_dpsgd_0 = np.mean(((X[g == 0, :] @ dpsgd_w_hat) - y[g == 0]) ** 2)
    loss_dpsgd_1 = np.mean(((X[g == 1, :] @ dpsgd_w_hat) - y[g == 1]) ** 2)
    rho = (loss_dpsgd_0 - loss_dpsgd_1) / (loss_sgd_0 - loss_sgd_1)
    phi = (loss_dpsgd_0 - loss_sgd_0) / (loss_dpsgd_1 - loss_sgd_1)
    print(f"loss_dpsgd_0: {loss_dpsgd_0}")
    print(f"loss_dpsgd_1: {loss_dpsgd_1}")
    print(f"loss_sgd_0: {loss_sgd_0}")
    print(f"loss_sgd_1: {loss_sgd_1}")
    print("[INFO] rho : {} = {} / {}".format(rho, loss_dpsgd_0 - loss_dpsgd_1,
                                             loss_sgd_0 - loss_sgd_1))
    print("[INFO] phi : {} = {} / {}".format(phi, loss_dpsgd_0 - loss_sgd_0,
                                             loss_dpsgd_1 - loss_sgd_1))
    metrics = {"rho": rho,
               "phi": phi,
               "loss_dpsgd_0": loss_dpsgd_0,
               "loss_dpsgd_1": loss_dpsgd_1,
               "loss_sgd_0": loss_sgd_0,
               "loss_sgd_1": loss_sgd_1}
    return metrics


def compute_rho_lr(H_min: np.array, H_maj: np.array, alpha, sigma_dp, sigma_noise=1.):
    """Compute the quantity defined as \rho_{LR} in the paper."""
    H_min = maybe_cast_scalar_to_square_ary(H_min)
    H_maj = maybe_cast_scalar_to_square_ary(H_maj)
    H = alpha * H_maj + (1 - alpha) * H_min
    H_inv = np.linalg.pinv(H)
    H_minus2 = np.matmul(H_inv, H_inv)
    rho_lr = (
            np.trace((H_min - H_maj) @ H_minus2) * sigma_dp ** 2
            / (np.trace((H_min - H_maj) @ H_inv) * sigma_noise ** 2)
    )
    return rho_lr


def print_dpsgd_diagnostics(L_1, L_2, L_3, k, sigma_dp, n, delta):
    """Print various important quantities used to compute sigma_DP."""
    print(f"L_1 = {L_1}; L_2 = {L_2}; L_3 = {L_3}")
    print(f"k={k}")
    print("(L_2 * L_3 + L_1 * L_3**2): %s" % (L_2 * L_3 + L_1 * L_3 ** 2))
    print("sqrt(2 * ln(1.25 * 2 * k / delta)): %s" % sqrt(2 * ln(1.25 * 2 * k / delta)))
    print("sqrt(k * ln(2 * n / delta)): %s" % sqrt(k * ln(2 * n / delta)))
    print("sigma_dp: %f" % sigma_dp)


def tail_average_iterates(w, T, s):
    return np.vstack(w[-(T - s):]).mean(axis=0)


def dp_sgd(X, y, T, delta, eps, s, lr, w_star, verbose=True, batch_size=64,
           random_seed=RANDOM_SEED):
    """Implements Algorithm 1 (DP-SGD), with fixed seed for reproducibility."""
    torch.manual_seed(random_seed)
    n, d = X.shape
    assert d == len(w_star), "shape mismatch between X and w_star"
    # Compute the various constants needed for the algorithm.
    L_1, L_2, L_3, k = compute_L_and_k(X, y, w_star, n, T, delta)
    sigma_dp = compute_sigma_dp(L_1, L_2, L_3, k=k, delta=delta, eps=eps, n=n)
    print_dpsgd_diagnostics(L_1, L_2, L_3, k=k, sigma_dp=sigma_dp, n=n, delta=delta)

    # Initialization
    loader = build_loader(X, y, batch_size)
    t = 0
    w_hat = torch.zeros(size=(d,), dtype=torch.double)
    w_hat.requires_grad = True
    lr = torch.Tensor([lr]).double()
    L_3 = torch.Tensor([L_3, ])
    iterates = list()
    losses = list()
    while t < T:
        for i, (X_i, y_i) in enumerate(loader):

            y_hat = torch.matmul(X_i, w_hat)
            loss = torch.mean((y_i - y_hat) ** 2)
            loss.backward()

            with torch.no_grad():
                grad_noise = torch.normal(mean=0, std=sigma_dp, size=w_star.shape)
                w_hat -= lr * (w_hat.grad + grad_noise)

                # Project back onto ball of radius L_3
                w_hat_norm = torch.norm(w_hat)
                w_hat /= w_hat_norm
                w_hat *= L_3

                w_hat.grad.zero_()
                iterate_numpy = w_hat.clone().detach().numpy()
                loss_numpy = loss.clone().detach().numpy()
                iterates.append(iterate_numpy)
                losses.append(loss_numpy)

            if verbose and (t % 1000 == 0):
                print(
                    "iteration {} loss: {} new w_hat: {}".format(t, loss, iterate_numpy))

            t += 1
            if t >= T:
                print("[INFO] completed %s iterations of DP-SGD." % t)
                break
    w_hat_bar = tail_average_iterates(iterates, T, s)
    return iterates, losses, w_hat_bar


def tail_averaged_sgd(X, y, T, s, lr, verbose=True, batch_size=64,
                      random_seed=RANDOM_SEED):
    """Implements tail-averaged SGD. This is DP-SGD but with no projection step and no
    noise."""
    torch.manual_seed(random_seed)
    n, d = X.shape

    # Initialization
    loader = build_loader(X, y, batch_size)
    t = 0
    w_hat = torch.zeros(size=(d,), dtype=torch.double)
    w_hat.requires_grad = True
    lr = torch.Tensor([lr]).double()

    iterates = list()
    losses = list()
    while t < T:
        for i, (X_i, y_i) in enumerate(loader):

            y_hat = torch.matmul(X_i, w_hat)
            loss = torch.mean((y_i - y_hat) ** 2)
            loss.backward()

            with torch.no_grad():
                w_hat -= lr * w_hat.grad

                w_hat.grad.zero_()
                iterate_numpy = w_hat.clone().detach().numpy()
                loss_numpy = loss.clone().detach().numpy()
                iterates.append(iterate_numpy)
                losses.append(loss_numpy)

            if verbose and (t % 1000 == 0):
                print(
                    "iteration {} loss: {} new w_hat: {}".format(t, loss, iterate_numpy))

            t += 1
            if t >= T:
                print("[INFO] completed %s iterations of SGD." % t)
                break
    w_hat_bar = tail_average_iterates(iterates, T, s)
    return iterates, losses, w_hat_bar


def vanilla_sgd(X, y, T, lr, verbose=True, batch_size=64, random_seed=RANDOM_SEED):
    """Implements vanilla SGD for the dataset."""
    print("[WARNING] this function is deprecated and provided only for reproducibility.")
    print("Use tail_averaged_sgd() instead (note different return types and signatures).")
    torch.manual_seed(random_seed)
    n, d = X.shape
    loader = build_loader(X, y, batch_size)
    w_hat = torch.zeros(size=(d,), dtype=torch.double)
    w_hat.requires_grad = True
    lr = torch.Tensor([lr]).double()
    t = 0
    iterates = list()
    losses = list()
    while t < T:
        for i, (X_i, y_i) in enumerate(loader):
            y_hat = torch.matmul(X_i, w_hat)
            loss = torch.mean((y_i - y_hat) ** 2)
            # Computes the gradients for all tensors with grad=True
            loss.backward()
            with torch.no_grad():
                w_hat -= lr * w_hat.grad
                w_hat.grad.zero_()
                iterate_numpy = w_hat.clone().detach().numpy()
                loss_numpy = loss.clone().detach().numpy()
                iterates.append(iterate_numpy)
                losses.append(loss_numpy)
            if verbose and (t % 1000 == 0):
                print("iteration {} loss: {} new w_hat: {}".format(t, loss,
                                                                   w_hat.detach().numpy()))
            t += 1
            if t >= T:
                print("[INFO] completed %s iterations of SGD." % t)
                break
    return iterates, losses
