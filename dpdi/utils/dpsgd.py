"""
Implements DP-SGD algorithm from the paper and related utility functions.
"""
import numpy as np
import torch
from math import sqrt
from math import log as ln
from numpy.random import default_rng


def build_loader(X, y, batch_size=64):
    inputs = torch.from_numpy(X).double()
    targets = torch.from_numpy(y).double()
    train_ds = torch.utils.data.TensorDataset(inputs, targets)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
    return loader


def build_dataset(H_min, H_maj, mu, n: int, alpha: float, w_star: np.ndarray,
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
    n_maj = int(alpha * n)
    n_min = n - n_maj
    if verbose:
        print(f"[INFO] built dataset with alpha={alpha}, n_maj={n_maj}, n_min={n_min}")
    rng = default_rng()
    eta = rng.normal(0., sd_eta, n)
    X_min = rng.multivariate_normal(mu, H_min, n_min)
    X_maj = rng.multivariate_normal(mu, H_maj, n_maj)
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
    """Compute the quantity defined as \rho in the paper, along with its constituents."""
    loss_sgd_0 = np.mean(((X[g == 0, :] @ sgd_w_hat) - y[g == 0]) ** 2)
    loss_sgd_1 = np.mean(((X[g == 1, :] @ sgd_w_hat) - y[g == 1]) ** 2)
    loss_dpsgd_0 = np.mean(((X[g == 0, :] @ dpsgd_w_hat) - y[g == 0]) ** 2)
    loss_dpsgd_1 = np.mean(((X[g == 1, :] @ dpsgd_w_hat) - y[g == 1]) ** 2)
    rho = (loss_dpsgd_0 - loss_dpsgd_1) / (loss_sgd_0 - loss_sgd_1)
    print(f"loss_dpsgd_0: {loss_dpsgd_0}")
    print(f"loss_dpsgd_1: {loss_dpsgd_1}")
    print(f"loss_sgd_0: {loss_sgd_0}")
    print(f"loss_sgd_1: {loss_sgd_1}")
    print("[INFO] rho : {} = {} / {}".format(rho, loss_dpsgd_0 - loss_dpsgd_1,
                                             loss_sgd_0 - loss_sgd_1))
    metrics = {"rho": rho,
               "loss_dpsgd_0": loss_dpsgd_0,
               "loss_dpsgd_1": loss_dpsgd_1,
               "loss_sgd_0": loss_sgd_0,
               "loss_sgd_1": loss_sgd_1}
    return metrics


def print_dpsgd_diagnostics(L_1, L_2, L_3, k, sigma_dp, n, delta):
    """Print various important quantities used to compute sigma_DP."""
    print(f"L_1 = {L_1}; L_2 = {L_2}; L_3 = {L_3}")
    print(f"k={k}")
    print("(L_2 * L_3 + L_1 * L_3**2): %s" % (L_2 * L_3 + L_1 * L_3 ** 2))
    print("sqrt(2 * ln(1.25 * 2 * k / delta)): %s" % sqrt(2 * ln(1.25 * 2 * k / delta)))
    print("sqrt(k * ln(2 * n / delta)): %s" % sqrt(k * ln(2 * n / delta)))
    print("sigma_dp: %f" % sigma_dp)


def dp_sgd(X, y, T, delta, eps, s, lr, w_star, verbose=True, d=2):
    """Implements Algorithm 1 (DP-SGD)."""
    n = len(X)
    # Compute the various constants needed for the algorithm.
    L_1, L_2, L_3, k = compute_L_and_k(X, y, w_star, n, T, delta)
    sigma_dp = compute_sigma_dp(L_1, L_2, L_3, k=k, delta=delta, eps=eps, n=n)
    if verbose:
        print_dpsgd_diagnostics(L_1, L_2, L_3, k=k, sigma_dp=sigma_dp, n=n, delta=delta)

    # Initialization
    loader = build_loader(X, y)
    t = 0
    w_hat = torch.zeros(size=(d,), dtype=torch.double)
    w_hat.requires_grad = True
    lr = torch.Tensor([lr]).double()
    L_3 = torch.Tensor([L_3, ])
    iterates = list()
    losses = list()
    while t < T:
        for i, (X_i, y_i) in enumerate(loader):
            grad_noise = torch.normal(mean=0, std=sigma_dp, size=w_star.shape)
            y_hat = torch.matmul(X_i, w_hat)
            loss = torch.mean((y_i - y_hat) ** 2)
            loss.backward()

            with torch.no_grad():
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
                print("[INFO] completed %s iterations of SGD." % t)
                break
    w_hat = np.vstack(iterates[-(T - s):]).mean(axis=0)
    return iterates, losses, w_hat


def vanilla_sgd(loader, T, lr, d=2, verbose=True):
    """Implements vanilla SGD for the dataset."""
    w_hat = torch.zeros(size=(d,), dtype=torch.double)
    w_hat.requires_grad = True
    lr = torch.Tensor([lr]).double()
    t = 0
    iterates = list()
    losses = list()
    while t < T:
        for i, (X_i, y_i) in enumerate(
                loader):  # Should be iterating over batches, instead of one pass over
            # dataset
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
