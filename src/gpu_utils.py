"""
GPU-accelerated util functions.
"""
import torch


def mnorm(
    x: torch.TensorType, z: torch.TensorType, M: torch.TensorType, squared=True
) -> torch.TensorType:  # (n, d), (m,d), (d,d) -> (n, m)
    x_norm = ((x @ M) * x).sum(dim=1, keepdims=True)
    if x is z:
        z_norm = x_norm
    else:
        z_norm = ((z @ M) * z).sum(dim=1, keepdims=True)

    z_norm = z_norm.reshape(1, -1)

    distances = (x @ (M @ z.T) * -2) + x_norm + z_norm
    if squared:
        return distances
    else:
        return torch.sqrt(torch.clamp(distances, min=0))


def K_M(x, z, M, L):
    pairwise_distances = mnorm(x, z, M, squared=False)
    pairwise_distances = torch.clamp(pairwise_distances, min=0.0)
    return torch.exp(pairwise_distances * -(1.0 / L))


def K_M_grad(x, z, M, L):
    K = K_M(x, z, M, L)
    dist = mnorm(x, z, M, squared=False)
    dist = torch.where(dist < 1e-4, torch.zeros_like(dist).float(), dist)

    K = K / dist
    K = torch.where(K == float("inf"), torch.zeros_like(K).float(), K)
    return K


def grad_laplace_mat_gpu(X, a, L, M, batch_size=2, norm_control=False, **kwargs):
    """
    Gradient calculation done with einsum notation.

    Parameters
    ----------
    X : torch.Tensor, shape (n, d), all datapoints
    a : torch.Tensor, shape (n, c), solution to the kernel system (alpha)
    L : float, kernel width
    M : torch.Tensor, shape (d, d), metric matrix (M)
    batch_size : int, number of batches to split the gradient into. Doesn't need to be used.
    norm_control : bool, whether to perform norm control on the gradient.
    """
    # sample if X is too large
    if X.shape[0] > 20000:
        num_samples = 20000
        indices = torch.randperm(len(X), device=X.device)[:num_samples]
        z = X[indices]
    else:
        z = X.clone()

    K = K_M_grad(X, z, M, L)

    m, n = K.shape
    n, d = X.shape
    m, d = z.shape
    n, c = a.shape

    aKX = torch.einsum("nc, mn, nd -> mcd", a, K, (X @ M))
    aKz = torch.einsum("nc, mn, md -> mcd", a, K, (z @ M))

    # aKX = torch.einsum('mn, ncd -> mcd', K, a.view(n, c, 1) * torch.einsum('nd, dD -> nD', X, M).view(n, 1, d))
    # aKz = torch.einsum("mn, nc -> mc", K, a).view(m, c, 1) * torch.einsum('md, dD -> mD', z, M).view(n, 1, d)

    G = (aKX - aKz) * (-1.0 / L)

    M = torch.einsum("mcd, mcD -> dD", G, G) / len(G)

    return M

# ChatGPT
# p, d = self.centers.shape
# p, c = self.weights.shape
# n, d = samples.shape

# centers_term = torch.einsum('np, pcd -> ncd', K, self.weights.view(p, c, 1) * torch.einsum('pd, dD -> pcd', self.centers, self.M).view(p, 1, d))

# samples_term = torch.einsum('np, pc -> nc', K, self.weights).view(n, c, 1) * torch.einsum('nd, dD -> ncD', samples, self.M).view(n, 1, d)

# G = (centers_term - samples_term) / self.bandwidth
# self.M = torch.einsum('ncd, ncD -> dD', G, G) / len(samples)

