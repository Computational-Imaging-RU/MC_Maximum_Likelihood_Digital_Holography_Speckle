import time
import numpy as np
import torch
import torch.nn.functional as F
from bm3d import bm3d, BM3DStages
from PGD_MC_complex import A_operator


###############################################################
# (Complex value) Implementation of C-PnP EM in Casey's papers #
###############################################################

def nll_surrogate_F1(x, x_in, mu_square, C, std_F1):
    L_F1_value = np.sum((C + mu_square) / x + np.log(x) + (x**2 - 2*x_in*x + x_in**2) / 2*std_F1**2)
    return L_F1_value

def nll_surrogate_F1_coord(x, x_in, mu_square, C, std_F1):
    L_F1_value = (C + mu_square) / x + np.log(x) + (x**2 - 2*x_in*x + x_in**2) / 2*std_F1**2
    return L_F1_value

def E_step(x, y_mul, aperture, std_z):
    C = (x * std_z**2)/(std_z**2 + x)
    L = np.shape(y_mul)[0]
    mu_square_sum = np.zeros_like(x)
    for look_idx in range(L):
        y = y_mul[look_idx]
        Ay = A_operator(y, aperture)
        mu_square = (C**2 * np.square(np.abs(Ay))) / std_z**4
        mu_square_sum += mu_square
    mu_square_mean = mu_square_sum / L
    return mu_square_mean, C

def solve_r_minF_coord(r_in_i, mu_square_i, C_ii, std_F1, tol=1e-10):
    """
    Solve cubic equation r^3 - r_in*r^2 + sigma2*r - sigma2*K = 0 for r.
    Keep only real roots and return the one that minimizes nll_surrogate_F1_coord.
    """
    K = mu_square_i + C_ii
    coeffs = [1.0, - r_in_i, std_F1**2, - K * std_F1**2]
    roots = np.roots(coeffs)
    # Keep only (approximately) real roots
    real_roots = roots.real[np.isclose(roots.imag, 0.0, atol=tol)]
    # Keep only positive roots (> tol)
    pos_roots = real_roots[real_roots > 0.0]
    if pos_roots.size == 0:
        raise ValueError("No positive real roots found for given parameters.")
    # Evaluate surrogate function on positive roots
    vals = np.array([nll_surrogate_F1_coord(r, r_in_i, mu_square_i, C_ii, std_F1) for r in pos_roots])
    best_root = pos_roots[np.argmin(vals)]
    return best_root

def F1_minimizer(r_in_1, mu_square, C, std_F1):
    n = np.shape(r_in_1)[0]
    r1 = np.zeros_like(r_in_1)
    for i in range(n):
        r_in_1_i, mu_square_i, C_ii = r_in_1[i], mu_square[i], C[i]
        r_1_i = solve_r_minF_coord(r_in_1_i, mu_square_i, C_ii, std_F1)
        r1[i] = r_1_i
    return r1

def F1_grad(r_in_1_init, r_in_1, mu_square, C, std_F1):
    gradient_vec = - (C + mu_square)/r_in_1_init**2 + 1/r_in_1_init + (r_in_1_init-r_in_1)/std_F1**2
    return gradient_vec

def F1_GD(r_in_1_init, r_in_1, mu_square, C, std_F1, F1_ite, lr_F1):
    r_in_1_init = np.reshape(r_in_1_init, (-1,))
    r1_new = r_in_1_init
    for gd_ite in range(F1_ite):
        r1_new -= lr_F1 * F1_grad(r1_new, r_in_1, mu_square, C, std_F1)
    r1 = r1_new
    return r1

def F2_minimizer_DnCNN(r_in_2, denoiser):
    device = next(denoiser.parameters()).device
    H = W = int(np.sqrt(len(r_in_2)))
    r_in_2_img = np.reshape(r_in_2, (H, W))
    r_in_2_tensor = torch.as_tensor(r_in_2_img, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        r2_tensor, _ = denoiser(r_in_2_tensor)
    r2_img = r2_tensor.squeeze().detach().cpu().numpy()
    r2_img = np.clip(r2_img,0,1)
    r2 = np.reshape(r2_img, (H * W,))
    return r2

def F2_minimizer_DIP(r_in_2, denoiser_set, inner_ite):
    denoiser, latent_code, optimizer = denoiser_set[0], denoiser_set[1], denoiser_set[2]
    device = next(denoiser.parameters()).device
    H = W = int(np.sqrt(len(r_in_2)))
    r_in_2_img = np.reshape(r_in_2, (H, W))
    r_in_2_tensor = torch.as_tensor(r_in_2_img, dtype=torch.float64, device=device).unsqueeze(0).unsqueeze(0)

    for ee in range(inner_ite):
        denoiser.train()
        optimizer.zero_grad()
        x_gen_tensor = denoiser(latent_code)
        loss_train = F.mse_loss(x_gen_tensor, r_in_2_tensor.detach())
        loss_train.backward()
        optimizer.step()
    with torch.no_grad():
        r2_tensor = denoiser(latent_code).squeeze(0).squeeze(0)
        r2_img = r2_tensor.squeeze().detach().cpu().numpy()
        r2 = np.reshape(np.clip(r2_img, 0, 1), (H * W,))
        r2 = r2.astype(r_in_2.dtype, copy=False)
    return r2

def F2_minimizer_BM3D(r_in_2, sigma=0.1, profile: str = "np", stage: str = "all"):

    H = W = int(np.sqrt(len(r_in_2)))
    assert H * W == len(r_in_2)

    img = np.reshape(r_in_2.astype(np.float32), (H, W))
    img = np.clip(img, 0.0, 1.0)

    # Map strings to enums
    stage_map = {
        "all": BM3DStages.ALL_STAGES,
        "hard": BM3DStages.HARD_THRESHOLDING,
        "wiener": BM3DStages.WIENER_FILTERING,
    }
    stage_arg = stage_map.get(stage, BM3DStages.ALL_STAGES)
    den = bm3d(img, sigma_psd=float(sigma), stage_arg=stage_arg)
    den = np.clip(den, 0.0, 1.0).astype(np.float32)
    return den.reshape(H * W,)

def G_operator(z):
    # z is shape (2n,)
    n = z.shape[0] // 2
    z1, z2 = z[:n], z[n:]
    zbar = 0.5 * (z1 + z2)            # elementwise average
    return np.concatenate([zbar, zbar], axis=0)

def M_Step(r_in_1, r_in_2, mu_square, C, rho, denoiser, std_F1, denoiser_name, DIP_ite):
    
    M_F1_start_time = time.time()
    H, W = np.shape(r_in_1)[0], np.shape(r_in_1)[1]
    r_in_1, r_in_2, mu_square, C = np.reshape(r_in_1, (-1,)), np.reshape(r_in_2, (-1,)), np.reshape(mu_square, (-1,)), np.reshape(C, (-1,))
    r_in = np.concatenate((r_in_1, r_in_2))
    w1 = F1_minimizer(r_in_1, mu_square, C, std_F1)
    M_F1_end_time = time.time()
    print('M step F1 time:', M_F1_end_time - M_F1_start_time)
    # w1 = F1_GD(r_in_1, mu_square, C, F1_ite, lr_F1)
    # w1 = F1_GD(r_bar_prime, r_in_1, mu_square, C, std_F1, F1_ite, lr_F1)
    if denoiser_name == 'DnCNN':
        w2 = F2_minimizer_DnCNN(r_in_2, denoiser)
    elif denoiser_name == 'DIP':
        w2 = F2_minimizer_DIP(r_in_2, denoiser, DIP_ite)
    elif denoiser_name == 'BM3D':
        w2 = F2_minimizer_BM3D(r_in_2, sigma=denoiser)
    w = np.concatenate((w1, w2))
    r = r_in + 2 * rho * (G_operator(2 * w - r_in) - w)
    n = r.shape[0] // 2
    r1 = r[:n]
    r2 = r[n:]

    r_bar_prime = (r1 + r2)/2
    r_bar_prime, r1, r2 = np.reshape(r_bar_prime,(H,W)), np.reshape(r1,(H,W)), np.reshape(r2,(H,W))
    w1, w2 = np.reshape(w1, (H, W)), np.reshape(w2, (H, W))

    return r_bar_prime, r1, r2, w1, w2


