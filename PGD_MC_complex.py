import time
import numpy as np
import torch
from scipy.fft import fft2, ifft2, fftshift, ifftshift

def nll_identity(x, y_mul, std_z, eps=1e-12):
    """
    x      : real torch.Tensor, shape (...), on CPU or CUDA
    y_mul  : complex array-like (numpy/list) or torch.Tensor, shape (L, ...)
    std_z  : float or torch scalar/tensor (real)
    """

    device = x.device
    x_dtype = x.dtype  # float32/float64

    # Choose complex dtype consistent with x precision
    y_dtype = torch.complex64 if x_dtype == torch.float32 else torch.complex128

    # ---- y_mul -> complex tensor on x.device ----
    if not torch.is_tensor(y_mul):
        y_mul = torch.as_tensor(y_mul, device=device)
    else:
        y_mul = y_mul.to(device)

    # Ensure it's complex (if it came in as complex numpy, this keeps it complex;
    # if it came in as real accidentally, this upgrades it)
    y_mul = y_mul.to(dtype=y_dtype)

    # ---- std_z -> real tensor on x.device ----
    if not torch.is_tensor(std_z):
        std_z = torch.tensor(std_z, device=device, dtype=x_dtype)
    else:
        std_z = std_z.to(device=device, dtype=x_dtype)

    # ---- Sigma (real) ----
    Sigma = x + std_z**2  # real

    # log det of diagonal Sigma
    nll_1 = torch.sum(torch.log(Sigma + eps))

    # quadratic term: average over looks
    nll_2 = 0.0
    L = y_mul.shape[0]
    for look_idx in range(L):
        y = y_mul[look_idx]
        y_sq = torch.abs(y) ** 2            # real, shape (L, ...)
        nll_2 += torch.sum(y_sq / Sigma)    # real
    nll_2 = nll_2 / L

    return nll_1 + nll_2


def nll_grad_operator(x, y_mul, std_z):
    identity_matrix = np.ones_like(x)
    x_inv = 1.0 / (x + std_z**2 * identity_matrix) 
    DMD_u_hat_v_mean = x_inv

    L = np.shape(y_mul)[0]
    DMD_h_hat_g_mean = np.zeros_like(x)
    for look_idx in range(L):
        y = y_mul[look_idx]
        DMD_h_hat = y * x_inv
        # take squared magnitude is correct, see ICML 2024 eq.(76)
        DMD_h_hat_g_mean += np.square(np.abs(DMD_h_hat))
    DMD_h_hat_g_mean /= L

    grad_matrix = DMD_u_hat_v_mean - DMD_h_hat_g_mean

    return grad_matrix



#################################################################
## (Complex value) efficient operator + MC + conjugate gradient ##
#################################################################

# efficient operator and Monte-Carlo approximation
def nll_grad_operator_MC_CGD(x, y_mul, aperture, std_z, num_ite_MC):

    # new version
    # print('gradient 1st term', 'K=', num_ite_MC) 
    # DMD_u_hat_v_sum = np.zeros_like(x, dtype=np.complex128)
    # for MC_ite in range(num_ite_MC):
    #     v_real = np.random.normal(0.0, 1.0, size=x.shape)
    #     v_imag = np.random.normal(0.0, 1.0, size=x.shape)
    #     v = (v_real + 1j * v_imag) / np.sqrt(2.0)
    #     v_conj = np.conjugate(v)
    #     Av = A_operator(v, aperture)
    #     u = conjugate_gradient(x, aperture, Av, std_z, x0=None, tol=1e-6, max_iter=1000)
    #     u_hat = u.reshape(x.shape)
    #     DMD_u_hat = A_operator(u_hat, aperture)
    #     DMD_u_hat_v_sum += DMD_u_hat * v_conj
    # # take real part only *after* averaging (ICML eq.76)
    # DMD_u_hat_v_mean_complex = DMD_u_hat_v_sum / num_ite_MC
    # DMD_u_hat_v_mean = DMD_u_hat_v_mean_complex.real

    # old version
    print('gradient 1st term', 'K =', num_ite_MC)
    DMD_u_hat_v_sum = np.zeros_like(x, dtype=np.complex64)
    CG_iter_1, CG_iter_2 = 0, 0
    for MC_ite in range(num_ite_MC):
        v = np.random.normal(0, 1, size=np.shape(x))
        Av = A_operator(v, aperture)
        # conjugate gd
        u, CG_iter_1_ite = conjugate_gradient(x, aperture, Av, std_z, x0=None, tol=1e-6, max_iter=1000)
        CG_iter_1 += CG_iter_1_ite
        # use FFT operator
        u_hat = np.reshape(u, np.shape(x))
        DMD_u_hat = A_operator(u_hat, aperture)
        DMD_u_hat_v = DMD_u_hat * v
        DMD_u_hat_v_sum += DMD_u_hat_v  # diagonal of Hermitian A(AXA^H)^-1A^H is real value
    DMD_u_hat_v_mean = DMD_u_hat_v_sum.real / num_ite_MC

    L = np.shape(y_mul)[0]
    print('gradient 2nd term', 'L =', L)
    DMD_h_hat_g_mean = np.zeros_like(x)
    for look_idx in range(L):
        y = y_mul[look_idx]
        h, CG_iter_2_ite = conjugate_gradient(x, aperture, y, std_z, x0=None, tol=1e-6, max_iter=1000)
        CG_iter_2 += CG_iter_2_ite
        h_hat = np.reshape(h, np.shape(x))
        DMD_h_hat = A_operator(h_hat, aperture)
        # take squared magnitude is correct, see ICML 2024 eq.(76)
        DMD_h_hat_g_mean += np.square(np.abs(DMD_h_hat))
    DMD_h_hat_g_mean /= L

    # real-valued is correct in gradient, see ICML 2024 eq.(76)
    grad_matrix = DMD_u_hat_v_mean - DMD_h_hat_g_mean

    CG_iter_1 /= num_ite_MC
    CG_iter_2 /= L
    return grad_matrix, CG_iter_1, CG_iter_2 

# Forward operator implementation of A = F^-1 M F
def A_operator(h, aperture):
    # consider A=F^H M F
    D_h = fftshift(fft2(h))
    MD_h = D_h * aperture
    DMD_h = ifft2(ifftshift(MD_h))
    return DMD_h

# efficient operator implementation of AXA^H + sigma^2_z*I, A = A^H
def B_operator(h, x, aperture, std_z):
    # consider B = A X A^H + sigma^2 I
    DMD_h = A_operator(h, aperture)
    x2_DMD_h = x * DMD_h
    DMD_x2_DMD_h = A_operator(x2_DMD_h, aperture)
    DMD_x2_DMD_h += std_z**2 * h # consider additive term
    return DMD_x2_DMD_h

# Conjugate Gradient Algorithm to solve Ax = b
def conjugate_gradient(x, aperture, b, std_z, x0=None, tol=1e-6, max_iter=1000):
    b = np.asarray(b)
    shape = b.shape
    # Initial guess
    if x0 is None:
        u = np.zeros_like(b, dtype=np.complex128)
    else:
        u = np.asarray(x0, dtype=np.complex128).copy()
    # Helper applying B and flattening
    def B_flat(v_flat):
        v = v_flat.reshape(shape)
        Bv = B_operator(v, x, aperture, std_z)  # this returns same shape as v
        return Bv.reshape(-1)
    # Initial residual r = b - B u
    r = (b - B_operator(u, x, aperture, std_z)).reshape(-1)
    p = r.copy()
    # Hermitian inner product <r, r> = sum conj(r_i) * r_i
    rs_old = np.vdot(r, r).real  #  complex scalar; for SPD operator this is real ≥ 0
    for i in range(max_iter):

        CG_iter_start_time = time.time()

        Ap = B_flat(p)
        denom = np.vdot(p, Ap).real
        # avoid division by zero
        if np.abs(denom) < 1e-30:
            break
        alpha = rs_old / denom    # alpha = <r, r> / <p, Ap>
        # Update solution and residual
        u = u + alpha * p.reshape(shape)
        r = r - alpha * Ap
        rs_new = np.vdot(r, r).real
        # Convergence check: norm(r) = sqrt(<r,r>)
        if np.sqrt(rs_new.real) < tol:
            break
        beta = rs_new / rs_old
        p = r + beta * p
        rs_old = rs_new

        CG_iter_end_time = time.time()
        # print('CG 1 iter time:', CG_iter_end_time - CG_iter_start_time)

    return u, i+1  # same shape as b

