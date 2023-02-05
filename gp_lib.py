from functools import partial as _partial

import numpy as np
import scipy.stats as scystat
import matplotlib.pyplot as mplt
from matplotlib.animation import FuncAnimation

def feature_func_polynomial(x, num_features=2):
    """."""
    return x.ravel()[:, None]**(np.arange(num_features)[None, :])


feature_func_linear = _partial(feature_func_polynomial, num_features=2)


def feature_func_sines(x, num_freqs=6, L=1):
    mat = np.empty((x.size, num_freqs*2-1), dtype=float)
    x_mat = x.ravel()[:, None]/L
    x_mat = x_mat * np.arange(num_freqs)[None, :]
    mat[:, :num_freqs] = np.sin(x_mat)
    mat[:, num_freqs:] = np.cos(x_mat[:, 1:])
    return mat


def feature_func_gauss(x, mean_range=6, sigma=1):
    means = np.arange(-mean_range, mean_range+1)
    mat = np.exp(-(x.ravel()[:, None]-means[None, :])**2/2/sigma**2)
    return mat


def feature_func_steps(x, step_range=6):
    steps = np.arange(-step_range, step_range+1)
    mat = (x.ravel()[:, None] > steps[None, :])*2 - 1
    return mat


def _prepare_input(x1, x2, leng=None):
    x1 = np.array(x1, ndmin=2)
    x2 = np.array(x2, ndmin=2)
    if leng is not None:
        leng = np.array(leng, ndmin=1)
        x1 /= leng[:, None]
        x2 /= leng[:, None]
    return x1, x2


def _calc_distance_squared(x1, x2, leng):
    x1, x2 = _prepare_input(x1, x2, leng)
    dx = x1[..., None] - x2[..., None, :]
    return np.einsum('i...,i...->...', dx, dx)


def kernel_squared_exponential(x1, x2, leng=1, sigma=1):
    """This is the most common kernel.
    
    It in infinitely differentiable and its samples are very smooth.
    It is the result of the parametric regression of equally distributed and
    equally spaced gaussians.
    """
    dx2 = _calc_distance_squared(x1, x2, leng*np.sqrt(2))
    return sigma*np.exp(-dx2)


def kernel_exponential(x1, x2, leng=1, sigma=1):
    """This kernel results in continuous but not differentiable functions.
    
    Its samples are compatible with the velocity of a particle undergoing
    brownian motion.
    """
    dx = np.sqrt(_calc_distance_squared(x1, x2, leng))
    return sigma*np.exp(-dx)


def kernel_sine_squared(x1, x2, leng=1, sigma=1, period=1):
    """This kernel is valid to model periodic functions."""
    dx = np.sqrt(_calc_distance_squared(x1, x2, leng=1))
    sin2_dx = np.sin((np.pi/period)*dx)
    sin2_dx *= sin2_dx
    return sigma*np.exp(-(2/leng/leng)*sin2_dx)
#     # The implementation below makes use of sin(x)^2 = (1-cos(2x))/2
#     # it might be faster then the one above
#     msin2_dx = np.cos((2*np.pi/period)*dx)
#     msin2_dx -= 1
#     return sigma * np.exp((1/leng/leng)*msin2_dx)


def kernel_rational_quadratic(x1, x2, leng=1, sigma=1, alpha=1):
    """This kernel is the sum of infinite squared exponential kernels.
    
    It is good to capture phenomena with different lengthscales.
    The parameter alpha controls the mixture of the summing kernels, where
    alpha -> infinity recovers the single squared exponential with lenghtscale
    given by leng.
    """
    dx2 = _calc_distance_squared(x1, x2, leng*np.sqrt(2*alpha))
    return (1 + dx2)**(-alpha)


def kernel_from_parametric_function(x1, x2, feature_func, cov):
    """This kernel put parametric and non-parametric regression.
    
    This is a consequence of the fact that parametric regression using 
    Bayesian Inference is a particular case of non-parametric regression.
    """
    ph1 = feature_func(x1)
    ph2 = feature_func(x2)
    return ph1 @ cov @ ph2.T


def parametric_bayesian_regression(
        x, x_data, y_data, prior, feature_func, sigma_err=1):
    phi_x = feature_func(x)
    phi_data = feature_func(x_data)
    num_features = phi_data.shape[1]
    mu_prior = prior.mean[:, None]
    sigma_prior = prior.cov_object.covariance

    muf_data = phi_data @ mu_prior
    kXX = phi_data @ sigma_prior @ phi_data.T  # kXX
    err_data = y_data[:, None]-muf_data

    if x_data.size < num_features:
        # this method has the advantage of easy interpretation of a 
        # joint distribution between measured data and infered data and also
        # shows more clearly the importance of the kernel function
        kXXi = np.linalg.inv(kXX + np.eye(x_data.size)*sigma_err**2)
        B = sigma_prior @ phi_data.T
        mu_post = mu_prior + B @ kXXi @ err_data
        sigma_post = sigma_prior - B @ kXXi @ B.T
    else:
        # this method is optimized for larger sets of data, due to the fact 
        # that the matrix inversion is done in the parameters covariance 
        # matrix, which is generally smaller than the data matrix.
        A = phi_data.T @ phi_data/sigma_err**2 + np.linalg.inv(sigma_prior)
        sigma_post = np.linalg.inv(A)
        mu_post = mu_prior + 1/sigma_err**2*sigma_post @ phi_data.T @ err_data
    return scystat.multivariate_normal(mu_post.ravel(), sigma_post)


def animate_bayes(frames, mu_x, sigma_x, x, prior):
    post_pdf = prior.copy()

    # Plot the prior PDF
    fig, ax = mplt.subplots(1, 1, figsize=(5, 3))
    ax.plot(x, prior, label='Prior')
    ax.set_xlabel('y')
    ax.set_ylabel('Probability density')
    ax.legend(loc='best')
    fig.tight_layout()

    line, = ax.plot([0], [0])
    lines = ax.plot([0], np.array([[0]]*len(frames)).T)
    for l in lines:
        l.set_visible(False)
    def update(i, post):
        mux = mu_x[i]
        sigx = sigma_x[i]
        # Calculate the likelihood PDF
        likelihood_pdf = scystat.norm.pdf(x, mux, sigx)

        # Plot the likelihood PDF
        lines[i].set_data(x, likelihood_pdf)
        lines[i].set_label(f'y{i:d}~N({mux:.1f},{sigx:.1f})')
        lines[i].set_visible(True)

        # Multiply the prior PDF and the likelihood PDF to 
        # get the unnormalized posterior PDF
        post *= likelihood_pdf

        # Normalize the unnormalized posterior PDF
        norm_constant = np.trapz(post, x)
        post /= norm_constant

        # Plot the posterior PDF
        line.set_data(x, post)
        line.set_label(f'Posterior {i:d}')
        ax.legend(loc='best', fontsize='x-small')
        ax.relim()
        ax.autoscale_view()
        fig.tight_layout()
        return []

    return FuncAnimation(
        fig, update, fargs=(post_pdf, ), frames=frames,
        repeat=False, repeat_delay=2, interval=1000, init_func=lambda: [])


def animate_regression(
        x, x_data, y_data, prior, feature_func, sigma_err=1, truth=None):

    post = parametric_bayesian_regression(
        x, x_data, y_data, prior, feature_func, sigma_err)
    dist = prior
    phix = feature_func(x)
    muf = phix @ dist.mean
    stdf = phix @ dist.cov_object.covariance @ phix.T
    stdf = np.sqrt(np.diag(stdf))
    
    is2d = len(dist.mean) == 2
    if is2d:
        fig, (ax, ay) = mplt.subplots(1, 2, figsize=(7, 3))

        wgrid = np.linspace(-6, 6, 100)
        w1grid, w2grid = np.meshgrid(wgrid, wgrid)
        pos = np.empty(w1grid.shape + (2, ))
        pos[:, :, 0] = w1grid
        pos[:, :, 1] = w2grid
    else:
        fig, ay = mplt.subplots(1, 1, figsize=(5, 3))
   
    def animate(frm):
        dist = parametric_bayesian_regression(
            x, x_data[:frm], y_data[:frm], prior, feature_func, sigma_err)
        if is2d:
            pdf = dist.pdf(pos)
            ax.clear()
            ax.grid(False)
            levels = pdf.max()*np.array([0.1, 0.4, 0.7, 0.95])
            ax.contour(
                w1grid, w2grid, pdf, cmap='copper', levels=levels)
            ax.pcolormesh(w1grid, w2grid, pdf, cmap='copper', alpha=0.5)
            if truth is not None and not isinstance(truth[0], np.ndarray):
                ax.plot(*truth, 'ko', label='Truth')

            ax.legend(loc='lower right')
            ax.set_xlabel('w1 (intercept)')
            ax.set_ylabel('w2 (angular coeff)')
            ax.set_xlim([-3, 3])
            ax.set_ylim([-3, 3])
            
        muf = phix @ dist.mean
        stdf = phix @ dist.cov_object.covariance @ phix.T
        stdf = np.sqrt(np.diag(stdf))
        
        ay.clear()
        ay.set_xlabel('x')
        ay.set_ylabel('y = f(x) + epsilon')
        if truth is not None:
            if not isinstance(truth[0], np.ndarray):
                ay.plot(x, phix@truth, 'k--', label='Truth')
            else:
                ay.plot(truth[0], truth[1], 'k--', label='Truth')
        
        ay.plot(x, muf, label='expected f(x)')
        ay.fill_between(
            x, muf+1.96*stdf, muf-1.96*stdf, color='C0', alpha=0.2,
            label='95% confidence')
        ay.errorbar(
            x_data[:frm], y_data[:frm], yerr=sigma_err, linestyle='',
            marker='o', color='k', barsabove=True, label='Data')
        ay.legend(loc='best', fontsize='x-small')
        fig.tight_layout()
        return []
    
#     return animate(0)
    return FuncAnimation(
        fig, animate, frames=np.arange(x_data.size+1),
        repeat=True, repeat_delay=1000, interval=1000)


def animate_distribution(x, dist, feature_func):
    """."""
    frames = np.arange(40)
    phix = feature_func(x)
    muf = phix @ dist.mean
    stdf = np.sqrt(np.diag(phix @ dist.cov_object.covariance @ phix.T))

    # Create rotating samples from the distribution:
    # First define the levels we want:
    lev_smpl = -2*np.log([0.9, 0.7, 0.5, 0.36])
    # Create normalized random vectors for rotation:
    N = phix.shape[1]
    nvec = lev_smpl.size
    v = np.random.randn(N, nvec, 2)  # create 2 lists of nvec random vectors
    v[..., 0] /= np.linalg.norm(v[..., 0], axis=0)  # normalize first list
    # make ith vector of second list orthogonal to ith vector of first list:
    v[..., 1] -= v[..., 0]*np.sum(v[..., 0]*v[..., 1], axis=0)
    v[..., 1] /= np.linalg.norm(v[..., 1], axis=0)  # normalize second list

    # Create transformation matrix from v to w
    L = np.linalg.cholesky(dist.cov_object.covariance)
    # Create the parametrized rotation matrix:
    R = lambda x: np.array([[np.cos(x)], [np.sin(x)]])
    
    is2d = len(dist.mean) == 2
    if is2d:
        fig, (ax, ay) = mplt.subplots(1, 2, figsize=(7, 3))

        sigm = max(*np.diag(dist.cov_object.covariance), 0.5)
        lims1 = dist.mean[0] - 2.1*sigm, dist.mean[0] + 2.1*sigm
        lims2 = dist.mean[1] - 2.1*sigm, dist.mean[1] + 2.1*sigm
        
        w1grid, w2grid = np.meshgrid(
            np.linspace(*lims1, 100), np.linspace(*lims2, 100))
        pos = np.empty(w1grid.shape + (2, ))
        pos[:, :, 0] = w1grid
        pos[:, :, 1] = w2grid
        pdf = dist.pdf(pos)
        lev_cont = pdf.max()*np.array([0.1, 0.4, 0.7, 0.95])
    else:
        fig, (ax, ay) = mplt.subplots(
            2, 1, figsize=(5, 3), sharex=True, height_ratios=[1, 3])
        ax.plot(x, phix, 'k', lw=1)
        ax.set_ylabel('Basis')
   
    def animate(frame):
        # Rotate and transform the vector
        theta = 2*np.pi * frame/frames.size
        vr = (v @ R(theta)).squeeze()
        vr *= lev_smpl[None, :]
        wr = dist.mean[:, None] + L @ vr

        if is2d:
            ax.clear()
            ax.grid(False)
            ax.contour(
                w1grid, w2grid, pdf, cmap='copper', levels=lev_cont)
            ax.pcolormesh(
                w1grid, w2grid, pdf, cmap='copper', alpha=0.5)
            for i, dum in enumerate(wr.T, 1):
                ax.plot(*dum, 'o', color=f'C{i:d}')
            ax.set_xlabel(r'$\omega_1$ (intercept)')
            ax.set_ylabel(r'$\omega_2$ (angular coeff)')
            sigm = max(*np.diag(dist.cov_object.covariance), 0.5)
            ax.set_xlim(lims1)
            ax.set_ylim(lims2)
            
        muf = phix @ dist.mean
        stdf = phix @ dist.cov_object.covariance @ phix.T
        stdf = np.sqrt(np.diag(stdf))
        
        ay.clear()
        ay.set_xlabel('x')
        ay.set_ylabel(r'$y = f(x) + \varepsilon$')
        
        ay.plot(x, muf, label=r'$\mathbb{E}(f(x))$')
        ay.fill_between(
            x, muf+1.96*stdf, muf-1.96*stdf, color='C0', alpha=0.2,
            label='95% confidence')
        lines = [
            ay.plot(
                x, xd, '--', color=f'C{i:d}',lw=1)[0]
            for i, xd in enumerate((phix @ wr).T, 1)]
        lines[0].set_label('samples')

        ay.legend(
            loc='lower center', bbox_to_anchor=(0.5, 0), fontsize='small')
        fig.tight_layout()
        return []
    
#     return animate(0)
    return FuncAnimation(
        fig, animate, frames=frames,
        repeat=True, repeat_delay=20, interval=20)


def draw_samples_from_gp(x_samples, kernel_func=None, nsamples=10):
    if kernel_func is None:
        kernel_func = kernel_squared_exponential
    cov = kernel_func(x_samples, x_samples)
    mean = np.zeros(cov.shape[0])
    return np.random.multivariate_normal(mean, cov, size=nsamples)


def non_parametric_regression(x, x_data, y_data, kernel_func=None):
    pass
    
    
    