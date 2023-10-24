import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.io import loadmat
from numpy import (
    linspace, meshgrid, pi, zeros_like,
    cdouble, sqrt, exp)
from numpy.fft import fftfreq, ifftn
from numpy.random import rand, seed


def zero_mean_normal_noise(X, std):
    noise = std*np.random.randn(X.shape[0], X.shape[1])
    noise = noise - np.mean(noise)
    X_noisy = X + noise
    return X_noisy


def zero_mean_uniform_noise(X, std):
    noise = np.random.uniform(-std, std, (X.shape[0], X.shape[1]))
    noise = noise - np.mean(noise)
    X_noisy = X + noise
    return X_noisy


def compute_rms(
    random_seeds=np.array([123, 124]),
    folder_location='../../data/r{0}_g64by64_nu0.006',
        dt=0.01, time_range=[10, 15], solver='jax-cfd'):
    vel_x = []
    vel_y = []
    print('Collecting data from {0}'.format(folder_location))
    for random_seed in random_seeds:
        if solver == 'fenics':
            folder = (folder_location+'/flow_reg.mat').format(random_seed)
            data_temp = loadmat(
                folder)
            vel_x.append(data_temp['vel_x_reg'][
                int(time_range[0]/dt):int(time_range[1]/dt)])
            vel_y.append(data_temp['vel_y_reg'][
                int(time_range[0]/dt):int(time_range[1]/dt)])
        elif solver == 'jax-cfd':
            folder = (
                folder_location+'/results_fine.npz').format(random_seed)
            data_temp = np.load(folder)
            vel_x.append(data_temp['u'][
                int(time_range[0]/dt):int(time_range[1]/dt)])
            vel_y.append(data_temp['v'][
                int(time_range[0]/dt):int(time_range[1]/dt)])

    rms = np.sqrt(np.mean(0.5*(np.array(vel_x)**2 + np.array(vel_y)**2)))
    return rms


def plot_norms(
        series, titles, figsize=(24, 10), display=False, save_file=None):
    plt.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')

    for i in range(min(4, len(series))):
        plt.subplot(2, 2, i+1)
        plt.semilogy(series[i])
        plt.title(titles[i])
        plt.xlabel('time')

    if save_file is not None:
        plt.savefig(save_file, dpi=300)

    if display:
        plt.show()


def plot_stats(
    u_mean, v_mean,
    u_std, v_std,
    figsize=(12, 10),
        display=False, save_file=None):
    plt.figure(num=None, figsize=figsize, dpi=80, facecolor='w', edgecolor='k')
    plt.subplot(2, 1, 1)
    plt.ylabel('mean value')
    plt.xlabel('time')
    plt.plot(u_mean, label='mean of x-component')
    plt.plot(v_mean, label='y-component')
    plt.legend()
    plt.title('mean of u and v components')

    plt.subplot(2, 1, 2)
    plt.ylabel('std value')
    plt.xlabel('time')
    plt.plot(u_std)
    plt.plot(v_std)
    plt.title('std of u and v components')

    if save_file is not None:
        plt.savefig(save_file, dpi=300)

    if display:
        plt.show()


def compute_flow_stats(flow):
    num_steps = len(flow)
    u_mean = np.zeros((num_steps,))
    v_mean = np.zeros((num_steps,))
    u_std = np.zeros((num_steps,))
    v_std = np.zeros((num_steps,))

    flow_iterator = tqdm(range(num_steps))
    flow_iterator.set_description('computing flow stats')
    for n in flow_iterator:
        u, v = flow[n]
        u_mean[n] = np.mean(u)
        v_mean[n] = np.mean(v)

        u_std[n] = np.std(u)
        v_std[n] = np.std(v)

    return u_mean, v_mean, u_std, v_std


class RandomField:
    ''' Script implementing the spectral method of Masanobu Shinozuka, where we
    represent the stochastic field as a sum of trigonometric functions with
    random phase angles that are independent, identically distributed random
    variables, uniformly distributed from 0 to 2π.

    Code was written with the help of Robert Sawko (rsawko@uk.ibm.com).

    '''
    def __init__(
        self, l_dom=2*pi, N_cells=64,
            N_modes=32, kappa_peak=8, Q=1e-8, seed=123):
        self.seed = seed
        # Note: For the field to be periodic N_cells=2*N_modes
        x = linspace(0, l_dom, N_cells)
        y = linspace(0, l_dom, N_cells)
        self.x, self.y = meshgrid(x, y, indexing='ij')
        # kappa0 = 2.0*pi / l_dom
        self.N_cells = x.shape[0]
        delta_x = l_dom / N_cells
        self.delta_x = delta_x
        # Nyquist frequency
        self.kappa_max = pi / delta_x
        # Equation 48
        self.delta_kappa = self.kappa_max / N_modes
        kappa_x = 2.0*pi*fftfreq(N_cells, d=delta_x)
        kappa_y = 2.0*pi*fftfreq(N_cells, d=delta_x)
        self.kappa_x, self.kappa_y = meshgrid(
            kappa_x,
            kappa_y,
            indexing='ij')
        self.kappa_peak = kappa_peak
        self.Q = Q
        self.psi, self.u, self.v = self._compute_streamfunction()

    def energy_function(self, kappa):
        return self.Q*kappa**8*exp(-4*(kappa/self.kappa_peak)**2)

    def _compute_streamfunction(self):
        '''this method uses the expression found in lowe for the magnitude of
        the streamfunction fourier transfrom:
            |\\hat psi|^2 = \\frac{e(k)}{\\pi k^3}
        subsequently velocity transforms are computed by using differentiation
        in fourier space but the function itself is simulated with shinozuka
        and deodatis method.
        '''
        # todo: consider reimplementing this with irfftn
        seed(self.seed)
        m1, m2 = self.x.shape
        # stream function transform
        psi_hat = zeros_like(self.x, dtype=cdouble)
        for n1 in range(1, m1//2-1):
            for n2 in range(1, m2):
                self.__single_term(
                    n1, n2, m2,
                    self.delta_kappa,
                    self.energy_function,
                    psi_hat)
        self.psi_hat = psi_hat
        psi = ifftn(psi_hat)
        u_hat = 1j*self.kappa_y*psi_hat
        v_hat = -1j*self.kappa_x*psi_hat
        u = ifftn(u_hat)
        v = ifftn(v_hat)
        if abs(psi.imag).any() > 0:
            raise ValueError('non-zero imaginary values in psi')
        return psi.real, u.real, v.real

    def __single_term(
            self,
            n1, n2, M2,
            delta_kappa,
            E,
            psi_hat):
        '''A single term in the sum for process simulation formula. This
        function avoids copy/paste programming in the main two loops.'''
        kappa_x = self.kappa_x[n1, n2]
        kappa_y = self.kappa_y[n1, n2]
        kappa = sqrt(kappa_x**2 + kappa_y**2)
        # mag_psi_hat_k = M2**2*delta_kappa*sqrt(
        #     E(kappa) / (pi * kappa**3))
        mag_psi_hat_k = sqrt(E(kappa) / (pi * kappa**3))
        theta = 2*pi*rand()
        # theta = 2*pi*xi[n1, n2]
        psi_hat[n1, n2] = mag_psi_hat_k * exp(1j*theta)
        # Conjugate-even order is necessary to real and is imposed
        # manually here:
        psi_hat[-n1, -n2] = mag_psi_hat_k * exp(-1j*theta)
