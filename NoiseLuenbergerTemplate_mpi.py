import os
from oasis.problems.NSfracStep import *
from matplotlib import pyplot as plt

import numpy as np
import numpy.linalg as nla
import utils
from scipy.io import savemat, loadmat
from scipy.sparse import csr_matrix
from dolfin import MPI
from mpi4py.MPI import DOUBLE
import h5py

WORK_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
COMM = MPI.comm_world
MPI_SIZE = MPI.size(COMM)
MPI_RANK = MPI.rank(COMM)
NPROCS = COMM.Get_size()
rb_x = 2*np.pi
rb_y = 2*np.pi
# grid size
GRID = gridflag
# compression ratio
CR = crflag
# signal-to-noise ratio
ALPHA = alphaflag
SEED = seedflag
# Luenberger gain
LGain = gainflag
# viscosity
NU = nuflag
# folder with data
FOLDER = WORK_DIR + '/data/r{0}_g{1}by{1}_nu{2}'.format(SEED, GRID, NU)
# Noise RMS
RMS = utils.compute_rms(
    random_seeds=np.array([123]),
    folder_location=FOLDER,
    dt=0.01, time_range=[10, 15], solver='jax-cfd')
print('RMS:', RMS)
# Override some problem specific parameters


def read_hdf5(title='flow_reg'):
    dict_new = {}
    file = h5py.File(title + '.h5', 'r')
    dict_group_load = file['dict_data']
    dict_group_keys = dict_group_load.keys()
    for k in dict_group_keys:
        dict_new[k] = dict_group_load[k]
    return dict_new


def problem_parameters(NS_parameters, NS_expressions, **NS_namespace):
    NS_parameters.update(
        lb_x=0,
        lb_y=0,
        rb_x=rb_x,
        rb_y=rb_y,
        len_x=rb_x,
        len_y=rb_y,
        nu=NU,
        T=15,
        dt=0.01,
        Nx=int(GRID/2), Ny=int(GRID/2),  # number of elements
        velocity_degree=2,
        pressure_degree=2,
        f_mode=10,
        f_amp=1,
        Nrec_x=int(GRID/CR),
        Nrec_y=int(GRID/CR),
        compr_ratio=int(CR),
        L=LGain,
        folder='./results_mpi',
        obs_file=FOLDER + '/results_fine.npz',
        init_cond_file=None,
        std=RMS*ALPHA,
        obs_type='jax-cfd',
        # obs_type='fenics',
        max_iter=50,
        max_error=1e-12,
        plot_interval=100000,
        save_step=10000,
        checkpoint=10000,
        print_intermediate_info=500,
        compute_error=50,
        use_krylov_solvers=True,
        krylov_report=False,
        solver='BDFPC_Fast')

    NS_parameters['krylov_solvers'] = {'monitor_convergence': False,
                                       'report': False,
                                       'relative_tolerance': 1e-12,
                                       'absolute_tolerance': 1e-12}
    NS_expressions.update(dict(
        constrained_domain=PeriodicDomain(),
        initial_fields=dict(
            u0='0',
            u1='0',
            p='0'),
        total_error=zeros(3)))


def mesh(Nx, Ny, **params):
    return RectangleMesh(Point(0, 0), Point(params['len_x'], params['len_y']), Nx, Ny)


class PeriodicDomain(SubDomain):

    def inside(self, x, on_boundary):
        # return True if on left or bottom boundary AND NOT on one of the two corners (0, 2) and (2, 0)
        return bool((near(x[0], 0) or near(x[1], 0)) and
                    (not ((near(x[0], 0) and near(x[1], rb_y)) or
                          (near(x[0], rb_x) and near(x[1], 0)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], rb_x) and near(x[1], rb_y):
            y[0] = x[0] - rb_x
            y[1] = x[1] - rb_y
        elif near(x[0], rb_x):
            y[0] = x[0] - rb_x
            y[1] = x[1]
        elif near(x[1], rb_y):
            y[0] = x[0]
            y[1] = x[1] - rb_y


flow = []
preassure = []
vel_x_h1_err = []
vel_y_h1_err = []
flow_h1_err = []
source_data = {}
norms_mat = {}
observer = {}
observer_mpi = {}
vel_x_vec_err = []
vel_y_vec_err = []
flow_vec_err = []
flow_vec_norm = []


def initialize(q_, q_1, q_2, VV, t, nu, dt, initial_fields, **NS_namespace):
    """Initialize solution.

    Use t=dt/2 for pressure since pressure is computed in between timesteps.

    """

    for ui in q_:
        deltat = 0.
        vv = interpolate(Expression((initial_fields[ui]),
                                     element=VV[ui].ufl_element(),
                                     t=t + deltat, nu=nu), VV[ui])
        q_[ui].vector()[:] = vv.vector()[:]
        if not ui == 'p':
            q_1[ui].vector()[:] = vv.vector()[:]
            deltat = -dt
            vv = interpolate(Expression((initial_fields[ui]),
                                        element=VV[ui].ufl_element(),
                                        t=t + deltat, nu=nu), VV[ui])
            q_2[ui].vector()[:] = vv.vector()[:]
    q_1['p'].vector()[:] = q_['p'].vector()[:]

    q_u0 = q_['u0'].vector().get_local()
    q_u1 = q_['u1'].vector().get_local()

    all_q_u0_mpi = COMM.gather(q_u0, root=0)
    all_q_u1_mpi = COMM.gather(q_u1, root=0)

    if MPI_RANK == 0:
        all_q_u0 = np.concatenate(all_q_u0_mpi)
        all_q_u1 = np.concatenate(all_q_u1_mpi)
        flow.append([all_q_u0, all_q_u1])

    # flow.append([q_['u0'].copy(deepcopy=True), q_['u1'].copy(deepcopy=True)])
    # preassure.append(q_['p'].copy(deepcopy=True))

    print('computing norms matrices')
    u = NS_namespace['u']
    v = NS_namespace['v']
    norms_mat['l2'] = assemble(u*v*dx)
    norms_mat['h1'] = assemble((u*v + inner(grad(u), grad(v)))*dx)

    # initialise observations matrices
    dofs = VV['u0'].tabulate_dof_coordinates()
    mapper = PeriodicDomain()
    for i in range(dofs.shape[0]):
        mapper.map(dofs[i, :], dofs[i, :])

    # get sizes
    dofs_sizes = np.array(COMM.gather(np.size(dofs)))
    if MPI_RANK == 0:
        all_dofs = np.empty(sum(dofs_sizes))
    else:
        all_dofs = None
    # move the dofs to RANK=0
    COMM.Gatherv(dofs, (all_dofs, dofs_sizes), root=0)

    # initialize the observer on RANK=0
    if MPI_RANK == 0:
        all_dofs_reshaped = np.reshape(
            all_dofs, [int(np.sum(dofs_sizes)/2), 2])
        observer['f2m_ind'] = np.lexsort([all_dofs_reshaped[:, 1],
                                          all_dofs_reshaped[:, 0]])
        observer['m2f_ind'] = observer['f2m_ind'].argsort()
        if NS_namespace['Nrec_x'] != NS_namespace[
                'velocity_degree']*NS_namespace['Nx']:
            proj_mat, recon_mat = spatial_average_proj_mat(
                all_dofs_reshaped, NS_namespace)
            observer['proj_mat'] = proj_mat
            observer['recon_mat'] = recon_mat

        # pdb.set_trace()
        observer['L'] = NS_namespace['L']
        if NS_namespace['obs_type'] == 'fenics':
            data = read_hdf5(NS_namespace['obs_file'] + 'flow_reg')
            observer['vel_x'] = np.zeros(data['vel_x_reg'].shape)
            observer['vel_y'] = np.zeros(data['vel_x_reg'].shape)
            data['vel_x_reg'].read_direct(observer['vel_x'])
            data['vel_y_reg'].read_direct(observer['vel_y'])
            N = int(np.product(data['vel_x_reg'].shape[1:]))
            observer['vel_x'] = observer['vel_x'].reshape((-1, N), order='F').T
            observer['vel_y'] = observer['vel_y'].reshape((-1, N), order='F').T

        elif NS_namespace['obs_type'] == 'jax-cfd':
            data = np.load(NS_namespace['obs_file'])
            data_u = np.transpose(data['u'], (0, 2, 1))
            data_v = np.transpose(data['v'], (0, 2, 1))
            N = int(np.shape(data_u)[1]*np.shape(data_u)[2])
            observer['vel_x'] = data_u.reshape(
                (-1, N), order='F').T[observer['m2f_ind'], :]
            observer['vel_y'] = data_v.reshape(
                (-1, N), order='F').T[observer['m2f_ind'], :]
        else:
            raise ValueError('There is no such optioni for input data')
        # add noise
        observer['vel_x_noisy'] = utils.zero_mean_normal_noise(
            observer['vel_x'], NS_namespace['std'])
        observer['vel_y_noisy'] = utils.zero_mean_normal_noise(
            observer['vel_y'], NS_namespace['std'])
        # project the data onto low-resolution
        if NS_namespace['Nrec_x'] != NS_namespace[
                'velocity_degree']*NS_namespace['Nx']:
            observer['avg_vel_x'] = observer[
                'proj_mat']@observer['vel_x_noisy']
            observer['avg_vel_y'] = observer[
                'proj_mat']@observer['vel_y_noisy']
        else:
            observer['avg_vel_x'] = observer['vel_x_noisy']
            observer['avg_vel_y'] = observer['vel_y_noisy']


def body_force(VV, len_y, f_mode, f_amp, **NS_namespace):
    # set kolmogorov forcing

    # Kolmogorov forcing (fourier modes)
    fourier_coef = '-{1}/(2*pi*{2})*{3}*sin(2*pi/{1}*{2}*x[{0}])'
    fourier_y = fourier_coef.format(1, len_y, f_mode, -f_mode*f_amp)

    f1_klm = Function(VV['u0'])
    f1_klm.assign(Expression(
        f'{fourier_y}', degree=NS_namespace['velocity_degree']))
    source_data['f1_klm'] = f1_klm

    f2_klm = Function(VV['u1'])
    f2_klm.assign(Expression(f'{fourier_y}', degree=NS_namespace['velocity_degree']))
    source_data['f2_klm'] = f2_klm

    f1 = Function(VV['u0'])
    f1.assign(f1_klm)
    f2 = Function(VV['u1'])
    f2.assign(f2_klm)

    source_data['f1'] = f1
    source_data['f2'] = f2

    return (f1, f2)


t = 0


def start_timestep_hook(q_, f, tstep, VV, **NS_namespace):

    q_u0 = q_['u0'].vector().get_local()
    q_u1 = q_['u1'].vector().get_local()

    all_q_u0_mpi = COMM.gather(q_u0, root=0)
    all_q_u1_mpi = COMM.gather(q_u1, root=0)

    if MPI_RANK == 0:
        all_q_u0 = np.concatenate(all_q_u0_mpi)
        all_q_u1 = np.concatenate(all_q_u1_mpi)

        if NS_namespace['Nrec_x'] != NS_namespace[
                'velocity_degree']*NS_namespace['Nx']:
            # project onto a lower resolution grid
            averages_x_err = observer[
                'avg_vel_x'][:, tstep-1] - observer['proj_mat']@all_q_u0
            averages_y_err = observer[
                'avg_vel_y'][:, tstep-1] - observer['proj_mat']@all_q_u1
            all_add_to_source_x = observer[
                'L']*observer['recon_mat']@averages_x_err
            all_add_to_source_y = observer[
                'L']*observer['recon_mat']@averages_y_err
        else:
            averages_x_err = observer[
                'avg_vel_x'][:, tstep-1] - all_q_u0
            averages_y_err = observer[
                'avg_vel_y'][:, tstep-1] - all_q_u1
            all_add_to_source_x = observer[
                'L']*averages_x_err
            all_add_to_source_y = observer[
                'L']*averages_y_err

        # count: the size of each sub-task
        count = [element.size for element in all_q_u0_mpi]
        count = np.array(count, dtype=int)
        # displacement: the starting index of each sub-task
        displ = [sum(count[:p]) for p in range(NPROCS)]
        displ = np.array(displ)
    else:
        all_add_to_source_x = None
        all_add_to_source_y = None
        # initialize count on worker processes
        count = np.zeros(NPROCS, dtype=np.int)
        displ = None

    # broadcast count
    COMM.Bcast(count, root=0)

    # initialize the additional source on all processes
    add_to_source_x = np.zeros(count[MPI_RANK])
    add_to_source_y = np.zeros(count[MPI_RANK])

    COMM.Scatterv([np.ascontiguousarray(all_add_to_source_x), count, displ,
                   DOUBLE], add_to_source_x, root=0)
    COMM.Scatterv([np.ascontiguousarray(all_add_to_source_y), count, displ,
                   DOUBLE], add_to_source_y, root=0)

    source_data['f1'].vector()[:] = source_data[
        'f1_klm'].vector()[:] + add_to_source_x
    source_data['f2'].vector()[:] = source_data[
        'f2_klm'].vector()[:] + add_to_source_y


def vector_norm(f_vec, norm_type='L2'):
    mat = norms_mat['l2']
    if norm_type == 'H1':
        mat = norms_mat['h1']

    tmp = f_vec.copy()
    mat.mult(f_vec, tmp)
    return (tmp[:]@f_vec[:])**0.5


def temporal_hook(
    q_, t, nu, VV, dt, plot_interval, initial_fields, tstep, sys_comp,
        compute_error, total_error, **NS_namespace):
    """Function called at end of timestep.

    Plot solution and compute error by comparing to analytical solution.
    Remember pressure is computed in between timesteps.

    """

    q_u0 = q_['u0'].vector().get_local()
    q_u1 = q_['u1'].vector().get_local()

    all_q_u0_mpi = COMM.gather(q_u0, root=0)
    all_q_u1_mpi = COMM.gather(q_u1, root=0)

    if MPI_RANK == 0:
        all_q_u0 = np.concatenate(all_q_u0_mpi)
        all_q_u1 = np.concatenate(all_q_u1_mpi)
        flow.append([all_q_u0, all_q_u1])

        u_est = flow[tstep-1][0]
        v_est = flow[tstep-1][1]
        u_true = observer['vel_x'][:, tstep-1].copy()
        v_true = observer['vel_y'][:, tstep-1].copy()

        vel_x_vec_err.append(nla.norm(
            u_est-u_true)/nla.norm(u_true))
        vel_y_vec_err.append(nla.norm(
            v_est-v_true)/nla.norm(v_true))
        flow_vec_err.append((nla.norm(
            v_true-v_est)**2+nla.norm(
                v_true-v_est)**2)**0.5 / (
                    nla.norm(u_true)**2+nla.norm(
                        v_true)**2)**0.5)
        flow_vec_norm.append(
            (nla.norm(
                u_est)**2+nla.norm(v_est)**2)**0.5)

        if tstep % compute_error == 0:
            print('time = ', t)
            print('flow error : ', flow_vec_err[tstep-1])
            print('vel_x error: ', vel_x_vec_err[tstep-1])
            print('vel_y error: ', vel_y_vec_err[tstep-1])


def spatial_average_proj_mat(dofs, params):
    # dofs = V.tabulate_dof_coordinates()

    rec_dx = params['len_x'] / params['Nrec_x']
    rec_dy = params['len_y'] / params['Nrec_y']
    recon_mat = np.zeros(
        (dofs.shape[0], params['Nrec_x']*params['Nrec_y']), dtype=int)
    proj_mat = np.zeros((params['Nrec_x']*params['Nrec_y'], dofs.shape[0]))
    print('++ 1')
    eps = DOLFIN_EPS*1000
    for i in range(params['Nrec_x']):
        for j in range(params['Nrec_y']):
            rec_ind = i*params['Nrec_y'] + j
            rec_nodes = (
                dofs[:, 0] <= 0.99999*((i+1)*rec_dx) + eps) & (
                    dofs[:, 0] >= 0.99999*(i*rec_dx) - eps) & (
                        dofs[:, 1] <= 0.99999*((j+1)*rec_dy) + eps) & (
                            dofs[:, 1] >= 0.99999*(j*rec_dy) - eps)
            recon_mat[:, rec_ind] = rec_nodes
    print('++ 2')
    recon_mat = csr_matrix(recon_mat)
    print('++ 3')
    for i in range(params['Nrec_x']):
        for j in range(params['Nrec_y']):
            rec_ind = i*params['Nrec_y'] + j
            rec_nodes = (
                dofs[:, 0] <= 0.99999*((i+1)*rec_dx) + eps) & (
                    dofs[:, 0] >= 0.99999*(i*rec_dx) - eps) & (
                        dofs[:, 1] <= 0.99999*((j+1)*rec_dy) + eps) & (
                            dofs[:, 1] >= 0.99999*(j*rec_dy) - eps)
            proj_mat[rec_ind, :] = rec_nodes/rec_nodes.sum()
    proj_mat = csr_matrix(proj_mat)
    print('++ 4')
    return proj_mat, recon_mat


def vector_to_func(vec, V):
    ff = Function(V)
    ff.vector()[:] = vec
    return ff


def theend_hook(
    mesh, q_, t, nu, VV,
    sys_comp, total_error,
        initial_fields, **NS_namespace):

    if MPI_RANK == 0:
        log_dir = NS_namespace['folder']
        if log_dir is not None:
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            vec_norms_file = os.path.join(log_dir, 'solution_vec_norms.png')
            norms_file = os.path.join(log_dir, 'solution_norms.png')
            stats_file = os.path.join(log_dir, 'solution_stats.png')

            series = [
                flow_vec_norm, flow_vec_err,
                vel_x_vec_err, vel_y_vec_err]
            titles = [
                'flow vec norm', 'flow vec err',
                'velocity x vec err', 'velocity y vec err']
            utils.plot_norms(
                series, titles,
                display=False, save_file=vec_norms_file)

            u_mean, v_mean, u_std, v_std = utils.compute_flow_stats(flow)
            utils.plot_stats(
                u_mean, v_mean, u_std, v_std, save_file=stats_file)

        savemat(os.path.splitext(norms_file)[0]+'.mat',
                {
                    'vel_x_vec_err': vel_x_vec_err,
                    'vel_y_vec_err': vel_y_vec_err,
                    'flow_vec_err': flow_vec_err,
                    'flow_vec_norm': flow_vec_norm
                })
        print('The end.')
