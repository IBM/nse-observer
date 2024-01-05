from oasis.problems.NSfracStep import *
try:
    from matplotlib import pyplot as plt
except:
    pass

import numpy as np
import bz2
import _pickle as cPickle
import h5py
from scipy.io import savemat, loadmat
import scipy.interpolate as spi
from dolfin import MPI
from generate2d_py import RandomField
from mpi4py.MPI import DOUBLE
import os

COMM = MPI.comm_world
MPI_SIZE = MPI.size(COMM)
MPI_RANK = MPI.rank(COMM)
SEED = seedflag
GRID = gridflag
NU = nuflag
WORK_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir, os.pardir))
FOLDER = WORK_DIR + '/data/r{0}_g{1}by{1}_nu{2}'.format(SEED, GRID, NU)
# kappa_peak determines the peak in the spectral energy
# these need to be chosen based on a grid size;
# in the paper, we use KAPPA_PEAK=16, RF_Q=100
KAPPA_PEAK = 2
RF_Q = 1e-16

rb_x = 2*np.pi
rb_y = 2*np.pi

# Override some problem specific parameters


def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        cPickle.dump(data, f)


def save_hdf5(title, dictionary):
    hf = h5py.File(title + '.h5', 'w')
    dict_group = hf.create_group('dict_data')
    for k, v in dictionary.items():
        dict_group[k] = v
    hf.close()


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
        Nrec_x=64,
        Nrec_y=64,
        folder=FOLDER,
        init_cond_file=None,
        random_init_cond=True,
        max_iter=50,
        max_error=1e-12,
        plot_interval=100000,
        save_step=10,
        checkpoint=10,
        print_intermediate_info=1000,
        compute_error=50,
        use_krylov_solvers=True,
        krylov_report=False,
        # solver='Chorin')
        # solver='IPCS_ABCN')
        # solver='IPCS')
        solver='BDFPC_Fast')

    NS_parameters['krylov_solvers'] = {'monitor_convergence': False,
                                       'report': False,
                                       'relative_tolerance': 1e-12,
                                       'absolute_tolerance': 1e-12}
    if MPI_RANK == 0:
        random_number = np.random.rand()
    else:
        random_number = None
    random_number = COMM.bcast(random_number, root=0)
    NS_expressions.update(dict(
        constrained_domain=PeriodicDomain(),
        initial_fields=dict(
            u0='exp(-2*pow(x[0]-pi, 2) - 4*pow(x[1]-pi+1,2))*(-8)*(x[1]-pi+1)*10*{0}*sin(x[0])'.format(random_number),
            u1='exp(-2*pow(x[0]-pi, 2)*{0} - 4*pow(x[1]-pi+1,2))*4*(x[0]-pi)*10*{0}*cos(x[1])'.format(random_number),
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


def initialize(q_, q_1, q_2, VV, t, nu, dt, initial_fields, **NS_namespace):
    """Initialize solution.

    Use t=dt/2 for pressure since pressure is computed in between timesteps.

    """
    # initialise observations matrices
    dofs = VV['u0'].tabulate_dof_coordinates()
    mapper = PeriodicDomain()
    for i in range(dofs.shape[0]):
        mapper.map(dofs[i, :], dofs[i, :])

    if NS_namespace['init_cond_file'] is not None:
        f2m_ind = np.lexsort([dofs[:, 1], dofs[:, 0]])
        m2f_ind = f2m_ind.argsort()

        grid_x = np.unique(dofs[:, 0])
        grid_y = np.unique(dofs[:, 1])

        data = loadmat(NS_namespace['init_cond_file'])
        x = np.arange(0, rb_x, rb_x/(data['vort'].shape[1]+1))[:-1]
        y = np.arange(0, rb_y, rb_y/(data['vort'].shape[0]+1))[:-1]
        u0 = {}
        f = spi.interp2d(x, y, data['vel_x'], kind='cubic')
        u0['u0'] = f(grid_x, grid_y).ravel(order='F')[m2f_ind]
        f = spi.interp2d(x, y, data['vel_y'], kind='cubic')
        u0['u1'] = f(grid_x, grid_y).ravel(order='F')[m2f_ind]

        for ui in q_:
            if ui != 'p':
                q_[ui].vector()[:] = u0[ui]
                q_1[ui].vector()[:] = u0[ui]
                q_2[ui].vector()[:] = u0[ui]

        q_['p'].vector()[:] = np.zeros(q_['p'].vector()[:].shape)
        q_1['p'].vector()[:] = q_['p'].vector()[:]
    elif NS_namespace['random_init_cond']:
        print(MPI_RANK, np.shape(dofs))
        # get sizes
        dofs_sizes = COMM.gather(int(np.size(dofs)))

        if MPI_RANK == 0:
            dofs_sizes = np.array(dofs_sizes)
            all_dofs = np.empty(sum(dofs_sizes))
            count = [int(size/2) for size in dofs_sizes]
            count = np.array(count)
            # displacement: the starting index of each sub-task
            displ = [sum(count[:p]) for p in range(MPI_SIZE)]
            displ = np.array(displ)
        else:
            count = np.zeros(MPI_SIZE, dtype=np.int)
            all_dofs = None
            displ = None

        COMM.Gatherv(dofs, (all_dofs, dofs_sizes), root=0)

        if MPI_RANK == 0:
            N = NS_namespace['Nx']*NS_namespace['velocity_degree']
            field = RandomField(
                N_cells=N,
                N_modes=int(N/2), kappa_peak=KAPPA_PEAK, Q=RF_Q)
            all_dofs_reshaped = np.reshape(
                all_dofs, [int(np.sum(dofs_sizes)/2), 2])
            f2p_ind = np.lexsort([
                all_dofs_reshaped[:, 1], all_dofs_reshaped[:, 0]])
            p2f_ind = f2p_ind.argsort()

            all_vel_x = field.u.ravel(order='F')[p2f_ind]
            all_vel_y = field.v.ravel(order='F')[p2f_ind]

        else:
            all_vel_x = None
            all_vel_y = None
            all_dofs = None

        COMM.Bcast(count)
        # initialize random fields on all processes
        vel_x = np.zeros(count[MPI_RANK])
        vel_y = np.zeros(count[MPI_RANK])

        COMM.Scatterv(
            [np.ascontiguousarray(all_vel_x), count, displ,
                DOUBLE], vel_x, root=0)
        COMM.Scatterv(
            [np.ascontiguousarray(all_vel_y), count, displ,
                DOUBLE], vel_y, root=0)

        u0 = {}
        u0['u0'] = vel_x
        u0['u1'] = vel_y
        for ui in q_:
            if ui != 'p':
                q_[ui].vector()[:] = u0[ui]
                q_1[ui].vector()[:] = u0[ui]
                q_2[ui].vector()[:] = u0[ui]

        q_['p'].vector()[:] = np.zeros(q_['p'].vector()[:].shape)
        q_1['p'].vector()[:] = q_['p'].vector()[:]
    else:
        for ui in q_:
            if 'IPCS' in NS_parameters['solver']:
                deltat = dt / 2. if ui=='p' else 0.
            else:
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

    flow.append([q_['u0'].copy(deepcopy=True), q_['u1'].copy(deepcopy=True)])
    preassure.append(q_['p'].copy(deepcopy=True))


def body_force(VV, len_y, f_mode, f_amp, **NS_namespace):
    # Kolmogorov forcing (fourier modes)
    fourier_coef = '-{1}/(2*pi*{2})*{3}*sin(2*pi/{1}*{2}*x[{0}])'
    # this one is actually not needed
    # fourier_x    = fourier_coef.format(0, params['len_x'], params['f_mode'], -params['f_mode']*params['f_amp']/2)
    # only y direction polynomials are used
    fourier_y = fourier_coef.format(1, len_y, f_mode, -f_mode*f_amp)

    return Expression((f'{fourier_y}', f'{fourier_y}'), degree=2)

t = 0


def temporal_hook(q_, t, nu, VV, dt, plot_interval, initial_fields, tstep, sys_comp,
                  compute_error, total_error, **NS_namespace):
    """Function called at end of timestep.

    Plot solution and compute error by comparing to analytical solution.
    Remember pressure is computed in between timesteps.

    """
    flow.append([q_['u0'].copy(deepcopy=True), q_['u1'].copy(deepcopy=True)])
    preassure.append(q_['p'].copy(deepcopy=True))

    if tstep % compute_error == 0:
        if MPI_RANK == 0:
            print("at time = ", t)


def vector_to_func(vec, V):
    ff = Function(V)
    ff.vector()[:] = vec
    return ff


def theend_hook(mesh, q_, t, nu, VV, sys_comp, total_error, initial_fields, **NS_namespace):
    # flow as a vector function
    vec_flow = [None] * len(flow)
    W = VectorFunctionSpace(
        mesh, 'CG', dim=2, degree=NS_namespace['velocity_degree'])
    for i in range(len(flow)):
        vec_flow_n = Function(W)
        assign(vec_flow_n, [flow[i][0], flow[i][1]])
        vec_flow[i] = vec_flow_n

    dofs = flow[0][0].function_space().tabulate_dof_coordinates()
    mapper = PeriodicDomain()
    for i in range(dofs.shape[0]):
        mapper.map(dofs[i, :], dofs[i, :])

    n_node_x = NS_namespace['velocity_degree']*NS_namespace['Nx']
    n_node_y = NS_namespace['velocity_degree']*NS_namespace['Ny']
    # Gather solutions

    # get sizes
    dofs_sizes = np.array(COMM.gather(np.size(dofs)))
    # velocity
    u_regular = np.array([flow_n[0].vector() for flow_n in flow]).T
    v_regular = np.array([flow_n[1].vector() for flow_n in flow]).T
    # get sizes
    u_reg_sizes = np.array(COMM.gather(np.size(u_regular)))
    v_reg_sizes = np.array(COMM.gather(np.size(v_regular)))
    if MPI_RANK == 0:
        all_dofs = np.empty(sum(dofs_sizes))
        all_u_gather = np.empty(sum(u_reg_sizes))
        all_v_gather = np.empty(sum(v_reg_sizes))
    else:
        all_dofs = None
        all_u_gather = None
        all_v_gather = None

    COMM.Gatherv(dofs, (all_dofs, dofs_sizes), root=0)
    COMM.Gatherv(u_regular, (all_u_gather, u_reg_sizes), root=0)
    COMM.Gatherv(v_regular, (all_v_gather, v_reg_sizes), root=0)
    # Write results
    if MPI_RANK == 0:
        all_dofs_reshaped = np.reshape(
            all_dofs, [int(np.sum(dofs_sizes)/2), 2])
        sort_ind = np.lexsort(
            [all_dofs_reshaped[:, 1], all_dofs_reshaped[:, 0]])
        n_tsteps = int(
            NS_parameters['T']/NS_parameters['dt'] + 1)
        pointer_u = 0
        pointer_v = 0
        all_u_gather_reshape = []
        all_v_gather_reshape = []
        for i in range(MPI_SIZE):
            all_u_gather_reshape.append(np.array(
                all_u_gather[pointer_u: pointer_u + u_reg_sizes[i]]
                             ).reshape(n_tsteps,
                                       int(u_reg_sizes[i]/n_tsteps)).T)
            pointer_u += u_reg_sizes[i]

            all_v_gather_reshape.append(np.array(
                all_v_gather[pointer_v: pointer_v + v_reg_sizes[i]]
                             ).reshape(n_tsteps,
                                       int(v_reg_sizes[i]/n_tsteps)).T)
            pointer_v += v_reg_sizes[i]

        u_regular_final = np.array(
            np.concatenate(all_u_gather_reshape)[sort_ind].T).reshape(
                (n_tsteps, n_node_x, n_node_y), order='F')
        v_regular_final = np.array(
            np.concatenate(all_v_gather_reshape)[sort_ind].T).reshape(
                (n_tsteps, n_node_x, n_node_y), order='F')
        savemat(NS_namespace['folder']+'/flow_info.mat', {
                                                         'n_node_x': n_node_x,
                                                         'n_node_y': n_node_x,
                                                         'type': 'oasis'})

        save_hdf5(NS_namespace['folder']+'/flow_reg', {'vel_x_reg':
                                                       u_regular_final,
                                                       'vel_y_reg':
                                                       v_regular_final})
        print('The end')
