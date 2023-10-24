import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
import xarray

import jax_cfd.base as cfd
import jax_cfd.base.grids as grids
import jax_cfd.spectral as spectral
import numpy as np
import sys


def KolmogorovFlow2D(viscosity, grid, smooth):
    '''
    Sets up the flow that matches MATLAB configuration
    '''
    forcing_mode = 10  # wave_number
    forcing_amplitude = 1.0
    offsets = ((0, 0), (0, 0))
    forcing_fn = lambda grid: cfd.forcings.kolmogorov_forcing(
        grid, k=forcing_mode, scale=forcing_amplitude,
        offsets=offsets)
    return spectral.equations.NavierStokes2D(
        viscosity,
        grid,
        drag=0.0,
        smooth=smooth,
        forcing_fn=forcing_fn)


def simulation_run(
            viscosity, max_velocity,
            domain, grid, grid_coarse,
            dt, final_time,
            outer_steps, random_number,
            plot_results=True,
            check_div=False,
            folder_name='random124_256by256'):
    inner_steps = int(int(final_time / dt) / outer_steps)

    # **use predefined settings for Kolmogorov flow**
    step_fn = spectral.time_stepping.crank_nicolson_rk4(
        KolmogorovFlow2D(
            viscosity, grid, smooth=smooth), dt)

    trajectory_fn = cfd.funcutils.trajectory(
        cfd.funcutils.repeated(step_fn, inner_steps), outer_steps)

    # create an initial velocity field and compute the fft of the vorticity.
    # the spectral code assumes an fft'd vorticity for an initial state
    v0 = cfd.initial_conditions.filtered_velocity_field(
        jax.random.PRNGKey(random_number),
        grid, max_velocity, peak_wavenumber=16)
    vorticity0 = cfd.finite_differences.curl_2d(v0).data
    vorticity_hat0 = jnp.fft.rfftn(vorticity0)

    _, trajectory = trajectory_fn(vorticity_hat0)

    # check the divergence of the initial field
    jnp.mean(cfd.finite_differences.divergence(v0).data)

    if grid_coarse is not None:
        # downsample the trajectory in Fourier
        trajectory_coarse = jax.vmap(
            cfd.resize.downsample_spectral,
            in_axes=(None, None, 0))(None, grid_coarse, trajectory)

    # extract velocities in Fourier;
    # solve for the stream function and then uses the stream function
    # to compute the velocity

    velocity_solve = spectral.utils.vorticity_to_velocity(grid)
    uhat, vhat = velocity_solve(trajectory)

    if grid_coarse is not None:
        # downsample the velocity in Fourier
        uhat_coarse = jax.vmap(
            cfd.resize.downsample_spectral,
            in_axes=(None, None, 0))(None, grid_coarse, uhat)

        vhat_coarse = jax.vmap(
            cfd.resize.downsample_spectral,
            in_axes=(None, None, 0))(None, grid_coarse, vhat)

        spatial_coord_coarse = jnp.arange(
                grid_coarse.shape[0]
                )*2*jnp.pi / grid_coarse.shape[0]  # same for x and y
        coords_coarse = {
          'time': dt*jnp.arange(outer_steps)*inner_steps,
          'x': spatial_coord_coarse,
          'y': spatial_coord_coarse,
        }
        u_coarse = xarray.DataArray(
            jnp.fft.irfftn(uhat_coarse, axes=(1, 2)),
            dims=['time', 'x', 'y'], coords=coords_coarse)
        v_coarse = xarray.DataArray(
            jnp.fft.irfftn(vhat_coarse, axes=(1, 2)),
            dims=['time', 'x', 'y'], coords=coords_coarse)
        np.savez_compressed(
            folder_name + '/results_coarse.npz',
            u_coarse=u_coarse.to_numpy(), v_coarse=v_coarse.to_numpy())

    spatial_coord = jnp.arange(
            grid.shape[0])*2*jnp.pi / grid.shape[0]  # same for x and y
    coords = {
      'time': dt*jnp.arange(outer_steps)*inner_steps,
      'x': spatial_coord,
      'y': spatial_coord,
    }

    u = xarray.DataArray(
        jnp.fft.irfftn(uhat, axes=(1, 2)),
        dims=['time', 'x', 'y'], coords=coords)
    v = xarray.DataArray(
        jnp.fft.irfftn(vhat, axes=(1, 2)),
        dims=['time', 'x', 'y'], coords=coords)
    # save the data
    np.savez_compressed(
        folder_name + '/results_fine.npz', u=u.to_numpy(), v=v.to_numpy())

    if check_div:
        velocities = jax.vmap(lambda v_init, uu, vv: tuple(
            grids.GridVariable(grids.GridArray(vel, offset, grid), bc)
            for vel, offset, bc in zip(
                [
                    uu, vv
                ],
                [v_init[0].offset, v_init[1].offset],
                [v_init[0].bc, v_init[1].bc])), in_axes=(None, 0, 0)
                  )(v0, u.data, v.data)
        divergence_values = jax.vmap(
                lambda vel: jnp.mean(
                    cfd.finite_differences.divergence(vel).data))(velocities)

        print('Average divergence for fine vel:', jnp.mean(divergence_values))

        velocities = jax.vmap(lambda v_init, uu, vv: tuple(
            grids.GridVariable(grids.GridArray(vel, offset, grid), bc)
            for vel, offset, bc in zip(
                [
                    uu, vv
                ],
                [v_init[0].offset, v_init[1].offset],
                [v_init[0].bc, v_init[1].bc])), in_axes=(None, 0, 0)
                  )(v0, u_coarse.data, v_coarse.data)
        divergence_values = jax.vmap(
                lambda vel: jnp.mean(
                    cfd.finite_differences.divergence(vel).data))(velocities)

        print('Average divergence for coarse vel:', jnp.mean(
            divergence_values))

    if plot_results:
        # transform the velocity into real-space
        # and wrap in xarray for plotting
        plotting_interval = outer_steps//10

        time_steps = (
            dt*jnp.arange(outer_steps)*inner_steps)[0:-1:plotting_interval]

        coords = {
          'time': time_steps,
          'x': spatial_coord,
          'y': spatial_coord,
        }

        xarray.DataArray(
            u.data[0:-1:plotting_interval],
            dims=['time', 'x', 'y'], coords=coords).plot.contourf(
                col='time', col_wrap=5, cmap=sns.cm.icefire,
                robust=True, levels=100)
        plt.savefig(folder_name + '/velocity_x_fine', dpi=400)

        xarray.DataArray(
            v.data[0:-1:plotting_interval],
            dims=['time', 'x', 'y'], coords=coords).plot.contourf(
                col='time', col_wrap=5, cmap=sns.cm.icefire,
                robust=True, levels=100)
        plt.savefig(folder_name + '/velocity_y_fine', dpi=400)

        xarray.DataArray(
            jnp.fft.irfftn(trajectory, axes=(1, 2))[0:-1:plotting_interval],
            dims=['time', 'x', 'y'], coords=coords).plot.contourf(
                col='time', col_wrap=5, cmap=sns.cm.icefire,
                robust=True, levels=100)
        plt.savefig(folder_name + '/vorticity_fine', dpi=400)

        if grid_coarse is not None:
            coords_coarse = {
              'time': time_steps,
              'x': spatial_coord_coarse,
              'y': spatial_coord_coarse,
            }

            xarray.DataArray(
                u_coarse[0:-1:plotting_interval],
                dims=['time', 'x', 'y'], coords=coords_coarse).plot.contourf(
                    col='time', col_wrap=5, cmap=sns.cm.icefire,
                    robust=True, levels=100)
            plt.savefig(folder_name + '/velocity_x_coarse', dpi=400)

            xarray.DataArray(
                jnp.fft.irfftn(
                    trajectory_coarse[0:-1:plotting_interval], axes=(1, 2)),
                dims=['time', 'x', 'y'], coords=coords_coarse).plot.contourf(
                    col='time', col_wrap=5, cmap=sns.cm.icefire,
                    robust=True, levels=100)
            plt.savefig(folder_name + '/vorticity_coarse', dpi=400)


if __name__ == '__main__':
    # input parameters
    random_number = int(sys.argv[1])  # set random number for intial field
    grid_size = int(sys.argv[2])  # mesh = grid_size x grid_size
    viscosity = float(sys.argv[3])
    folder_name = sys.argv[4]  # output folder

    # simulation set-up
    max_velocity = 2
    domain = ((0, 2 * jnp.pi), (0, 2 * jnp.pi))
    grid = grids.Grid((grid_size, grid_size), domain=domain)
    # if projection onto a coarser mesh is needed, set the grid size, e.g.
    # grid_coarse = grids.Grid(((64, 64)), domain=domain)
    # otherwise:
    grid_coarse = None
    dt = 0.001
    # if adaptive time-step, use
    # dt = cfd.equations.stable_time_step(max_velocity, .5, viscosity, grid)
    # set the step function using crank-nicolson runge-kutta order 4
    smooth = True  # use anti-aliasing
    # run the simulation up until final_time but only save outer_steps
    final_time = 15.01
    outer_steps = int(0.1*final_time/dt)
    plot_results = True

    '''Run the simulation with a given set-up'''
    simulation_run(
            viscosity, max_velocity,
            domain, grid, grid_coarse,
            dt, final_time,
            outer_steps, random_number,
            plot_results, folder_name=folder_name)
