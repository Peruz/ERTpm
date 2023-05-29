from IPython import embed
import os
import numpy as np
import scipy
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
# from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap

import argparse

from discretize import TreeMesh
from discretize.utils import refine_tree_xyz
# from discretize.utils import mkvc, active_from_xyz

from SimPEG import (
    maps,
    data_misfit,
    regularization,
    optimization,
    inverse_problem,
    inversion,
    directives,
    utils,
)
from SimPEG.electromagnetics.static import resistivity as dc
from SimPEG.utils.io_utils.io_utils_electromagnetics import read_dcip2d_ubc, read_dcip_xyz

try:
    from pymatsolver import Pardiso as Solver
except ImportError:
    from SimPEG import SolverLU as Solver

mpl.rcParams.update({"font.size": 13})


def get_cmd():
    parse = argparse.ArgumentParser()
    main = parse.add_argument_group('main')
    main.add_argument('-f', type=str, help='data file to invert', nargs='+')
    main.add_argument('-topo', type=str, help='topography')
    main.add_argument('-m', default=None, type=str, help='mesh file')
    main.add_argument('-type', default='volt', type=str, help='')
    main.add_argument('-rho0', type=float, help='starting model conductivity', default=0.04)
    main.add_argument('-dir_inv', type=str, help='output directory', default='inversions')
    main.add_argument('-clim', default=None, type=float, help='clim (min, max)', nargs='+')
    args = parse.parse_args()
    return args


def plot_nice(x, y, rho, alpha, clim=(None, None)):
    ext = (min(x), max(x), min(y), max(y))
    cmap = np.ones([256, 4])
    cmap[:, 3] = np.linspace(1, 0, 256)
    cmap = ListedColormap(cmap)
    grid_x, grid_y = np.mgrid[min(x):max(x):2000j, min(y):max(y):200j]
    image = scipy.interpolate.griddata(
        (x, y), rho, (grid_x, grid_y), method='cubic',
    )
    image_alpha = scipy.interpolate.griddata(
        (x, y), alpha, (grid_x, grid_y), method='cubic',
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    f = ax.imshow(image.T, extent=ext, origin='lower', cmap='turbo', clim=clim)
    _ = ax.imshow(image_alpha.T, extent=ext, origin='lower', cmap=cmap, clim=(0, 1))
    ax.set_xlabel('m')
    ax.set_ylabel('m')
    ax.xaxis.labelpad = 1
    clb = plt.colorbar(f, orientation='horizontal', shrink=0.5)
    clb.set_label(r'$\rho \quad ohm \, m$')
    return (fig, ax)


def run_inv(fname, args):

    print('\n', fname)

    # set up the directories, relative to the data file being inverted
    head, tail = os.path.split(fname)
    tail_name, tail_ext = os.path.splitext(tail)
    outdir = os.path.join(head, args.dir_inv)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    # volt data_type is actually normalized for the current, i.e., it is the resistance
    if args.type == 'volt':
        if fname.endswith('.xyz'):
            data = read_dcip_xyz(
                fname,
                data_type="volt",
                data_header='V',
                uncertainties_header='SD',
            )
        elif fname.endswith('.obs'):
            data = read_dcip2d_ubc(fname, "volt", "general")

    if args.type == 'ip':
        data = read_dcip_xyz(
            fname,
            data_type="apparent_chargeability",
            data_header='ip',
            uncertainties_header='SD',
        )

    unique_locations = data.survey.unique_electrode_locations
    print('num data:', len(data.dobs))

    if args.m is None:

        # mesh for 2.5D inversions, cellcentered
        min_spacing = np.min(np.diff(unique_locations[:, 0]))
        xmin = min(unique_locations[:, 0])
        xmax = max(unique_locations[:, 0])
        electrode_length = xmax - xmin

        dh = round(min_spacing / 5, 6)
        dom_width_x = electrode_length * 30
        dom_width_z = electrode_length * 8

        # number of blocks in x and z, must be a power of 2
        nbcx = 2 ** int(np.round(np.log(dom_width_x / dh) / np.log(2.0)))
        nbcz = 2 ** int(np.round(np.log(dom_width_z / dh) / np.log(2.0)))
        # actual model dimensions, given the number of blocks
        lenx = dh * nbcx
        lenz = dh * nbcz
        array_len = electrode_length
        array_half = int(array_len / 2)
        x0 = - ((lenx) / 2 - array_half)
        x0 -= dh / 2  # because the electrodes goes at the center of the top cell face
        hx = dh * np.ones(nbcx)
        hz = dh * np.ones(nbcz)
        mesh = TreeMesh([hx, hz], x0=(x0, -lenz))

        # optional radial refinement around the electrodes
        mesh = refine_tree_xyz(
            mesh,
            unique_locations[:, 0:2],
            octree_levels=[2, 2, 3, 3, 3],
            method="radial",
            finalize=False
        )

        # optional box refinement
        # refine_x = round(electrode_length / 8, 2)
        # refine_z = round(electrode_length / 16, 2)
        # xp, zp = np.meshgrid([- refine_x, electrode_length + refine_x], [- refine_z, 0.0])
        # xyz = np.c_[mkvc(xp), mkvc(zp)]
        # mesh = refine_tree_xyz(
        #     mesh,
        #     xyz,
        #     octree_levels=[2, 2, 2, 4],
        #     method="box",
        #     finalize=False,
        # )
        mesh.finalize()

    # else:
    #     mesh = load_mesh(args.mesh)

    starting_conductivity = np.ones(mesh.nC) * np.log(1 / 200)
    # starting_conductivity = np.ones(mesh.nC) * 1/ 20

    mesh.plot_image(starting_conductivity, grid=True)
    plt.show()

    # Survey
    survey = data.survey
    # finds the active cells, here all cells are active because there is no topography
    actind = utils.surface2ind_topo(mesh, np.c_[mesh.vectorCCx, mesh.vectorCCx * 0.0])

    # make sure electrodes matches the top surface
    survey.drape_electrodes_on_topography(mesh, actind, option="top")
    data.survey = survey


    # Mapping
    mapping = maps.ExpMap(mesh)
    # defines which cells are active, and give fixes values to the non-activate
    active_map = maps.InjectActiveCells(mesh, actind, np.log(args.rho0))
    conductivity_map = active_map * maps.ExpMap()

    # Simulation,
    # make sure the position of the electrodes and cells/nodes
    # matches based on type of simulation
    simulation = dc.simulation_2d.Simulation2DCellCentered(
        mesh,
        survey=survey,
        sigmaMap=mapping,
        solver=Solver,
        storeJ=True,
    )

    # Elec positions and Meshing
    elec_onMesh = survey.unique_electrode_locations[:, 0:2]
    elecPositions_diff = elec_onMesh - unique_locations[:, 0:2]
    if np.max(np.abs(elecPositions_diff)) > 0.02:
        print('on data: ', unique_locations)
        print('on mesh: ', elec_onMesh)
        print('diff: ', elecPositions_diff)
        raise ValueError('elec positions were changed to match the mesh')



    # Data MisFit
    # Here the data misfit is the L2 norm of the weighted residual, normalized by their standard deviation.
    # The variable f during the inversion is the sum of these misfit values
    dmis = data_misfit.L2DataMisfit(data=data, simulation=simulation)


    reg = regularization.WeightedLeastSquares(
        mesh,
        indActive=actind,
        reference_model=starting_conductivity,
        alpha_s=0.001,
        alpha_x=1,
        alpha_y=1,
        alpha_z=1,
        # alpha_xx=0.0005,
        # alpha_yy=0.0005,
        # alpha_zz=0.0005,
    )
    reg.reference_model_in_smooth = True  # Reference model in smoothness term

    # Inversion
    # Define how the optimization problem is solved.
    # Here we will use an Inexact Gauss Newton approach.
    opt = optimization.InexactGaussNewton(maxIter=50)
    # Here we define the inverse problem that is to be solved
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)
    # Apply and update sensitivity weighting as the model updates
    update_sensitivity_weighting = directives.UpdateSensitivityWeights()

    # Defining a starting value for the trade-off parameter (beta) between the data misfit and the regularization.
    # Simpeg uses 1
    starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=0.6)
    # starting_beta = directives.BetaEstimate_ByEig(beta0_ratio=10)

    # Set the rate of reduction in trade-off parameter (beta) each time the the inverse problem is solved.
    # Set the number of Gauss-Newton iterations for each trade-off paramter value.
    # coolingRate: num of iterations per each beta
    # coolingFactor: beta reduction factor after each set of iterations
    # Simpeg uses 3 and 2, respectively.
    # beta_schedule = directives.BetaSchedule(coolingFactor=3, coolingRate=2)
    beta_schedule = directives.BetaSchedule(coolingFactor=2, coolingRate=2)
    # Options for outputting recovered models and predicted data for each beta.
    save_iteration = directives.SaveOutputEveryIteration(save_txt=False)
    # Setting a stopping criteria for the inversion.
    target_misfit = directives.TargetMisfit(chifact=1)
    # Update preconditioner
    update_jacobi = directives.UpdatePreconditioner()

    directives_list = [
        update_sensitivity_weighting,
        starting_beta,
        beta_schedule,
        save_iteration,
        target_misfit,
        update_jacobi,
    ]

    # Here we combine the inverse problem and the set of directives
    dc_inversion = inversion.BaseInversion(inv_prob, directiveList=directives_list)

    # Run inversion
    recovered_conductivity_model = dc_inversion.run(starting_conductivity)

    # resistivity
    recovered_conductivity = conductivity_map * recovered_conductivity_model
    # recovered_conductivity[~actind] = np.NaN
    rho = 1 / recovered_conductivity

    # PLOT

    fig, ax = plt.subplots(figsize=(16, 6))
    f = mesh.plot_image(np.log(rho), ax=ax, grid=False, clim=args.clim, pcolor_opts={'cmap': 'turbo'})
    # zmin = - (xmax - xmin) / 5
    # plt.xlim(xmin, xmax)
    # plt.ylim(zmin, 0)
    ax.set_aspect('equal')
    plt.colorbar(f[0], orientation='horizontal')

    f, old_ext = os.path.splitext(fname)
    fname_out = tail_name + '_tree.png'
    dirfname_out = os.path.join(outdir, fname_out)
    plt.savefig(dirfname_out, dpi=180)
    plt.close()

    # sensitivity
    j = simulation.getJ(dc_inversion.m)  # Jacobian, each row a cell, each column a measurement
    j_abs = np.abs(j)  # abs before summing the contributions of all the measurements
    j_abs_sum_cells = np.sum(j_abs, axis=0)  # sum along the rows (total for each row-cell)
    j_abs_sum_cells /= mesh.cell_volumes  # cell_volumes returns length (1d), area (2d), or volume (3d)
    j_abs_sum_cells_log = np.log(j_abs_sum_cells)  # take log because the sensitivity changes a lot
    # a wenner would have
    # sens_lower = -1.7
    # sens_upper = -1
    # a good grad would have something like this
    sens_lower = -2
    sens_upper = 2
    embed()
    j_abs_sum_cells_log[j_abs_sum_cells_log > sens_upper] = sens_upper
    j_abs_sum_cells_log[j_abs_sum_cells_log < sens_lower] = sens_lower
    sens = j_abs_sum_cells_log
    sens_0to1 = (sens - sens_lower) / (sens_upper - sens_lower)
    j_abs_sum_cells_log[j_abs_sum_cells_log > sens_upper] = sens_upper
    j_abs_sum_cells_log[j_abs_sum_cells_log < sens_lower] = sens_lower
    sens = j_abs_sum_cells_log
    sens_0to1 = (sens - sens_lower) / (sens_upper - sens_lower)

    plot_df = pd.DataFrame(
        data={
            'x': mesh.cell_centers[:, 0],
            'y': mesh.cell_centers[:, 1],
            'rho': 1 / recovered_conductivity,
            'sens': sens,
            'alpha': sens_0to1,
            'j_abs_sum_cells': j_abs_sum_cells,
        }
    )

    plot_df.to_csv('inv_df.csv')

    plot_df = pd.read_csv('inv_df.csv')

    # plot
    zmin = - round(electrode_length / 4, 2)
    int_xpad = round(electrode_length / 10, 2)
    int_zpad = round(electrode_length / 20, 2)
    int_xmin = xmin - int_xpad
    int_xmax = xmax + int_xpad
    int_zmin = round(zmin, 2) - int_zpad
    int_zmax = 0
    mask_x = plot_df['x'].between(int_xmin, int_xmax)
    mask_y = plot_df['y'].between(int_zmin, int_zmax)
    mask = mask_x & mask_y
    plot_df = plot_df[mask]
    f, old_ext = os.path.splitext(fname)
    fname_out = tail_name + '_smooth_tree.pdf'
    dirfname_out = os.path.join(outdir, fname_out)
    fig, ax = plot_nice(
        x=plot_df['x'].to_numpy(),
        y=plot_df['y'].to_numpy(),
        rho=plot_df['rho'].to_numpy(),
        alpha=plot_df['alpha'].to_numpy(),
        clim=args.clim,
    )
    ax.scatter(unique_locations[:, 0], unique_locations[:, 1], color='k', s=5)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(zmin, 0)
    ax.set_aspect('equal')
    plt.tight_layout()
    f, old_ext = os.path.splitext(fname)
    fname_out = tail_name + '_nice.png'
    dirfname_out = os.path.join(outdir, fname_out)
    plt.savefig(dirfname_out, bbox_inches='tight', dpi=180)
    plt.close()

    # misfit analysis
    dpred = dc_inversion.invProb.dpred
    dobs = data.dobs
    std = data.standard_deviation
    chi_squared = np.sum(np.square((dobs - dpred) / std)) / len(dobs)
    print(fname, ' chi_squared: ', chi_squared)
    print('chi_squared: ', chi_squared)
    plt.plot(dobs, dpred, 'o')
    fname_out = tail_name + '_misfit.png'
    dirfname_out = os.path.join(outdir, fname_out)
    plt.savefig(dirfname_out, dpi=80)
    plt.close()
    # save_iteration.plot_misfit_curves()
    # save_iteration.plot_tikhonov_curves()


if __name__ == "__main__":
    args = get_cmd()
    for f in args.f:
        run_inv(f, args)

# from ERTpm.ertds import ERTdataset
# from SimPEG import SolverLU as Solver
# from SimPEG.utils.io_utils.io_utils_electromagnetics import read_dcip2d_ubc
# from SimPEG.utils.io_utils.io_utils_electromagnetics import read_dcip_xyz
# from SimPEG.electromagnetics.static import resistivity as dc
# from SimPEG.data import Data
# from SimPEG import (
#     maps,
#     data_misfit,
#     regularization,
#     optimization,
#     inverse_problem,
#     inversion,
#     directives,
#     utils,
# )
# from discretize.utils import mkvc
# from discretize.utils import refine_tree_xyz
# from discretize import TreeMesh
# from discretize import load_mesh

# mpl.rcParams.update({"font.size": 13})

