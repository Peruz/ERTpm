import os
import argparse
import subprocess
import pygimli as pg
import pygimli.physics.ert as ert
from pygimli.meshtools import readGmsh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_jacobian(inv):
    """
    Jacobian(jaco) is mxn, data(dd) is mx1, and model(mm) is is 1xn

    1. https://github.com/gimli-org/gimli/issues/154
    The original jacobian coming from the forward operator (fop.jacobian())
    is the partial derivative of the modelled apparent resistivity with respect to the subsurface resistivity.
    However, in the inversion we usually use the logarithm of both quantities as data and model, respectively.
    This scales the inverse problem favourably and ensures positive values.
    So the jacobian (d log rhoa/d log rho) needs to be scaled
    by the inner derivative of the logarithms, which is 1/rho and 1/rhoa.
    After cumulating its absolute values over all data (the cumulative sensitivity = coverage),
    it is scaled by the cell size and logarithmized for easier use.

    2. https://gitlab.com/resistivity-net/bert/-/issues/153
    Need for reordering the jacobian.

    3. https://gitlab.com/resistivity-net/bert/-/issues/35
    fop.jacobian() is the normal sensitivity,
    i.e. the derivative of the apparent resistivity w.r.t. the subsurface resistivity.
    The actual jacobian depends on the Transformation used, bounds etc.,
    but is never formed explicitly. for this a special bert function is used.

    It seems that inv.fop.jacobian() already accounts for the values of rhoa and rho,
    it produces (without any correction, i.e., multiply and divide by dd and mm)
    the smooth coverage, without coverage anomalies in regions of different resisitivity.
    While the coverage is expected to depend on the resistivity, for similar values,
    I don't expect anomalies the clearly match the resisitivity,
    looking more the result of the multiplication
    rather than the resitivity control on the coverage.

    For this reason, it may be logical not to multiply and divide the jacobian.
    They still need the size correction though.
    """
    markers = inv.paraDomain.cellMarkers()  # get cell markers, to control their ordering
    sizes = np.array(inv.paraDomain.cellSizes())  # get the cell sizes, already ordered
    jaco = np.array(inv.fop.jacobian())[:, markers]  # get jacobian, and reorder its columns (dlogrhoa/drho)

    # NOTE this is the part where the jacobian may need to be adjusted, but probably it doesn't need
    # because it isn't the derivatives of the logs but of the values themselves.
    # dd = np.array(inv.inv.response())  # data, apparent resistivities (dd as in pygimli coverageDCtrans)
    # mm = np.array(inv.inv.model()[markers])  # model resitivity (mm); it needs order from markers
    # jaco = jaco * dd[:, None]  # from jaco * (1 / dd), every column * (1 / dd), see also Stummer eq. 4
    # jaco = jaco / mm  # from jaco / (1 / mm), every row *  mm

    jaco = jaco / sizes  # every row / sizes
    abs_jaco = np.abs(jaco)  # abs is needed before sum and log10
    sum_abs_jaco = np.sum(abs_jaco, axis=0)  # sum sensitivities (all rows together)
    log_ind_abs_jaco = np.log10(abs_jaco)  # log10 of each sensitivity
    log_sum_abs_jaco = np.log10(sum_abs_jaco)  # log10 of the sum
    return (jaco, log_ind_abs_jaco, log_sum_abs_jaco)


def mesh_add_jacobian(jaco, log_ind_abs_jaco, log_sum_abs_jaco, mesh):
    for i, row in enumerate(jaco):
        n = str(i + 1)
        mesh.addData('jaco' + n, row)
    for i, row in enumerate(log_ind_abs_jaco):
        n = str(i + 1)
        mesh.addData('log10_jaco' + n, row)
    mesh.addData('log10_sum', log_sum_abs_jaco)


def get_cmd():
    """ get command line arguments for data processing """
    parse = argparse.ArgumentParser()
    main = parse.add_argument_group('main')
    parameters = parse.add_argument_group('parameters')
    # MAIN
    main.add_argument('-f', type=str, help='data file to process', nargs='+')
    main.add_argument('-m', type=str, help='mesh file')
    main.add_argument('-o', type=str, help='output inversion directory', default='inversion')
    main.add_argument('-v', action='store_true', help='verbose info and output')

    # PARAMETERS
    parameters.add_argument('-lam', type=float, help='starting lambda', default=20)
    parameters.add_argument('-err', type=float, help='overwrite data error', default=None)
    parameters.add_argument('-chi2_lims', type=float, help='chi2 (min, max)', nargs='+', default=(0.9, 1.2))
    parameters.add_argument('-chi2_opt', action='store_true', help='chi2 optimization', default=True)
    parameters.add_argument('-l1', action='store_true', help='L1 norm spatial regularization')
    # GET ARGS
    args = parse.parse_args()
    return args


def output_file(old_fname, new_ext, directory='out'):
    """ return name for the output file and clean them if already exist """
    head, tail = os.path.split(old_fname)
    full_outdir = os.path.join(head, directory)
    os.makedirs(full_outdir, exist_ok=True)
    stem, ext = os.path.splitext(tail)
    new_tail = stem + new_ext
    new_dfname = os.path.join(full_outdir, new_tail)
    if os.path.exists(new_dfname):
        os.remove(new_dfname)
    return new_dfname


def update_lam(chi2_record, lam_record):
    print("UPDATING LAMBDA")
    chi2_lam = {c: l for c, l in zip(chi2_record, lam_record)}
    last_chi2 = chi2_record[-1]
    last_lam = lam_record[-1]
    if last_chi2 > 1:
        print("chi2 higher than 1, decreasing lambda")
        lowers = [c for c in chi2_record if c < 1]
        if lowers:
            lower_closest = max(lowers)
            lam_opposite = chi2_lam[lower_closest]
        else:
            lam_opposite = 0
        new_lam = (last_lam + lam_opposite) / 2
    elif last_chi2 < 1:
        print("chi2 lower than 1, increasing lambda")
        highers = [c for c in chi2_record if c > 1]
        if highers:
            higher_closest = min(highers)
            lam_opposite = chi2_lam[higher_closest]
        else:
            lam_opposite = last_lam * 3
        new_lam = (last_lam + lam_opposite) / 2
    return new_lam


def save_pareto_csv(lam_record, chi2_record, fout):
    """ save lambda and chi2 values to dataframe """
    pareto = pd.DataFrame({'lambda': lam_record, 'chi2': chi2_record})
    pareto.to_csv(fout, float_format='%6.3f')


def save_pareto_vtk(ertmgr, model_record, lam_record, chi2_record, fout):
    """ save all to vtk """
    mesh = ertmgr.paraDomain
    for m, l, c in zip(model_record, lam_record, chi2_record):
        model_name = '{:.2f}_{:.2f}'.format(l, c)
        mesh.addData(model_name, m)
    mesh.exportVTK(fout)


def save_inv_vtk(ertmgr, fout):
    mesh = ertmgr.paraDomain
    res = ertmgr.paraModel().array()
    cov = ertmgr.coverage()
    mesh.addData('res', res)
    mesh.addData('cov', cov)
    mesh.exportVTK(fout)


def save_misfit_csv(ertmgr, fcsv):
    """ save to csv the misfit measured data and fwd response """
    fwd_response = np.array(ertmgr.inv.response.array())
    measured = np.array(ertmgr.fop.data['rhoa'].array())
    misfit = pd.DataFrame({'fwd': fwd_response, 'measured': measured})
    misfit['misfit'] = misfit['fwd'] - misfit['measured']
    misfit['abs_misfit'] = misfit['misfit'].abs()
    misfit['percent_misfit'] = misfit['abs_misfit'] / misfit['measured'].abs()
    misfit.to_csv(fcsv)


if __name__ == '__main__':

    args = get_cmd()

    if args.m.endswith('geo'):
        meshmsh = output_file(args.m, 'msh', args.o)
        subprocess.call(["gmsh", "-format", "msh2", "-2", "-o", meshmsh, args.m])
        mesh = readGmsh(meshmsh, verbose=args.v)
    elif args.m.endswith('msh'):
        mesh = readGmsh(args.m, verbose=args.v)

    if args.v:
        pg.show(mesh, markers=True)
        plt.show()

    for data_file in args.f:
        data = pg.load(data_file)
        data['k'] = ert.createGeometricFactors(data, numerical=True, mesh=mesh)
        data['rhoa'] = data['r'] * data['k']
        if args.v:
            plt.plot(data['rhoa'], 'o')
            plt.show()
            plt.plot(data['k'], 'o')
            plt.show()
            plt.plot(data['err'], data['k'], 'o')
            plt.show()
        data.markInvalid(data['rhoa'] < 10)
        data.markInvalid(data['rhoa'] > 5000)
        data.markInvalid(abs(data['k']) > 3000)
        data.removeInvalid()

        ertmgr = ert.ERTManager()

        if args.err:
            data['err'] = ertmgr.estimateError(data, absoluteError=0.05, relativeError=args.err)
            plt.plot(data['err'], data['k'], 'o')
            plt.show()

        if args.chi2_opt:
            chi_min = args.chi2_lims[0]
            chi_max = args.chi2_lims[1]
        else:
            chi_min = 0
            chi_max = np.inf

        lam_record = []
        lam = args.lam
        chi2_record = []
        model_record = []
        chi2 = -1
        i_max = 10
        model = None

        while not chi_min < chi2 < chi_max:

            if len(chi2_record) > i_max:
                continue

            if chi2_record:
                lam = update_lam(chi2_record, lam_record)

            _ = ertmgr.invert(
                data=data,
                lam=lam,
                startModel=model,
                verbose=True,
                mesh=mesh,
                robustData=True,
            )

            chi2 = ertmgr.inv.chi2()
            chi2_record.append(chi2)
            model = ertmgr.inv.model
            model_record.append(model)
            lam_record.append(lam)
            print('chi2 record: ', chi2_record)
            print('lambda record: ', lam_record)

        jacmesh = ertmgr.paraDomain
        jaco, log_ind_abs_jaco, log_sum_abs_jaco = get_jacobian(ertmgr)
        mesh_add_jacobian(jaco, log_ind_abs_jaco, log_sum_abs_jaco, jacmesh)
        out_jac_vtk = output_file(data_file, '_jac.vtk', args.o)
        jacmesh.exportVTK(out_jac_vtk)

        out_inv_vtk = output_file(data_file, '_inv.vtk', args.o)
        save_inv_vtk(ertmgr, out_inv_vtk)
        out_misfit_csv = output_file(data_file, '_misfit.csv', args.o)
        save_misfit_csv(ertmgr, out_misfit_csv)
        out_pareto_vtk = output_file(data_file, '_pareto.vtk', args.o)
        save_pareto_vtk(ertmgr, model_record, lam_record, chi2_record, out_pareto_vtk)
        out_pareto_csv = output_file(data_file, '_pareto.csv', args.o)
        save_pareto_csv(lam_record, chi2_record, out_pareto_csv)
