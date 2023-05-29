from IPython import embed
import os
import argparse
import subprocess
import pygimli as pg
import pygimli.physics.ert as ert
from pygimli.meshtools import readGmsh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    parameters.add_argument('-chi2_lims', help='chi2 (min, max)', nargs='+', default=(0.9, 1.2))
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

    meshmsh = output_file(args.m, 'msh', args.o)
    subprocess.call(["gmsh", "-format", "msh2", "-2", "-o", meshmsh, args.m])
    mesh = readGmsh(meshmsh, verbose=args.v)

    if args.v:
        pg.show(mesh, markers=True)
        plt.show()

    for data_file in args.f:
        data = pg.load(data_file)
        data['k'] = ert.createGeometricFactors(data, numerical=True)
        data['rhoa'] = data['r'] * data['k']
        if args.v:
            plt.plot(data['rhoa'], 'o')
            plt.show()
            plt.plot(data['k'], 'o')
            plt.show()
            plt.plot(data['err'], data['k'], 'o')
            plt.show()
        data.markInvalid(abs(data['rhoa']) < 10)
        data.markInvalid(abs(data['err']) > 0.1)
        data.markInvalid(abs(data['k']) > 3000)
        data.removeInvalid()
        embed()

        ertmgr = ert.ERTManager()


        if args.err:
            data['err'] = ertmgr.estimateError(data, absoluteError=0.00001, relativeError=args.err)

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
            )

            chi2 = ertmgr.inv.chi2()
            chi2_record.append(chi2)
            model = ertmgr.inv.model
            model_record.append(model)
            lam_record.append(lam)
            print('chi2 record: ', chi2_record)
            print('lambda record: ', lam_record)


        out_inv_vtk = output_file(data_file, '_inv.vtk', args.o)
        save_inv_vtk(ertmgr, out_inv_vtk)
        out_misfit_csv = output_file(data_file, '_misfit.csv', args.o)
        save_misfit_csv(ertmgr, out_misfit_csv)
        out_pareto_vtk = output_file(data_file, '_pareto.vtk', args.o)
        save_pareto_vtk(ertmgr, model_record, lam_record, chi2_record, out_pareto_vtk)
        out_pareto_csv = output_file(data_file, '_pareto.csv', args.o)
        save_pareto_csv(lam_record, chi2_record, out_pareto_csv)
