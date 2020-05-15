"""
using pygimli:
......................................................................................
MethodManager    -->           | Inversion     ->                     |  RVector
framework/methodManager.py     | frameworks/inversion.py              |  gimli c core
                               | ertmgr.fw
                               | also as ertmgr.inv
                               | ert.fw.model also as ertmgr.model
--------------------------
    ^
    |
--------------------------         -----------------
MeshMesthodManager                |MeshModeling
framework/methodManager.py        |frameworks/modelling.py
                                  |
this class calls                  |this class defines drawModel()
the inversion,                    |called by ergmgr.showModel()
delegating to Inversion,          |and calling pg.viewer.showMesh()
which is inhereted from           |
MethodManager().                  |
get rho from:
m = self.fw.model
p = self.paraModel()
the two are the same, same memory
they are pg.RVector
can convert to numpy
p.array() or np.array(p)
can print
print(p)

--------------------------    ------------------
    ^                            ^
    |                            |
--------------------------    ------------------
ERTManager    -->            | ERTModelling
physics/ert.py               | physics/ert.py
                             | ertmgr.fop

"""

import argparse
import pygimli as pg
import pygimli.physics.ert as ert
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from IPython import embed


def get_cmd():
    """ get command line arguments for data processing """
    parse = argparse.ArgumentParser()
    main = parse.add_argument_group('main')
    parameters = parse.add_argument_group('parameters')
    output = parse.add_argument_group('output')
    # MAIN
    main.add_argument('-fName', type=str, help='data file to process', nargs='+')
    main.add_argument('-mesh', type=str, help='mesh file', default='mesh.bms')
    # PARAMETERS
    parameters.add_argument('-lam', type=float, help='lambda', default=20)
    parameters.add_argument('-keep_lam', action='store_true', help='update lam', default=True)
    parameters.add_argument('-err', type=float, help='data error', default=0.05)
    parameters.add_argument('-chi2_lims', help='chi2 (min, max)', nargs='+', default=[0.8, 1.2])
    parameters.add_argument('-opt', action='store_true', help='chi2 optimization', default=True)
    parameters.add_argument('-ref', type=str, help='reference dataset', default=None)
    # OUTPUT
    output.add_argument('-misfit_dir', type=str, help='misfit directory', default='misfit')
    output.add_argument('-pareto_dir', type=str, help='pareto directory', default='pareto')
    output.add_argument('-inv_dir', type=str, help='inversion directory', default='inversion')
    output.add_argument('-inv_vtk', type=str, help='output vtk extension', default='_inv.vtk')
    output.add_argument('-inv_png', type=str, help='output png extension', default='_inv.png')
    output.add_argument('-pareto_csv', type=str, help='pareto csv extension', default='_pareto.csv')
    output.add_argument('-pareto_vtk', type=str, help='pareto vtk extension', default='_pareto.vtk')
    output.add_argument('-pareto_png', type=str, help='pareto png extension', default='_pareto.png')
    output.add_argument('-misfit_csv', type=str, help='misfit csv extension', default='_misfit.csv')
    output.add_argument('-misfit_png', type=str, help='misfit png extension', default='_misfit.png')
    # GET ARGS
    args = parse.parse_args()
    return(args)


def update_args(cmd_args, dict_args):
    """ update cmd-line args with args from dict """
    args = get_cmd()
    for key, val in dict_args.items():
        if not hasattr(args, key):
            raise AttributeError('unrecognized option: ', key)
        else:
            setattr(args, key, val)
    return(args)


def check_args(args):
    """ check consistency of args """
    if args.ref is not None:
        if not os.path.exists(args.ref):
            raise FileNotFoundError('reference model not found')
    if isinstance(args.fName, str):
        args.fName = [args.fName]
    return(args)


# pareto


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


def plot_pareto(fcsv, fpng):
    pareto = pd.read_csv(fcsv)
    plt.plot(pareto['lambda'], pareto['chi2'], 'or')
    plt.savefig(fpng)
    plt.show()

# vtk


def save_inv_vtk(ertmgr, fout):
    mesh = ertmgr.paraDomain
    res = ertmgr.paraModel().array()
    cov = ertmgr.coverage()
    mesh.addData('res', res)
    mesh.addData('cov', cov)
    mesh.exportVTK(fout)


def plot_inv(fvtk, fpng, cm, cM):
    mesh = pg.load(fvtk)
    embed()
    rho = mesh.data('res')
    cov = mesh.data('cov')
    _ = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax, cbar = pg.show(mesh=mesh, data=rho, coverage=cov, ax=ax, hold=True,
                       cMap='jet', cMin=cm, cMax=cM, colorBar=True,
                       showelectrodes=True, label='resistivity [ohm m]')
    ax.set_xlabel('m')
    ax.set_ylabel('m')
    plt.tight_layout()
    plt.savefig(fpng, dpi=600)
    plt.show()


# misfit


def save_misfit_csv(ertmgr, fcsv):
    """ save to csv the misfit measured data and fwd response """
    fwd_response = np.array(ertmgr.inv.response.array())
    measured = np.array(ertmgr.fop.data['rhoa'].array())
    misfit = pd.DataFrame({'fwd': fwd_response, 'measured': measured})
    misfit['misfit'] = misfit['fwd'] - misfit['measured']
    misfit['abs_misfit'] = misfit['misfit'].abs()
    misfit['percent_misfit'] = misfit['abs_misfit'] / misfit['measured'].abs()
    misfit.to_csv(fcsv)


def plot_misfit(fcsv, fpng):
    misfit = pd.read_csv(fcsv)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.plot(misfit['measured'], 'bo', markersize=7)
    ax1.plot(misfit['fwd'], 'ro', markersize=7)
    ax1.set_xlabel('num')
    ax1.set_ylabel('rho')
    ax2.plot(misfit['measured'], misfit['fwd'], 'og', markersize=7)
    ax2.set_xlabel('measured')
    ax2.set_ylabel('fwd calculated')
    ax2.axis('equal')
    fig.tight_layout()
    plt.savefig(fpng, dpi=600)
    plt.show()


# INVERSION


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
    return(new_lam)


def _invert_(fName, mesh, lam=20, err=0.05, opt=False, chi2_lims=(0.8, 1.2), ref=False, ref_fName=None, i_max=10):
    """ args are passed to ERTManager, run optimize inversion, and saves to para_mesh """
    print('\n\n\n')
    print('fName', fName)
    print('opt: ', opt)
    print('lam: ', lam)
    print('\n')
    data = pg.load(fName)
    ertmgr = ert.ERTManager()
    data['err'] = ertmgr.estimateError(data, absoluteError=0.001, relativeError=err)
    mesh = pg.load(mesh)
    ertmgr.fop.setMesh(mesh)

    print('opt is: ', opt)
    if opt:
        chi_min = chi2_lims[0]
        chi_max = chi2_lims[1]
    else:
        chi_min = 0
        chi_max = np.inf

    print('ref is: ', ref)
    if ref:
        ref_model_vtk = pg.load(ref_fName)
        ref_model = ref_model_vtk.exportData('res')
        smr = True
    else:
        model = np.ones(len(mesh.cells()))
        smr = False

    print('preparing for inversion while loop')
    lam_record = []
    chi2_record = []
    model_record = []
    chi2 = -1

    while not chi_min < chi2 < chi_max:
        if len(chi2_record) > i_max:
            raise ValueError("optimization reached maximum number of runs, i_max: ", i_max)
        if chi2_record:
            lam = update_lam(chi2_record, lam_record)

        if ref:
            _ = ertmgr.invert(data=data, lam=lam, startModel=ref_model, startModelIsReference=smr, verbose=True)
        else:
            _ = ertmgr.invert(data=data, lam=lam, verbose=True)

        chi2 = ertmgr.inv.chi2()
        chi2_record.append(chi2)
        model = ertmgr.inv.model
        model_record.append(model)
        lam_record.append(lam)
        print('chi2 record: ', chi2_record)
        print('lambda record: ', lam_record)
    return(ertmgr, lam_record, chi2_record, model_record)


def invert(**kargs):
    cmd_args = get_cmd()
    args = update_args(cmd_args, dict_args=kargs)
    args = check_args(args)
    for f in args.fName:
        print('\n', '-'*80, '\n', f)
        print(f)
        f_root, f_ext = os.path.splitext(f)
        out_inv_vtk = os.path.join(args.inv_dir, f_root + args.inv_vtk)
        out_inv_png = os.path.join(args.inv_dir, f_root + args.inv_png)
        out_pareto_vtk = os.path.join(args.pareto_dir, f_root + args.pareto_vtk)
        out_pareto_csv = os.path.join(args.pareto_dir, f_root + args.pareto_csv)
        out_pareto_png = os.path.join(args.pareto_dir, f_root + args.pareto_png)
        out_misfit_png = os.path.join(args.misfit_dir, f_root + args.misfit_png)
        out_misfit_csv = os.path.join(args.misfit_dir, f_root + args.misfit_csv)
        if not os.path.isdir(args.pareto_dir):
            os.mkdir(args.pareto_dir)
        if not os.path.isdir(args.misfit_dir):
            os.mkdir(args.misfit_dir)
        if not os.path.isdir(args.inv_dir):
            os.mkdir(args.inv_dir)

        ertmgr, r_lam, r_chi2, r_model = _invert_(fName=f, mesh=args.mesh, lam=args.lam,
                                                  err=args.err, opt=args.opt,
                                                  chi2_lims=args.chi2_lims, ref=args.ref)
        save_inv_vtk(ertmgr, out_inv_vtk)
        save_misfit_csv(ertmgr, out_misfit_csv)
        save_pareto_vtk(ertmgr, r_model, r_lam, r_chi2, out_pareto_vtk)
        save_pareto_csv(r_lam, r_chi2, out_pareto_csv)
        plot_inv(out_inv_vtk, out_inv_png, 2, 12)
        plot_misfit(out_misfit_csv, out_misfit_png)
        plot_pareto(out_pareto_csv, out_pareto_png)
        if args.keep_lam:
            args.lam = r_lam[-1]
        yield(out_inv_vtk)


if __name__ == '__main__':
    invert()
