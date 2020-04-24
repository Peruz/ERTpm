import os
import argparse
import numpy as np
import pandas as pd
import pyvista as pv

def get_cmd():
    """ get command line arguments for data processing """
    parse = argparse.ArgumentParser()
    main = parse.add_argument_group('main')
    option = parse.add_argument_group('option')
    output = parse.add_argument_group('output')
    # MAIN
    main.add_argument('-csv_datasets', type=str, help='csv file with information on the datasets')
    main.add_argument('-rho', type=str, default='res', help='vtk resistivity name')
    main.add_argument('-csv_vols', type=str, help='csv file with volumes names and coords')
    main.add_argument('-datetime_col', type=str, help='header of datetime column')
    main.add_argument('-vtk_col', type=str, help='header of the column with the vtk files')
    main.add_argument('-how', type=str, default='timelapse', help='single or timelapse mode')
    # OUTPUT
    output.add_argument('-dir_analysis', type=str, help='output directory', default='analysis')
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

def read_csv_datasets(fName):
    ds = pd.read_csv(fName)
    return(ds)

def read_csv_vols(fName):
    vols = pd.read_csv(fName, index_col=False)
    return(vols)

def init_table(ds, vols, vtk_col, datetime_col):
    nv = len(vols)
    nds = len(ds)
    nrows = nds * nv
    columns = ['vtk_name', 'datetime', 'vol_name', 'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
    table = pd.DataFrame(index=range(nrows), columns=columns)
    # vtk files
    file_vtk = ds[vtk_col].to_numpy()
    file_vtk = np.tile(file_vtk, (nv, 1))
    file_vtk = file_vtk.flatten('F')
    table['vtk_name'] = file_vtk
    file_datetime = ds[datetime_col].to_numpy()
    file_datetime = np.tile(file_datetime, (nv, 1))
    file_datetime = file_datetime.flatten('F')
    table['datetime'] = file_datetime
    # coords
    vols_coord = vols.to_numpy()
    vols_coord = np.tile(vols_coord, (nds, 1))
    vols_coord = pd.DataFrame(vols_coord)
    # table
    table = np.column_stack((file_vtk, file_datetime, vols_coord))
    table = pd.DataFrame(table, columns=columns)
    return(table)

def find_rho(table):
    df_grouped_files = table.groupby('vtk_name')
    #TODO

def _analysis_(args):
    ds = read_csv_datasets(args.csv_datasets)
    vols = read_csv_vols(args.csv_vols)
    table = init_table(ds, vols, args.vtk_col, args.datetime_col)

def analysis(**kargs):
    cmd_args = get_cmd()
    args = update_args(cmd_args, dict_args=kargs)
    pv.set_plot_theme("document")
    _analysis_(args)

if __name__ == '__main__':
    analysis()

