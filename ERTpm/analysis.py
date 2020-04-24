import os
import argparse
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import seaborn
import datetime

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
    file_datetime = ds[datetime_col].to_numpy()
    file_datetime = np.tile(file_datetime, (nv, 1))
    file_datetime = file_datetime.flatten('F')
    # coords
    vols_coord = vols.to_numpy()
    vols_coord = np.tile(vols_coord, (nds, 1))
    vols_coord = pd.DataFrame(vols_coord)
    # table
    table = np.column_stack((file_vtk, file_datetime, vols_coord))
    table = pd.DataFrame(table, columns=columns)
    return(table)

def add_avg_to_table(table, rho):
    """ table has the file names and the boundaries of the regions
    we group by vtk file, to read it only once for all regions
    for each vtk file, we loop through the regions, find the average, add it to table
    """
    grouped_table = table.groupby('vtk_name')
    for name, group in grouped_table:
        xyzr = get_xyzr(name, rho)
        for i, r in group.iterrows():
            avg, num = get_region_info(xyzr, r)
            table.loc[i, 'ravg'] = avg
            table.loc[i, 'rnum'] = num
    return(table)

def get_xyzr(fName, rho):
    """ read vtk and extract value and center of each cell in it
    """
    mesh = pv.read(fName)
    rho = mesh[rho]
    centers = mesh.cell_centers().points
    xyzr = np.column_stack((centers, rho))
    xyzr = pd.DataFrame(xyzr, columns=['x', 'y', 'z', 'r'])
    return(xyzr)

def get_region_info(xyzr, r):
    """ xyzr contains the data, table contains the regions """
    region_cells = xyzr[(xyzr['x'].between(r['xmin'], r['xmax'])) &
                        (xyzr['y'].between(r['ymin'], r['ymax']))]# &
                        #(xyzr['z'].between(r['zmin'], r['zmax']))]
    r_avg = region_cells['r'].mean()
    r_num = len(region_cells)
    return(r_avg, r_num)


def plot_seaborn(df, x, y, hue, output_name):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6.5))
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['datetimems'] = pd.to_datetime(df['datetime'], unit='ms')
    seaborn.scatterplot(data=df, x=x, y=y, ax=ax, s=90, hue=df[hue], alpha=1, legend='full')
    locator = matplotlib.dates.AutoDateLocator(minticks=5, maxticks=20)
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()
    ax.set_xlim([datetime.date(2019, 9, 1), datetime.date(2020, 5, 1)])
    ax.set_xlabel('time')
    ax.set_ylabel('rho')
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_name, dpi=600)
    plt.show()


def _analysis_(args):
    ds = read_csv_datasets(args.csv_datasets)
    vols = read_csv_vols(args.csv_vols)
    table = init_table(ds, vols, args.vtk_col, args.datetime_col)
    table = add_avg_to_table(table, args.rho)
    plot_seaborn(table, x='datetime', y='ravg', hue='vol_name', output_name='test.png')
    plot_datetime(table, 'ravg', 'test.png')
    plt.show()

def analysis(**kargs):
    cmd_args = get_cmd()
    args = update_args(cmd_args, dict_args=kargs)
    pv.set_plot_theme("document")
    _analysis_(args)

if __name__ == '__main__':
    analysis()

