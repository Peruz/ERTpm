"""
# api for managing ert datasets, their processing and inversion

1) create a table (df) which sortes:
    a) main information on the acquisition and project
    b) status of each dataset (processed, inverted, etc.)
    c) parameters used for processing and inverting each dataset
    d) file names (raw file, processed file, inverted rho(xyz))
2) manage the ERT datasets in time-lapse
    a) organaze them in chronological order
    b) mark dataset validity
3) manage inversion:
    a) set reference model (none, previous, or absolute reference)
    b) reference is used as model but possibly also for the parameters

------
notes:
* some defaults values are given for convenience
but use import * to avoid depending on module name
"""

import os
import pandas as pd
from ERTpm.invert import invert
from ERTpm.process import process
from ERTpm.plot2d import plot2d


def init_table(table_name, table_headers, table_dtypes):
    """ init table, read existing or create one and save it """
    if table_name in os.listdir():
        table = pd.read_csv(table_name, parse_dates=['datetime'])
    else:
        table = pd.DataFrame(columns=table_headers)
        table = table.astype(table_dtypes)
        table.to_csv(table_name, index=False)
    return(table)


def update_table(table, table_name, data_ext):
    """ check direcotry and update existing table, both file and dataframe """
    dir_files = [f for f in os.listdir() if f.endswith(data_ext)]
    dir_files_df = pd.DataFrame({'file': dir_files})
    table = pd.concat([table, dir_files_df], ignore_index=True)
    table = table.drop_duplicates('file', keep='first', ignore_index=True)
    table.to_csv(table_name, index=False)
    return(table)


def select_table(table, which, col_check, col_needed):
    """
    which (str, list): whether to keep all, new, or only specific data
    col_check (str): check among available data if already done or not
    col_needed (str): data needed for action, cannot invert if not processed
    """
    table = table.dropna(subset=[col_needed])
    if isinstance(which, str):
        if which == 'all':
            files = table
        elif which == 'new':
            if col_check is None:
                raise ValueError('to check col_check for new files, col_check cannot be None')
            files = table[pd.isnull(table[col_check])]
        elif table['file'].str.contains(which).any():
            files = table.loc[table['file'] == which]
        else:
            raise ValueError(which,
                             ' is not a valid string [new, all, filename]')
    else:
        files = table[table['file'].isin(which)]
    return(files)


# some default values
table_name = 'ert_datasets.txt'
table_headers = ['file', 'datetime', 'process', 'invert', 'plot', 'fcsv', 'finv', 'fvtk', 'fpng']
table_dtypes = {'file': str, 'datetime': 'datetime64[ns]',
                'process': bool, 'invert': bool, 'plot': bool,
                'fcsv': str, 'finv': str, 'fvtk': str, 'fpng': str}

if __name__ == '__main__':
    do_process = True
    do_invert = True
    do_plot2d = True
    data_ext = '.Data'  # labrecque
    table = init_table(table_name, table_headers, table_dtypes)
    table = update_table(table, table_name, data_ext)
    table.sort_values(by='datetime', inplace=True)  # based on updated datetime column
    print(table)

    if do_process:
        print('\nPROCESSING')
        table_to_process = select_table(table, which='new', col_check='process', col_needed='file')
        if table_to_process.empty:
            print('no new files')
        else:
            for i, r in table_to_process.iterrows():
                f = r['file']
                if 'GRAD' in f:
                    k_file = 'kfiles/sim_grad_nowen.data'
                elif 'DD' in f:
                    k_file = 'kfiles/sim_dd2.data'
                else:
                    raise ValueError('no file with geometric factors for this file name')
                fyield = process(fName=f, k_file=k_file, rec=5, rhoa=(0, 1E+3), w_rhoa=True)
                fcsv, finv, datetime = next(fyield)
                process_columns = ['process', 'fcsv', 'finv', 'datetime']
                process_values = [True, fcsv, finv, datetime]
                table.loc[table['file'] == f, process_columns] = process_values
            table = update_table(table, table_name, data_ext)
            table.sort_values(by='datetime', inplace=True)  # based on updated datetime column

    if do_invert:
        print('\nINVERSION')
        table_to_invert = select_table(table, which='new', col_check='invert', col_needed='finv')
        if table_to_invert.empty:
            print('no new files')
        else:
            for i, r in table_to_invert.iterrows():
                f = r['file']
                finv = r['finv']
                fref = None
                fyield = invert(fName=finv, mesh='mesh/mesh.bms',
                                lam=400, err=0.05, opt=True)
                fvtk = next(fyield)
                table.loc[table['file'] == f, ['invert', 'fvtk']] = True, fvtk
            table = update_table(table, table_name, data_ext)

    if do_plot2d:
        print('\nPLOT')
        table_to_plot = select_table(table, which='all', col_check='plot', col_needed='fvtk')
        if table_to_plot.empty:
            print('no new files')
        else:
            for i, r in table_to_plot.iterrows():
                f = r['file']
                fvtk = r['fvtk']
                dName = None
                gen_fpng = plot2d(fName=fvtk, dName=dName)
                fpng = next(gen_fpng)
                table.loc[table['file'] == f, ['plot', 'fpng']] = True, fpng
            table = update_table(table, table_name, data_ext)
