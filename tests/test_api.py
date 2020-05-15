from ERTpm.manager import select_table, init_table, update_table
from ERTpm.manager import table_headers, table_dtypes, table_name
from ERTpm.process import process
from ERTpm.invert import invert
from ERTpm.plot2d import plot2d


do_process = True
do_invert = True
do_plot2d = True
do_analysis = True
data_ext = '.Data'  # labrecque
table = init_table(table_name, table_headers, table_dtypes)
table = update_table(table, table_name, data_ext)
table.sort_values(by='datetime', inplace=True)  # on updated datetime column
print(table)

if do_process:
    print('\nPROCESSING')
    table_to_process = select_table(table, which='new',
                                    col_check='process', col_needed='file')
    if table_to_process.empty:
        print('no new files')
    else:
        for i, r in table_to_process.iterrows():
            f = r['file']
            if 'DD2' in f:
                k_file = 'sim_k_dd2.data'
            elif 'GRAD' in f:
                k_file = 'sim_k_grad.data'
            else:
                raise ValueError('no geometric factors for this file name')
            fyield = process(fName=f,
                             k_file=k_file,
                             rec=5,
                             rhoa=(0, 1E+3),
                             w_rhoa=True)
            fcsv, finv, datetime = next(fyield)
            process_columns = ['process', 'fcsv', 'finv', 'datetime']
            process_values = [True, fcsv, finv, datetime]
            table.loc[table['file'] == f, process_columns] = process_values
        table = update_table(table, table_name, data_ext)
        # based on updated datetime column
        table.sort_values(by='datetime', inplace=True)


if do_invert:
    print('\nINVERSION')
    table_to_invert = select_table(table,
                                   which='new',
                                   col_check='invert',
                                   col_needed='finv')
    if table_to_invert.empty:
        print('no new files')
    else:
        for i, r in table_to_invert.iterrows():
            f = r['file']
            finv = r['finv']
            fref = None
            fyield = invert(fName=finv, mesh='mesh.bms', lam=62, err=0.05, opt=True)
            fvtk = next(fyield)
            table.loc[table['file'] == f, ['invert', 'fvtk']] = True, fvtk
        table = update_table(table, table_name, data_ext)

if do_plot2d:
    print('\nPLOT')
    table_to_plot = select_table(table,
                                 which='all',
                                 col_check='plot',
                                 col_needed='fvtk')
    if table_to_plot.empty:
        print('no new files')
    else:
        for i, r in table_to_plot.iterrows():
            f = r['file']
            fvtk = r['fvtk']
            dName = None
            gen_fpng = plot2d(fName=fvtk, dName=dName, Cmin=None, Cmax=None)
            fpng = next(gen_fpng)
            table.loc[table['file'] == f, ['plot', 'fpng']] = True, fpng
        table = update_table(table, table_name, data_ext)
