""" ERT PROCESSING
it gets arguments from cmd-line and/or dictionary (e.g., ert manager script)
it uses a ERT processing class that delegates to two dataframes for data and elec tables
it filters the data based on args; set very loose args values to avoid filtering.  """

import os
import re
import argparse
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
import itertools

try: from numba import jit
except ImportError: numba_opt = False
else: numba_opt = True

def get_cmd():
    """ get command line arguments for data processing """
    parse = argparse.ArgumentParser()
    main = parse.add_argument_group('main')
    filters = parse.add_argument_group('filters')
    adjustments = parse.add_argument_group('adjustments')
    outputs = parse.add_argument_group('output')
    # MAIN
    main.add_argument('-fName', type=str, help='data file to process', nargs='+')
    main.add_argument('-fType', type=str, help='data type to process', default='labrecque')
    main.add_argument('-dir_proc', type=str, help='output directory', default='processing')
    # FILTERS
    filters.add_argument('-ctc', type=float, default=1E+5, help='max ctc, ohm')
    filters.add_argument('-stk', type=float, default=5, help='max stacking err, pct')
    filters.add_argument('-v', type=float, default=1E-5, help='min voltage, V')
    filters.add_argument('-rec', type=float, default=5, help='max reciprocal err, pct')
    filters.add_argument('-rec_couple', action='store_true', default=True, help='couple reciprocals')
    filters.add_argument('-rec_unpaired', action='store_true', default=True, help='keep unpaired')
    filters.add_argument('-k', type=float, default=1E+6, help='max geometrical factor, m')
    filters.add_argument('-k_file', type=str, help='file with geom factors and activates k and rhoa')
    filters.add_argument('-rhoa', default=[0, 1E+5], type=float, help='rhoa (min, max)', nargs='+')
    # OUTPUT
    outputs.add_argument('-w_rhoa', action='store_true', help='if true, write rhoa')
    outputs.add_argument('-w_ip', action='store_true', help='if true, write phase')
    outputs.add_argument('-w_err', type=str, default=None, help='error to write')
    outputs.add_argument('-plot', action='store_true', default=True, help='plot data quality figures')
    # ADJUSTMENTS
    adjustments.add_argument('-s_abmn', type=int, default=0, help='shift abmn')
    adjustments.add_argument('-s_meas', type=int, default=0, help='shift measurement number')
    adjustments.add_argument('-s_elec', type=int, default=0, help='shift electrode number')
    adjustments.add_argument('-f_elec', type=str, default=None, help='electrode coordinates: x y z')
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
    if isinstance(args.fName, str): args.fName = [args.fName]
    if (args.k_file is None and args.w_rhoa is True):
        error = """cannot calculate and write rhoa without k_file with geometric factors,
                this wont break the code (default rhoa=None) but it may be and argument err"""
        raise ValueError(error)
    return(args)

def output_file(old_fname, new_ext='.dat', directory='.'):
    """ return name for the output file and clean them if already exist"""
    f, old_ext = os.path.splitext(old_fname)
    new_fname = f + new_ext
    new_dfname = os.path.join(directory, new_fname)
    if not os.path.isdir(directory): os.mkdir(directory)
    elif os.path.exists(new_dfname): os.remove(new_dfname)
    return(new_dfname)


def read_labrecque(f=None):
    """ read a labrecque data file an return data and electrode dataframes"""
    AppRes = False
    FreDom = False
    with open(f) as fid:
        for i, l in enumerate(fid):
            if 'Appres' in l: AppRes = True
            elif 'FStcks' in l: FreDom = True
            elif 'elec_start' in l: es = i + 1
            elif 'elec_end' in l: ee = i - 1
            elif 'data_start' in l: ds = i + 1
            elif 'data_end' in l: de = i - 1
    # data
    c = {'! ID':'meas','A':'a','B':'b','M':'m','N':'n','Date_And_Time':'datetime'}
    t = {'meas':'Int16','a':'Int16','b':'Int16','m':'Int16','n':'Int16','r':float,
         'ip':float,'v':float,'ctc':float,'stk':float,'datetime':'datetime64[ns]'}
    if FreDom:
        c.update({'Amplitude,':'r','Real-Raw,':'v','Real-Std':'stk','Phase':'ip','ContctR':'ctc'})
        r = '(?<!\\!)(?<!\D\\.)(?<!CH)(?<!GN)(?<!GN\s\d)\s+'
    else:
        c.update({'V/I,':'r','Amp.,':'v','Std.':'stk','ContactR':'ctc'})
        r = '(?<!\\!)(?<!CH)(?<!GN)(?<!CH\s\d{2})(?<!IP)(?<!Window)\s+'
    dn = de - ds
    data = pd.read_csv(f, skiprows=ds, usecols=list(c), nrows=dn,
                       sep=r, error_bad_lines=False, engine='python', index_col=False)
    data.drop(0, inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.rename(columns=c, inplace=True)
    data['a'] = data['a'].str.extract('((?<=,)\d+)').astype(int)
    data['b'] = data['b'].str.extract('((?<=,)\d+)').astype(int)
    data['m'] = data['m'].str.extract('((?<=,)\d+)').astype(int)
    data['n'] = data['n'].str.extract('((?<=,)\d+)').astype(int)
    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y%m%d_%H%M%S')
    data['meas'] = data['meas'].astype(int)
    if not FreDom: data['ip'] = np.NaN
    data = data.astype(t)
    data['stk'] = data['stk'] / data['v'] * 100
    # elec
    ec = {'El#': 'num', 'Elec-X': 'x', 'Elec-Y': 'y', 'Elec-Z': 'z'}
    et = {'num': 'Int16', 'x': float, 'y': float, 'z': float}
    en = ee - es
    elec = pd.read_csv(f, skiprows=es, usecols=list(ec), nrows=en,
                       sep='\\,|\s+', index_col=False, engine='python')
    elec = elec.rename(columns=ec)
    elec = elec.astype(et)
    return(elec, data)


def read_bert(k_file=None):
    """read bert-type file and return elec and data"""
    with open(k_file) as fid:
        lines = fid.readlines()
    elec_num = int(lines[0])
    data_num = int(lines[elec_num + 2])
    elec_raw = pd.read_csv(k_file, delim_whitespace=True, skiprows=1, nrows=elec_num, header=None)
    elec = elec_raw[elec_raw.columns[:-1]]
    elec.columns = elec_raw.columns[1:]
    data_raw = pd.read_csv(k_file, delim_whitespace=True, skiprows=elec_num + 3, nrows=data_num)
    data = data_raw[data_raw.columns[:-1]]
    data.columns = data_raw.columns[1:]
    return(elec, data)


def fun_rec(a: np.ndarray, b: np.ndarray, m: np.ndarray, n: np.ndarray, x: np.ndarray):
    l = int(len(x))
    rec_num = np.zeros_like(x)
    rec_avg = np.zeros_like(x)
    rec_err = np.zeros_like(x)
    rec_fnd = np.zeros_like(x)
    for i in range(l):
        if rec_num[i] != 0:
            continue
        for j in range(i + 1, l):
            if (a[i] == m[j] and b[i] == n[j] and m[i] == a[j] and n[i] == b[j]):
                avg = (x[i] + x[j]) / 2
                err = abs(x[i] - x[j]) / abs(avg) * 100
                rec_num[i] = j + 1
                rec_num[j] = i + 1
                rec_avg[i] = avg
                rec_avg[j] = avg
                rec_err[i] = err
                rec_err[j] = err
                rec_fnd[i] = 1  # mark the meas with reciprocals, else leave 0
                rec_fnd[j] = 2  # distinguish between directs and reciprocals
                break
    return(rec_num, rec_avg, rec_err, rec_fnd)

if numba_opt:
    signature = 'UniTuple(float64[:],4)(int64[:],int64[:],int64[:],int64[:],float64[:])'
    fun_rec = jit(signature_or_function=signature, nopython=True,
                  parallel=False, cache=True, fastmath=True, nogil=True)(fun_rec)


class ERTdataset():
    """ A dataset class composed of two dataframes data and elec.
    delegation to pandas dataframes is use for data and elec tables """

    data_headers=['meas', 'a', 'b', 'm', 'n',
               'r', 'k', 'rhoa', 'ip', 'v', 'ctc', 'stk', 'datetime',
               'rec_num', 'rec_fnd', 'rec_avg', 'rec_err', 'rec_ip_avg', 'rec_ip_err',
               'rec_valid', 'k_valid', 'rhoa_valid', 'v_valid', 'ctc_valid', 'stk_valid',
               'valid']
    data_dtypes={'meas': 'Int16', 'a': 'Int16', 'b': 'Int16', 'm': 'Int16', 'n': 'Int16',
              'r': float, 'k': float, 'rhoa': float, 'ip': float,
              'v': float, 'ctc': float, 'stk': float, 'datetime': 'datetime64[ns]',
              'rec_num': 'Int16', 'rec_fnd': bool,
              'rec_avg': float, 'rec_err': float,
              'rec_ip_avg': float, 'rec_ip_err': float,
              'rec_valid': bool, 'k_valid': bool, 'rhoa_valid': bool, 'v_valid': bool,
              'ctc_valid': bool, 'stk_valid': bool, 'valid': bool}
    elec_headers=['num', 'x', 'y', 'z']
    elec_dtypes={'num': 'Int16', 'x': float, 'y': float, 'z': float}

    def __init__(self, data=None, elec=None):
        self.data = None
        self.elec = None

        if data is not None:
            self.init_EmptyData(data_len=len(data))
            self.data.update(data)
            self.data = self.data.astype(self.data_dtypes)

        if elec is not None:
            self.init_EmptyElec(elec_len=len(elec))
            self.elec.update(elec)
            self.elec = self.elec.astype(self.elec_dtypes)

    def init_EmptyData(self, data_len=None):
        """ wrapper to create empty (None) data dataframe with the proper headers and datatypes."""
        self.data = pd.DataFrame(None, index=range(data_len), columns=self.data_headers)
        self.data = self.data.astype(self.data_dtypes)

    def init_EmptyElec(self, elec_len=None):
        """ wrapper to create empty (None) data dataframe with the proper headers and datatypes."""
        self.elec = pd.DataFrame(None, index=range(elec_len), columns=self.elec_headers)
        self.elec = self.elec.astype(self.elec_dtypes)

    def default_types(self):
        self.data = self.data.astype(self.data_dtypes)

    def process_rec(self, fun_rec=fun_rec, x='r', x_avg='rec_avg', x_err='rec_err'):
        a = self.data['a'].to_numpy(dtype=int)
        b = self.data['b'].to_numpy(dtype=int)
        m = self.data['m'].to_numpy(dtype=int)
        n = self.data['n'].to_numpy(dtype=int)
        x = self.data['r'].to_numpy(dtype=float)
        rec_num, rec_avg, rec_err, rec_fnd = fun_rec(a, b, m, n, x)
        self.data['rec_num'] = rec_num
        self.data['rec_fnd'] = rec_fnd
        self.data[x_avg] = rec_avg
        self.data[x_err] = rec_err

    def get_k(self, data_k):
        if len(self.data) == len(data_k):
            self.data['k'] = data_k['k']
        elif len(data_k) < len(self.data):
            raise IndexError('len k < len data, make sure the right k file is used')
        elif len(self.data) < len(data_k):
            warnings.warn('len k != len data; make sure the right k file is used')
            abmn = ['a', 'b', 'm', 'n']
            abmnk = ['a', 'b', 'm', 'n', 'k']
            self.data = self.data.merge(data_k[abmnk], on=abmn, how='left', suffixes=('', '_'))
            self.data['k'] = self.data['k_']
            self.data.drop(columns='k_', inplace=True)

    def couple_rec(self, couple=False, unpaired=False, dir_mark=1, rec_mark=2, unpaired_mark=0):
        if (couple and unpaired):
            direct = self.data.loc[self.data['rec_fnd'] == dir_mark]
            unpaired = self.data.loc[self.data['rec_fnd'] == unpaired_mark]
            self.data = pd.concat([direct, unpaired])
        elif (couple and not unpaired):
            direct = self.data.loc[self.data['rec_fnd'] == dir_mark]
            self.data = direct
        elif (not couple and unpaired):
            self.data = self.data

    def to_bert(self, fname, w_rhoa, w_ip, w_err, data_cols, elec_cols):
        if w_rhoa: data_cols.append('rhoa')
        if w_ip: data_cols.append('ip')
        if w_err: data_cols.append(w_err)
        with open(fname, 'a') as file_handle:
            file_handle.write(str(len(self.elec)) + '\n')
            file_handle.write('# ' + ' '.join(elec_cols) + '\n')
            self.elec[elec_cols].to_csv(file_handle, sep=' ', index=None, header=False)
            data_wrt = self.data[self.data.valid == 1][data_cols]
            file_handle.write(str(len(data_wrt)) + '\n')
            file_handle.write('# ' + ' '.join(data_cols) + '\n')
            data_wrt.to_csv(file_handle, sep=' ', index=None, header=False)

    def plot(self, fname, plot_columns, valid_column='valid', dir_proc='.'):
        colors_validity = {1: 'b', 0: 'r'}
        labels_validity = {1: 'Valid', 0: 'Invalid'}
        groupby_df = self.data.groupby(self.data['valid'])
        for key in groupby_df.groups.keys():  # for group 1 (valid) and group 0 (invalid)
            meas = groupby_df.get_group(key)['meas'].to_numpy(dtype=int)
            for c in plot_columns:
                fig_fname = fname.replace('.Data', '_') + labels_validity[key] + '_' + c + '.png'
                fig_dfname = os.path.join(dir_proc, fig_fname)
                y = groupby_df.get_group(key)[c].to_numpy()
                plt.plot(meas, y, 'o', color=colors_validity[key], markersize=4)
                plt.ylabel(c)
                plt.xlabel('measurement num')
                plt.tight_layout()
                plt.savefig(fig_dfname)
                plt.close()

    def report(self, cols=['valid']):
        for c in cols:
            print('-----\n', self.data[c].value_counts())

def __process__(f, args):
    """ process ERT file """
    if args.fType == 'labrecque':
        elec, data = read_labrecque(f)
    # pass to ERTdataset class
    ds = ERTdataset(data=data, elec=elec)
    # adjust
    ds.data[['a', 'b', 'm', 'n']] += args.s_abmn
    ds.data['meas'] += args.s_meas
    ds.elec['num'] += args.s_elec
    if args.f_elec is not None:
        ds.elec = pd.read_csv(args.f_elec, delim_whitespace=True)
    # filters
    ds.process_rec(x='r', x_avg='rec_avg', x_err='rec_err')
    ds.data['rec_valid'] = ds.data['rec_err'] < args.rec
    ds.data['ctc_valid'] = ds.data['ctc'] < args.ctc
    ds.data['stk_valid'] = ds.data['stk'] < args.stk
    ds.data['v_valid'] = ds.data['v'].abs() > args.v
    if all(ds.data['ip'].notnull()):
        ds.process_rec(x='ip', x_avg='rec_ip_avg', x_err='rec_ip_err')
    if args.k_file:
        elec_kfile, data_kfile = read_bert(k_file=args.k_file)
        if not any(data_kfile['k']):
            print('!!! calculating k from r, assuming rho == 1')
            data_kfile['k'] = 1 / data_kfile['r']
        ds.get_k(data_kfile)
        ds.data['k_valid'] = ds.data['k'].abs() < args.k
        ds.data['rhoa'] = ds.data['r'] * ds.data['k']
        ds.data['rhoa_valid'] = ds.data['rhoa'].between(args.rhoa[0], args.rhoa[1])
    # combine filters
    filters = ['rec_valid', 'k_valid', 'rhoa_valid', 'v_valid', 'ctc_valid', 'stk_valid']
    ds.data['valid'] = ds.data[filters].all(axis='columns')
    # output csv
    ds.default_types()
    fcsv = output_file(f, new_ext='.csv', directory=args.dir_proc)
    ds.data.to_csv(fcsv, float_format='%#8g', index=False)
    ds.couple_rec(couple=args.rec_couple, unpaired=args.rec_unpaired)
    # output dat
    fdat = output_file(f, new_ext='.dat', directory='.')
    data_cols = ['a', 'b', 'm', 'n', 'r']
    elec_cols = ['x', 'y', 'z']
    ds.to_bert(fdat, args.w_rhoa, args.w_ip, args.w_err, data_cols, elec_cols)
    # report
    report_columns = ['rec_valid', 'k_valid', 'rhoa_valid', 'v_valid', 'ctc_valid', 'stk_valid']
    ds.report(cols=report_columns)
    # plot
    plot_columns = ['ctc', 'stk', 'v', 'rec_err', 'k', 'rhoa']
    ds.plot(f, plot_columns=plot_columns, valid_column='valid', dir_proc=args.dir_proc)
    # datetime
    fdatetime = ds.data.loc[0, 'datetime']
    return(fcsv, fdat, fdatetime)

def process(**kargs):
    """ get args both as command line and as function arguments, then process files """
    cmd_args = get_cmd()
    args = update_args(cmd_args, dict_args=kargs)
    args = check_args(args)

    for f in args.fName:
        print('\n', '-'*80, '\n', f)
        print(f)
        fcsv,  fdat, datetime = __process__(f, args)
        yield(fcsv, fdat, datetime)

if __name__ == '__main__':
    process()
