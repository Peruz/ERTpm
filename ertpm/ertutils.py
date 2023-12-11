import os
import warnings
import math
import numpy as np
from scipy.stats import skew
import matplotlib.pyplot as plt
from matplotlib.ticker import Locator
import pandas as pd


def process_rec(a: np.uint16, b: np.uint16, m: np.uint16, n: np.uint16, x: np.float64) -> (np.uint16, np.float64, np.float64, np.uint8):
    """
    Reciprocal pairing and check with polarity.
    1 2 3 4
    a m n b d i
    m a b n + j
    m b a n - j
    n a b m - j
    n b a m + j
    """
    len_sequence = int(len(x))
    rec_num = np.zeros_like(x, dtype=np.uint16)
    rec_avg = np.zeros_like(x, dtype=np.float64)
    rec_err = np.zeros_like(x, dtype=np.float64)
    rec_fnd = np.zeros_like(x, dtype=np.uint8)
    for i in range(len_sequence):
        if rec_num[i] != 0:
            continue
        for j in range(i + 1, len_sequence):
            polarity = 1

            if a[i] == m[j] and b[i] == n[j] and m[i] == a[j] and n[i] == b[j]:
                polarity = 1
            elif a[i] == m[j] and b[i] == n[j] and m[i] == b[j] and n[i] == a[j]:
                polarity = -1
            elif a[i] == n[j] and b[i] == m[j] and m[i] == a[j] and n[i] == b[j]:
                polarity = -1
            elif a[i] == n[j] and b[i] == m[j] and m[i] == b[j] and n[i] == a[j]:
                polarity = 1
            else:
                continue

            if rec_fnd[j] == 2:
                print("a second direct measurement would match this reciprocal: ", j + 1)
                print("ignore and look for a yet-to-match reciprocal")
                continue

            avg = (x[i] + (polarity * x[j])) / 2
            err = abs(x[i] - (polarity * x[j])) / abs(avg) * 100

            if polarity == -1:
                print("fixing polarity")
                print(a[i], b[i], m[i], n[i], x[i], avg, err)
                print(a[j], b[j], m[j], n[j], x[j], avg * polarity, err)

            rec_num[i] = j + 1
            rec_num[j] = i + 1
            rec_avg[i] = avg
            rec_avg[j] = avg * polarity
            rec_err[i] = err
            rec_err[j] = err
            rec_fnd[i] = 1  # mark meas as direct
            rec_fnd[j] = 2  # mark meas as reciprocal (keep 0 for unpaired)
            break

    Cnts = np.bincount(rec_fnd)
    if len(Cnts) == 1:
        unpairedCnt = Cnts[0]
        assert unpairedCnt == len(rec_fnd)
    elif len(Cnts) == 3:
        unpairedCnt, directCnt, reciprocalCnt = Cnts
        assert directCnt == reciprocalCnt
        assert directCnt + reciprocalCnt + unpairedCnt == len(rec_fnd)
    else:
        raise ValueError("failed reciprocity sanity check")

    return rec_num, rec_avg, rec_err, rec_fnd


def find_threshold_minnonzero(values, min_threshold=0.00001):
    values = np.array(values)
    min_nonzero = min([abs(v) for v in values if (v != 0) & (v is not None)])
    min_nonzero_rounded_log10 = 10 ** math.floor(np.log10(min_nonzero))
    threshold = max(min_nonzero_rounded_log10, min_threshold)
    return threshold


def find_best_yscale(values, lim_var=0.8, lim_skew=0.8):
    values = np.array(values)
    scale = 'linear'
    if len(values) == 0:
        warnings.warn('no values')
        return (scale, None, None)
    elif len(values) <= 2:
        warnings.warn('only 2 values')
        return (scale, None, None)
    vskewness = skew(values)
    vstd = np.std(values)
    vmedian = np.median(values)
    vstdmedian = vstd / vmedian
    if (abs(vstdmedian) > lim_var) | (abs(vskewness) > lim_skew):
        if any(values <= 0):
            scale = 'symlog'
        else:
            scale = 'log'
    return (scale, vstdmedian, vskewness)


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on
    the positions of major ticks for a symlog scaling.
    """

    def __init__(self, linthresh):
        """
        Ticks will be placed between the symlog major ticks.

        The placement is:
        * linear for x between -linthresh and +linthresh,
        * logarithmic below -linthresh and above this +linthresh
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()
        # my changes to previous solution
        # this adds one majortickloc below and above the axis range
        # to extend the minor ticks outside the range of majorticklocs
        # bottom of the axis (low values)
        first_major = majorlocs[0]
        if first_major == 0:
            outrange_first = -self.linthresh
        else:
            outrange_first = first_major * float(10) ** (- np.sign(first_major))
        # top of the axis (high values)
        last_major = majorlocs[-1]
        if last_major == 0:
            outrange_last = self.linthresh
        else:
            outrange_last = last_major * float(10) ** (np.sign(last_major))
        majorlocs = np.concatenate(([outrange_first], majorlocs, [outrange_last]))

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            major_current = majorlocs[i]
            major_previous = majorlocs[i - 1]
            majorstep = major_current - major_previous
            # print('major curernt: ', major_current, ' major previous: ', major_previous)
            if abs(major_previous + majorstep / 2) < self.linthresh:
                ndivs = 10  # linear gets 10 because it starts from 0 (i.e., 0 to threshold)
            else:
                ndivs = 9  # log gets 9 because there is no zero (e.g., 1 to 10)
            minorstep = majorstep / ndivs
            locs = np.arange(major_previous, major_current, minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError(
            'Cannot get tick locations for a {} type.'.format(type(self))
        )


def output_file(old_fname, new_ext, directory='processing'):
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


def naive_pairs(l):
    for c in l:
        for r in l:
            if c == r:
                pass
            else:
                yield (c, r)


def crossvalidity(df, summary_col_header="valid"):
    """
    1 is valid and 0 is invalid, the table will report the number of rejections
    """
    if summary_col_header not in df.columns.tolist():
        df[summary_col_header] = df.all(axis=1)

    cols = df.columns.tolist()

    dfc = pd.DataFrame(index=cols, columns=cols)

    # off-diagonal: number of data points that both filters reject
    for p0, p1 in naive_pairs(cols):
        common_rejections = np.sum((~df[[p0, p1]]).all(axis=1))
        dfc.loc[p0, p1] = common_rejections
        dfc.loc[p1, p0] = common_rejections

    # diagonal: number of data points that only that filter reject
    # remove the summary column, it would always match the other columns
    # ns: no summary
    cols.remove(summary_col_header)
    dfns = ~df[cols]
    sum_reject = dfns.sum(axis=1)
    dfns_unique = dfns[sum_reject == 1]
    colsns_unique_reject = dfns_unique.sum(axis=0)
    for c in cols:
        dfc.loc[c, c] = colsns_unique_reject[c]
    dfc.loc[summary_col_header, summary_col_header] = np.sum(~df[summary_col_header])
    return dfc


if __name__ == '__main__':

    # add here some tests
    values_list = [
        np.array([-10, 0.1, 1, 2, 1.5, 1.2, 0.5, 0.7, 10.5]),
        np.array([0.1, 1, 2, 1.5, 1.2, 0.5, 0.7, 10.5]),
        np.array([1, 2, 3, 4, 5]),
    ]
    for values in values_list:
        scale, vvariation, vskewness = find_best_yscale(values)
        print(scale, vvariation, vskewness)
        fig, ax = plt.subplots()
        plt.plot(values, 'o')
        if scale == 'symlog':
            threshold = find_threshold_minnonzero(values)
            print('threshold: ', threshold)
            ax.set_yscale('symlog', linthresh=threshold)
            plt.minorticks_on()
            ax.yaxis.set_minor_locator(MinorSymLogLocator(threshold))
        else:
            plt.yscale(scale)
        ax.grid(which='both', axis='both')
        plt.show()
