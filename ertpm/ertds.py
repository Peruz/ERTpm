from IPython import embed
import os
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator
from ertpm.ertutils import MinorSymLogLocator
from ertpm.ertutils import find_threshold_minnonzero
from ertpm.ertutils import find_best_yscale
from ertpm.ertutils import output_file
from ertpm.ertutils import process_rec

"""
ERT Dataset class (also ertds or ds)

# General convenctions

x, y, z indicates the names for the electrode coordinates.
For surface 2.5 ERT data sets, x should be the array direction.

a, b, m, n indicate the electrode numbers.

r indicates the measured resistance.
rhoa indicates the apparent resistivity.
k is the geometric factor.
Therefore, rhoa = k * r.

Error
Errors are expressed in percentage to be homogeneous and intuitive.
There are 3 common errors:
1. reciprocal (rec)
2. stacking (stk)
3. base

err indicates the main/chosen error, often the reciprocal error + a base error.
err is express as percentage error = (abs(a - b) / abs(mean(a, b))) * 100
err fractional = err / 100
err sd = err * meas / 2 / 100
"""


class ERTdataset:
    """
    A dataset class composed of 2 dataframes:
    data -> actual data
    elec -> electrodes
    and a metadata dictionary
    meta -> metadata
    this way, most of the work is delegated to these dataframes
    """

    data_dtypes = {
        "meas": "Int16",
        "a": "Int16",
        "b": "Int16",
        "m": "Int16",
        "n": "Int16",
        "r": float,
        "k": float,
        "rhoa": float,
        "ip": float,
        "v": float,
        "curr": float,
        "ctc": float,
        "stk": float,
        "datetime": "datetime64[ns]",
        "rec_num": "Int16",
        "rec_fnd": "Int16",
        "rec_avg": float,
        "rec_err": float,
        "rec_valid": bool,
        "k_valid": bool,
        "rhoa_valid": bool,
        "v_valid": bool,
        "ctc_valid": bool,
        "stk_valid": bool,
        "elec_valid": bool,
        "valid": bool,
    }

    data_header = data_dtypes.keys()

    elec_dtypes = {"num": "Int16", "x": float, "y": float, "z": float}
    elec_header = elec_dtypes.keys()

    def __init__(self, data=None, elec=None, meta=None):
        self.data = None
        self.elec = None
        self.meta = None

        # data
        if data is not None:
            self.init_EmptyData(data_len=len(data))
            self.data.update(data)
            self.data = self.data.merge(data, how="outer")
            self.data = self.data.astype(self.data_dtypes)
        else:
            self.init_EmptyData(data_len=0)
        if any(self.data['meas'].isnull()):
            self.data['meas'] = np.arange(1, len(self.data) + 1, dtype=np.int16)

        # elec
        if elec is not None:
            self.init_EmptyElec(elec_len=len(elec))
            self.elec.update(elec)
            self.elec = self.elec.astype(self.elec_dtypes)
        else:
            self.init_EmptyElec(elec_len=0)
        if any(self.elec['num'].isnull()):
            self.elec['num'] = np.arange(1, len(self.elec) + 1, dtype=np.int16)
        self.elec.fillna(0, inplace=True)

        # meta
        if meta is not None:
            self.meta = meta
        else:
            self.meta = {}

    def __repr__(self):
        t = '\n'.join((self.meta.__repr__(), self.elec.__repr__(), self.data.__repr__()))
        return t

    def __str__(self):
        t = '\n'.join((self.meta.__repr__(), self.elec.__repr__(), self.data.__repr__()))
        return t

    def to_csv(self, fcsv):
        ertds_csv = "\n".join(
            (
                "meta",
                "\n".join("{}\t{}".format(k, v) for k, v in sorted(self.meta.items(), key=lambda t: str(t[0]))),
                "elec",
                self.elec.to_csv(index=False),
                "data",
                self.data.to_csv(index=False),
            )
        )
        with open(fcsv, "w") as fopen:
            print(ertds_csv, file=fopen)

    def init_EmptyData(self, data_len=None):
        """wrapper to create empty (None) data dataframe with the proper headers and datatypes."""
        self.data = pd.DataFrame(None, index=range(data_len), columns=self.data_header)
        self.data = self.data.astype(self.data_dtypes)

    def init_EmptyElec(self, elec_len=None):
        """wrapper to create empty (None) data dataframe with the proper headers and datatypes."""
        self.elec = pd.DataFrame(None, index=range(elec_len), columns=self.elec_header)
        self.elec = self.elec.astype(self.elec_dtypes)

    def default_types(self):
        self.data = self.data.astype(self.data_dtypes)
        self.elec = self.elec.astype(self.elec_dtypes)

    def set_k(self, data_k):
        """get and set k from a vector of floating numbers"""

        if len(self.data) == len(data_k):
            self.data["k"] = data_k["k"]

        elif len(self.data) < len(data_k):
            warnings.warn(
                "len k {kl} > len data {dl}: wrong k file or incomplete data".format(
                    dl=len(self.data), kl=len(data_k)
                )
            )
            abmn = ["a", "b", "m", "n"]
            abmnk = ["a", "b", "m", "n", "k"]
            self.data = self.data.merge(data_k[abmnk], on=abmn, how="left", suffixes=("", "_"))
            self.data["k"] = self.data["k_"]
            self.data.drop(columns="k_", inplace=True)

        elif len(data_k) < len(self.data):
            raise IndexError(
                "len k {kl} < len data {dl}: probably the wrong k file".format(
                    dl=len(self.data), kl=len(data_k)
                )
            )

    def calc_k_1d(self, coord="x"):
        elec = self.elec
        data = self.data
        enx = elec.set_index('num').to_dict()[coord]
        apos = elec[coord][data['a'].map(enx)].to_numpy()
        bpos = elec[coord][data['b'].map(enx)].to_numpy()
        mpos = elec[coord][data['m'].map(enx)].to_numpy()
        npos = elec[coord][data['n'].map(enx)].to_numpy()
        AM = apos - mpos
        BM = bpos - mpos
        AN = apos - npos
        BN = bpos - npos
        AM[AM == 0] = np.nan
        BM[BM == 0] = np.nan
        AN[AN == 0] = np.nan
        BN[BN == 0] = np.nan
        k = 2 * np.pi / ((1 / AM) - (1 / BM) - (1 / AN) + (1 / BN))
        self.data['k'] = k

    def calc_k_3d(self):
        """ assuming flat 2D surface """
        elec = self.elec
        data = self.data
        enx = elec.set_index('num').to_dict()['x']
        eny = elec.set_index('num').to_dict()['y']
        enz = elec.set_index('num').to_dict()['z']
        aposx = data['a'].map(enx).to_numpy()
        aposy = data['a'].map(eny).to_numpy()
        aposz = data['a'].map(enz).to_numpy()
        bposx = data['b'].map(enx).to_numpy()
        bposy = data['b'].map(eny).to_numpy()
        bposz = data['b'].map(enz).to_numpy()
        mposx = data['m'].map(enx).to_numpy()
        mposy = data['m'].map(eny).to_numpy()
        mposz = data['m'].map(enz).to_numpy()
        nposx = data['n'].map(enx).to_numpy()
        nposy = data['n'].map(eny).to_numpy()
        nposz = data['n'].map(enz).to_numpy()
        AM = np.sqrt((aposx - mposx) ** 2 + (aposy - mposy) ** 2 + (aposz - mposz) ** 2)
        BM = np.sqrt((bposx - mposx) ** 2 + (bposy - mposy) ** 2 + (bposz - mposz) ** 2)
        AN = np.sqrt((aposx - nposx) ** 2 + (aposy - nposy) ** 2 + (aposz - nposz) ** 2)
        BN = np.sqrt((bposx - nposx) ** 2 + (bposy - nposy) ** 2 + (bposz - nposz) ** 2)
        AM[AM == 0] = np.nan
        BM[BM == 0] = np.nan
        AN[AN == 0] = np.nan
        BN[BN == 0] = np.nan
        k = 2 * np.pi / ((1 / AM) - (1 / BM) - (1 / AN) + (1 / BN))
        self.data['k'] = k

    def rec_process(self, rec_quantities=['r'], rec_max=10):
        for q in rec_quantities:
            print("checking reciprocal: ", q)
            a = self.data["a"].to_numpy(dtype=np.uint16)
            b = self.data["b"].to_numpy(dtype=np.uint16)
            m = self.data["m"].to_numpy(dtype=np.uint16)
            n = self.data["n"].to_numpy(dtype=np.uint16)
            x = self.data[q].to_numpy(dtype=np.float64)
            x_avg = q + "_rec_avg"
            x_err = q + "_rec_err"
            rec_num, rec_avg, rec_err, rec_fnd = process_rec(a, b, m, n, x)
            self.data["rec_num"] = rec_num
            self.data["rec_fnd"] = rec_fnd
            self.data[x_avg] = rec_avg
            self.data[x_err] = rec_err
        rec_err_columns = [c for c in self.data.columns if "_rec_err" in c]
        self.data["rec_err"] = self.data[rec_err_columns].mean(axis=1).to_numpy()
        self.data["rec_valid"] = self.data["rec_err"] < rec_max

    def rec_couple(self, keep_unpaired=True):
        directs = self.data.loc[self.data["rec_fnd"] == 1].copy()
        directs_valid = directs.loc[directs["valid"] == True]
        directs_invalid = directs.loc[directs["valid"] == False]
        directs_invalid_reciprocals = directs_invalid["rec_num"].values
        reciprocals = self.data.loc[self.data["rec_fnd"] == 2].copy()
        reciprocals_valid = reciprocals.loc[reciprocals["valid"] == True]
        reciprocals_valid_needed = reciprocals_valid[reciprocals_valid["meas"].isin(directs_invalid_reciprocals)]
        coupled = pd.concat((directs_valid, reciprocals_valid_needed))
        if "r_rec_avg" in coupled.columns:
            coupled["r_directs"] = coupled["r"]
            coupled["r"] = coupled["r_rec_avg"]
        if "rhoa_rec_avg" in directs.columns:
            coupled["rhoa_directs"] = coupled["rhoa"]
            coupled["rhoa"] = coupled["rhoa_rec_avg"]
        if keep_unpaired:
            unpaired = self.data.loc[self.data["rec_fnd"] == 0].copy()
            unpaired_valid = unpaired.loc[unpaired["valid"] == True]
            self.data = pd.concat([coupled, unpaired_valid])
        else:
            self.data = coupled

    def format_elec_coord(self, e_num, coordinates=["x", "z"]):
        string_format = ("{:10.3f} " * len(coordinates)).strip()
        ecs = []
        for c in coordinates:
            ec = self.elec.loc[self.elec["num"] == e_num, c].to_numpy()[0]
            ecs.append(ec)
        str_elec_coord = string_format.format(*ecs)
        return str_elec_coord

    def report(self, cols=["valid"]):
        for c in cols:
            print("-----\n", self.data[c].value_counts())

    def to_bert(self, fname, w_ip=False, w_err=True, w_rhoa=False, w_k=False):
        try:
            os.remove(fname)
        except OSError:
            pass

        elec_cols = ["x", "y", "z"]
        data_cols = ["a", "b", "m", "n"]
        meas_cols = ["r"]
        # meas_cols = ["r", "rhoa", "k"]

        if w_ip:
            meas_cols.append("ip")
        if w_err:
            meas_cols.append("err")
        if w_rhoa:
            meas_cols.append("rhoa")
        if w_k:
            meas_cols.append("k")

        for mc in meas_cols:
            if not any(self.data[mc].isnull()):
                data_cols.append(mc)
        data_header = data_cols
        data = self.data[data_cols].copy()

        # if error is present, convert to percentage to fractional
        try:
            data["err"] = data["err"] / 100
        except KeyError:
            pass

        with open(fname, "a") as file_handle:
            file_handle.write(str(len(self.elec)) + "\n")
            file_handle.write("#" + " ".join(elec_cols) + "\n")
            self.elec[elec_cols].to_csv(file_handle, sep=" ", index=None, header=False)
            file_handle.write(str(len(data)) + "\n")
            file_handle.write("#" + " ".join(data_header) + "\n")
            data.to_csv(file_handle, sep=" ", index=None, header=False, float_format="%g")

    def to_ubc_xyz(self, fname, meas_col="r", w_err=True):
        data = self.data
        elec = self.elec
        if meas_col == "r":
            data_header = "V"
        if meas_col == "rhoa":
            data_header = "rhoa"
        elec_dict_x = elec.set_index("num").to_dict()["x"]
        elec_dict_y = elec.set_index("num").to_dict()["y"]
        elec_dict_z = elec.set_index("num").to_dict()["z"]
        data_xyz = pd.DataFrame(
            data={
                "XA": data["a"].map(elec_dict_x),
                "YA": data["a"].map(elec_dict_y),
                "ZA": data["a"].map(elec_dict_z),
                "XB": data["b"].map(elec_dict_x),
                "YB": data["b"].map(elec_dict_y),
                "ZB": data["b"].map(elec_dict_z),
                "XM": data["m"].map(elec_dict_x),
                "YM": data["m"].map(elec_dict_y),
                "ZM": data["m"].map(elec_dict_z),
                "XN": data["n"].map(elec_dict_x),
                "YN": data["n"].map(elec_dict_y),
                "ZN": data["n"].map(elec_dict_z),
                data_header: data[meas_col],
            }
        )
        if w_err:
            data_xyz["SD"] = np.abs(data["err"].to_numpy()) / 100 * np.abs(data[meas_col]) / 2
        data_xyz.to_csv(fname, sep=" ", index=None, header=True, float_format="%g")

    def to_res2dinv(self, fname, meas_col="rhoa", w_err=True):
        # shallow copies for the sake of brevity
        data = self.data
        elec = self.elec
        if meas_col == "rhoa":
            rhoa_r = "0"
        elif meas_col == "r":
            rhoa_r = "1"
        unit_spacing = min(np.diff(elec["x"].to_numpy()))
        fours = np.ones(len(data)) * 4
        elec_dict_x = elec.set_index("num").to_dict()["x"]
        elec_dict_z = elec.set_index("num").to_dict()["z"]
        data = pd.DataFrame(
            data={
                "f": fours,
                "ax": data["a"].map(elec_dict_x),
                "az": data["a"].map(elec_dict_z),
                "bx": data["b"].map(elec_dict_x),
                "bz": data["b"].map(elec_dict_z),
                "mx": data["m"].map(elec_dict_x),
                "mz": data["m"].map(elec_dict_z),
                "nx": data["n"].map(elec_dict_x),
                "nz": data["n"].map(elec_dict_z),
                meas_col: data[meas_col],
            }
        )
        header_lines = [
            "Mixed array",
            "{:4.2f}".format(unit_spacing),
            "11",
            "0",
            "Type of measurement (0=rhoa, 1=r)",
            rhoa_r,
            str(int(len(data))),
            "1" "0",
        ]
        if w_err:
            header_lines += ["Error estimate for data present", "Type of error estimate (0=same unit as data)", "0"]
            data["err_sd"] = np.abs(self.data["err"].to_numpy()) / 100 * np.abs(self.data[meas_col]) / 2
        with open(fname, "w") as f:
            for hl in header_lines:
                f.write(hl + "\r\n")
            f.write(data.to_string(header=False, index=False))
            f.write("\r\n" + 4 * "0\r\n")

    def plot_together(self, fname, plot_columns, valid_column="valid", outdir="."):
        groupby_df = self.data.groupby(self.data[valid_column])
        try:
            group_valid = groupby_df.get_group(True)
        except KeyError:
            some_valid = False
        else:
            some_valid = True
        try:
            group_invalid = groupby_df.get_group(False)
        except KeyError:
            some_invalid = False
        else:
            some_invalid = True
        for col in plot_columns:
            if col not in self.data.columns:
                continue
            fig, ax = plt.subplots()
            if some_valid:
                nmeas_valid = group_valid["meas"].to_numpy()
                ax.plot(nmeas_valid, group_valid[col].to_numpy(), "o", color="b", markersize=4)
            if some_invalid:
                nmeas_invalid = group_invalid["meas"].to_numpy()
                ax.plot(nmeas_invalid, group_invalid[col].to_numpy(), "o", color="r", markersize=4)
            scale, vstdmedian, vskewness = find_best_yscale(self.data[col])
            if scale == "symlog":
                threshold = find_threshold_minnonzero(self.data[col])
                plt.minorticks_on()
                ax.set_yscale("symlog", linscale=0.2, linthresh=threshold, base=10)
                ax.yaxis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=threshold))
                ax.yaxis.set_minor_locator(MinorSymLogLocator(threshold))
            else:
                ax.set_yscale(scale)
                plt.minorticks_on()
            ax.grid(which="both", axis="both")
            plt.ylabel(col)
            plt.xlabel("measurement num")
            plt.tight_layout()
            fig_dirfname = output_file(fname, new_ext="_" + col + ".png", directory=outdir)
            print(fig_dirfname)
            plt.savefig(fig_dirfname, dpi=80)
            plt.close()
