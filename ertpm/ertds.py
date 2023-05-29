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

import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator
from ertpm.ertutils import MinorSymLogLocator
from ertpm.ertutils import find_threshold_minnonzero
from ertpm.ertutils import find_best_yscale
from ertpm.ertutils import output_file


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

    def __str__(self):
        ertds_string = "\n".join(
            (
                "--- --- ---",
                "meta",
                "\n".join("{}\t{}".format(k, v) for k, v in sorted(self.meta.items(), key=lambda t: str(t[0]))),
                "elec",
                self.elec.to_string(max_rows=1e5),
                "data",
                self.data.to_string(max_rows=1e5),
                "--- --- ---",
            )
        )
        return ertds_string

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
                "len k {kl} != len data {dl}; possibly the wrong k are used or the data are incomplete".format(
                    dl=len(self.data), kl=len(data_k)
                )
            )
            abmn = ["a", "b", "m", "n"]
            abmnk = ["a", "b", "m", "n", "k"]
            self.data = self.data.merge(data_k[abmnk], on=abmn, how="left", suffixes=("", "_"))
            self.data["k"] = self.data["k_"]
            self.data.drop(columns="k_", inplace=True)
        elif len(data_k) < len(self.data):
            raise IndexError("len k {kl} < len data {dl}; possibly the wrong k file".format(dl=len(self.data), kl=len(data_k)))

    def calc_1d_k(self, coord="x"):
        map_list = ["a", "b", "m", "n"]
        elec_dict = self.elec.set_index("num").to_dict()[coord]
        data_coord = self.data[map_list]
        for column in map_list:
            data_coord[column] = data_coord[column].map(elec_dict)
        data_coord = data_coord.to_numpy()
        k = (
            2
            * 3.14
            * (
                (1 / abs(data_coord[:, 0] - data_coord[:, 2]))
                - (1 / abs(data_coord[:, 0] - data_coord[:, 3]))
                - (1 / abs(data_coord[:, 1] - data_coord[:, 2]))
                + (1 / abs(data_coord[:, 1] - data_coord[:, 3]))
            )
            ** -1
        )
        self.data["k"] = k

    def calc_k(self):
        """
        analytical solution,
        valid only for a flat-earth model
        with surface electrodes.
        If all the coordiantes are not null, it raises an error.
        If 2 or 1 coordinates are not null, it uses those as surface coordinates.
        """
        map_list = ["a", "b", "m", "n"]
        elec_dict = self.elec.set_index("num").to_dict()["x", "y", "z"]
        data_coord = self.data[map_list]
        for column in map_list:
            data_coord[column] = data_coord[column].map(elec_dict)
        data_coord = data_coord.to_numpy()
        k = (
            2
            * 3.14
            * (
                (1 / abs(data_coord[:, 0] - data_coord[:, 2]))
                - (1 / abs(data_coord[:, 0] - data_coord[:, 3]))
                - (1 / abs(data_coord[:, 1] - data_coord[:, 2]))
                + (1 / abs(data_coord[:, 1] - data_coord[:, 3]))
            )
            ** -1
        )
        self.data["k"] = k

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

    def to_bert(self, fname, w_ip, w_err):
        elec_cols = ["x", "y", "z"]
        data_cols = ["a", "b", "m", "n"]
        meas_cols = ["r", "rhoa", "k"]
        if w_ip:
            meas_cols.append("ip")
        if w_err:
            meas_cols.append("err")
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
