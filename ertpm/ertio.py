import os
import warnings
import numpy as np
import pandas as pd
from ertpm.ertds import ERTdataset
from IPython import embed


class IncompleteFile(Exception):
    def __init__(self, fname=None, error_message=None):
        self.fname = fname
        self.error_message = "Incomplete file"
        if error_message is not None:
            self.error_message = error_message


def output_file(old_fname, new_ext=".dat", directory="."):
    """return name for the output file and clean them if already exist"""
    f, old_ext = os.path.splitext(old_fname)
    new_fname = f + new_ext
    new_dfname = os.path.join(directory, new_fname)
    if not os.path.isdir(directory):
        os.mkdir(directory)
    elif os.path.exists(new_dfname):
        os.remove(new_dfname)
    return new_dfname


def read_syscal(f):
    """
    Assuming only x coordinate is used
    """
    colNames = ("a", "b", "m", "n", "rhoa", "stk", "v", "curr")
    colIndexes = (2, 3, 4, 5, 6, 7, 10, 11)
    colTypes = (float, float, float, float, float, float, float)
    colTypes = dict(zip(colNames, colTypes))
    data = pd.read_csv(
        f,
        header=None,
        index_col=False,
        skiprows=1,
        usecols=colIndexes,
        names=colNames,
        dtype=colTypes,
    )
    # handle obseerved issues
    if any(data["curr"] == 0):
        print(data[data["curr"] == 0])
        data = data[data["curr"] != 0]
        warnings.warn("found and removed zero current values")
        data = data.reset_index(drop=True)

    data["r"] = data["v"] / data["curr"]
    data.insert(0, "meas", np.arange(1, len(data) + 1))
    # elec - define a basic geometry, to be updated anyway with the real geometry from file
    elecX = pd.concat(
        (data.a, data.b, data.m, data.n),
        axis=0,
        ignore_index=True,
    ).unique()
    elecX = np.sort(elecX)
    elecZeros = np.zeros_like(elecX)
    elecNums = np.arange(1, len(elecX) + 1, dtype=np.int16)
    elec = pd.DataFrame(data={"num": elecNums, "x": elecX, "y": elecZeros, "z": elecZeros})
    # data - remap elec coordinates to number
    map_list = ["a", "b", "m", "n"]
    elec_dict = elec.set_index("x").to_dict()["num"]
    for column in map_list:
        data[column] = data[column].map(elec_dict)
    # set first elec to 0
    elec["x"] = elec["x"] - min(elec["x"])
    meta = {}
    meta["meas_tot"] = len(data)
    ertds = ERTdataset(data=data, elec=elec, meta=meta)
    return ertds


def read_bert(f):
    with open(f) as fid:
        for i, l in enumerate(fid):
            if i == 0:
                elecCnt = int(l.strip().split('#', 1)[0])
            if i == 1:
                elecHeaderRaw = l.strip().replace('#', '').split(' ')
                elecHeader = [c for c in elecHeaderRaw if c != ' ' and c != '']
            if i == elecCnt + 2:
                dataCnt = int(l.strip().split('#', 1)[0])
            if i == elecCnt + 3:
                dataHeaderRaw = l.strip().replace('#', '').split(' ')
                dataHeader = [c for c in dataHeaderRaw if c != ' ' and c != '']
        expectedLineCnt = elecCnt + dataCnt + 4
        if i != expectedLineCnt - 1:
            warnings.warn('wrong number of lines')
    elec = pd.read_csv(
        f,
        skiprows=2,
        nrows=elecCnt,
        sep=r"\s+|,",
        skipinitialspace=True,
        index_col=None,
        engine='python',
        header=None,
        names=elecHeader,
    )
    elec.rename(columns={'#x': 'x'}, inplace=True)
    data = pd.read_csv(
        f,
        skiprows=elecCnt + 4,
        nrows=dataCnt,
        sep=r"\s+|,",
        skipinitialspace=True,
        index_col=None,
        engine='python',
        header=None,
        names=dataHeader,
    )
    data.rename(columns={'#a': 'a'}, inplace=True)
    data.rename(columns={'rho': 'r'}, inplace=True)
    data = data.astype({
        'a': np.int16,
        'b': np.int16,
        'm': np.int16,
        'n': np.int16,
    })
    ertds = ERTdataset(data=data, elec=elec, meta=None)
    embed()
    return ertds




def read_labrecque(f):
    """
    Read a labrecque data file an return data and electrode dataframes.
    """
    FreDom = False
    AppRes = False
    TW = 0
    SP = 0
    with open(f) as fid:
        enumerated_lines = enumerate(fid)
        i, l = next(enumerated_lines)
        while "elec_start" not in l:
            if "FStcks" in l:
                FreDom = True
            elif "#SAprs" in l:
                AppRes = True
            elif "#TW" in l:
                TW += 2  # for each window, adds IP Window n and its associated Std
            elif l.startswith("#SCltSP"):
                SP = 1
            i, l = next(enumerated_lines)
        es = i + 1
        while "elec_end" not in l:
            i, l = next(enumerated_lines)
        ee = i - 1
        while "data_start" not in l:
            i, l = next(enumerated_lines)
        ds = i + 3
        while "data_end" not in l:
            try:
                i, l = next(enumerated_lines)
            except StopIteration:
                raise IncompleteFile("no data_end")
        de = i
    print("FreDom: ", FreDom, "    AppRes: ", AppRes, "    TW: ", TW, "    SP: ", SP)
    # DATA without using header because it is too inconsistent
    col_name_num = {"meas": 0, "a": 2, "b": 4, "m": 6, "n": 8}
    col_name_num.update({"r": 9, "stk": 10, "v": 11, "curr": 13 + TW + SP, "ctc": 14 + TW + SP, "datetime": 15 + TW + SP})
    if (FreDom) and (not AppRes):
        col_name_num.update({"r": 9, "ip": 10, "v": 13, "stk": 14, "curr": 17, "ctc": 20, "datetime": 21 + SP})
    elif (not FreDom) and (AppRes):
        col_name_num.update({"r": 10, "stk": 11, "v": 12, "curr": 14 + TW + SP, "ctc": 15 + TW + SP, "datetime": 16 + TW + SP})
    elif (FreDom) and (AppRes):
        col_name_num.update({"r": 10, "ip": 11, "v": 14, "stk": 15, "curr": 18, "ctc": 21, "datetime": 22 + SP})
    col_names_dtypes = {
        "meas": "Int16",
        "a": "Int16",
        "b": "Int16",
        "m": "Int16",
        "n": "Int16",
        "r": float,
        "ip": float,
        "v": float,
        "curr": float,
        "ctc": float,
        "stk": float,
        "datetime": "datetime64[ns]",
    }
    (col_nums, col_names, col_dtypes) = zip(*[(v, k, col_names_dtypes[k]) for k, v in sorted(col_name_num.items(), key=lambda item: item[1])])
    dn = de - ds
    sep = r"\s+|,"
    error_strings = ["*", "TX", "Resist.", "out", "of", "range", "Error_Zero_Current", "Raw_Voltages:", "Run", "Complete"]
    print(col_name_num)
    data = pd.read_csv(
        f,
        header=None,
        index_col=False,
        nrows=dn,
        skiprows=ds,
        usecols=col_nums,
        names=col_names,
        dtype=col_names_dtypes,
        na_values=error_strings,
        parse_dates=["datetime"],
        date_parser=lambda d: pd.to_datetime(d, format="%Y%m%d_%H%M%S", errors="coerce"),
        sep=sep,
        engine="python",
        on_bad_lines="warn",
        # comment='*',
    )
    # print('data set:\n{}'.format(data))
    invalid_data = data.loc[data["r"].isna()]
    if not invalid_data.empty:
        print("\n!!! found invalid data\n", invalid_data)
        data = data.drop(index=invalid_data.index)
        data = data.reset_index()
        print("data set:\n{}".format(data))
    if not FreDom:
        data["ip"] = 1
    data["datetime"] = pd.to_datetime(data["datetime"], format="%Y%m%d_%H%M%S")
    data = data.astype(col_names_dtypes)
    data["stk"] = data["stk"] / np.abs(data["v"]) * 100
    data["ctc"] = data["ctc"] / 1e3  # convert ctc from ohm to kohm
    # elec using headers
    ec = {"El#": "num", "Elec-X": "x", "Elec-Y": "y", "Elec-Z": "z"}
    et = {"num": "Int16", "x": float, "y": float, "z": float}
    en = ee - es
    elec = pd.read_csv(
        f,
        skiprows=es,
        usecols=list(ec),
        nrows=en,
        header=0,
        sep=r",|\s+",
        index_col=False,
        engine="python",
    )
    elec = elec.rename(columns=ec)
    elec = elec.astype(et)
    # print('electrodes:\n{}'.format(elec))
    meta = {}
    meta["meas_tot"] = len(data)
    ertds = ERTdataset(data=data, elec=elec, meta=meta)
    return ertds


def read_electra_custom_complete(f):
    """
    read the COMPLETE output format
    return an ERTds with meta, elec, and data
    """
    print("reading custom ele file, complete format")
    meta = {}

    with open(f) as fid:
        enumerated_lines = enumerate(fid)

        for i, l in enumerated_lines:
            l = l.strip()

            if not l.startswith("#"):
                # we first need to find the parameter name, continue
                continue
            else:
                # remove #
                l = l[1:]
                # make sure only space is used as separator
                l = l.replace("\t", " ")

            if "Creation date" == l:
                i, lv = next(enumerated_lines)
                lv = lv.strip()
                try:
                    meta["datetime"] = pd.to_datetime(l, format="%Y-%m-%d %H:%M:%S")
                except ValueError:
                    meta["datetime"] = l
                    warnings.warn("could not covert datetime")
                continue

            elif "Configuration name" == l:
                i, lv = next(enumerated_lines)
                lv = lv.strip()
                meta["configuration"] = lv
                continue

            elif "Total samples time on" == l:
                i, lv = next(enumerated_lines)
                lv = lv.strip()
                meta["sampling"] = int(lv)
                continue

            elif "Site profile" == l:
                i, l = next(enumerated_lines)
                lv = lv.strip()
                meta["site"] = lv
                continue

            elif "Record profile" == l:
                i, lv = next(enumerated_lines)
                lv = lv.strip()
                meta["acquisition"] = lv
                continue

            elif "Frequency value [Hz]" == l:
                i, lv = next(enumerated_lines)
                lv = lv.strip()
                meta["freq"] = float(lv)
                continue

            elif "Current value [mA]" == l:
                i, lv = next(enumerated_lines)
                lv = lv.strip()
                meta["tgt_current"] = float(lv)
                continue

            elif "Total electrodes" == l:
                i, lv = next(enumerated_lines)
                lv = lv.strip()
                num_elec = int(lv)
                continue

            elif "X Y Z" in l:
                elec_header_line = i
                continue

            elif "Meas." in l:
                data_header_line = i
                break

        if "Duration" not in meta:
            meta["curr_dur"] = 1

        # elec
        elec = pd.read_csv(
            f,
            usecols=[0, 1, 2],
            names=["x", "y", "z"],
            skiprows=elec_header_line + 1,
            nrows=num_elec,
            sep="\t",
            header=None,
        )
        elec.insert(0, "num", np.arange(1, num_elec + 1))

        # data
        data = pd.read_csv(
            f,
            skiprows=data_header_line,
            sep="\t",
            header=[0, 1],
        )

        data.columns = data.columns.droplevel(1)
        data.drop(data.filter(regex="Unname"), axis=1, inplace=True)

        if "k" not in data.columns and "K/(2pi)" in data.columns:
            data.insert(8, "k", data["K/(2pi)"] * 2 * np.pi)

        data_header_map = {
            "A": "a",
            "B": "b",
            "M": "m",
            "N": "n",
            "#Meas.": "meas",
            "I(AB)": "curr",
            "Z(AB)": "ctc",
            "V(MN)": "M_m_v",
            "Rhoa": "M_m_rhoa",
            "R": "M_m_r",
        }

        data = data.rename(columns=data_header_map)

        meta["meas_tot"] = len(data)

        # sinusoids
        sinusoid_headers = ["VMN{:03d}".format(i) for i in range(meta["sampling"])]
        sinusoids = data[sinusoid_headers]
        data = data.drop(sinusoids, axis=1)

        if "M_m_r" not in data.columns and "v" in data.columns and "curr" in data.columns:
            data.insert(9, "M_m_r", data["v"] / data["curr"])

        ertds = ERTdataset(data=data, elec=elec, meta=meta)
        ertds.sinusoids = sinusoids

        return ertds




def read_res2dinv_gen(f):
    # read info lines
    lines_descriptions = {0: "name", 1: "spacing", 5: "type", 6: "num_meas"}
    file_dict = {}
    with open(f) as fin:
        for fin_ind, fin_line in enumerate(fin):
            fin_line = fin_line.strip()
            if fin_ind in lines_descriptions.keys():
                file_dict[lines_descriptions[fin_ind]] = fin_line
            if fin_ind == 9:
                break
    file_dict["spacing"] = float(file_dict["spacing"])
    file_dict["type"] = int(file_dict["type"])
    file_dict["num_meas"] = int(file_dict["num_meas"])

    # read data
    data_skiprows = 9
    data = pd.read_csv(
        f,
        header=None,
        delim_whitespace=True,
        skiprows=data_skiprows,
        nrows=file_dict["num_meas"],
        usecols=[1, 3, 5, 7, 9],
        names=["a", "b", "m", "n", "rhoa"],
    )
    # find unique coordinates
    unique_a = data["a"].unique()
    unique_b = data["b"].unique()
    unique_m = data["m"].unique()
    unique_n = data["n"].unique()
    unique = sorted(set([*unique_a, *unique_b, *unique_m, *unique_n]))
    # find number of unique coordinates and thus electrodes
    num_unique = len(unique)
    # init elec df
    elec_num = np.arange(1, num_unique + 1, dtype=np.int16)
    elec_x = np.array(unique)
    zeros = np.zeros_like(elec_x)
    elec = pd.DataFrame(data={"num": elec_num, "x": elec_x, "y": zeros, "z": zeros})
    # remap data based on elec numbering and coordinates
    elec_dict = elec.set_index("x").to_dict()["num"]
    map_list = ["a", "b", "m", "n"]
    for column in map_list:
        data[column] = data[column].map(elec_dict)
    ertds = ERTdataset(data=data, elec=elec)
    return ertds
