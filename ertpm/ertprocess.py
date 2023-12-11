#!/usr/bin/env python

"""
ERT PROCESSING
it gets arguments from cmd-line and/or dictionary (e.g., ert manager script)
it uses a ERT processing class that delegates to two dataframes for data and elec tables
it filters the data based on args; set very loose args values to avoid filtering.
"""

import os
import datetime
import argparse
import pandas as pd
import numpy as np
from ertpm.ertio import read_terra
from ertpm.ertio import read_terrabis
from ertpm.ertio import read_syscal
from ertpm.ertio import read_bert
from ertpm.ertio import read_labrecque
from ertpm.ertio import read_electra_custom_complete
from ertpm.ertutils import output_file


def general_get_cmd():

    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    main = parse.add_argument_group("main")
    filters = parse.add_argument_group("filters")
    adjustments = parse.add_argument_group("adjustments")
    err = parse.add_argument_group("output")
    outputs = parse.add_argument_group("output")

    main.add_argument("-fname", type=str, help="data file to process", nargs="+")
    main.add_argument("-ftype", type=str, help="data type to process", default="syscal", choices=["syscal", "electra", "labrecque", "bert", "terra", "terrabis"])
    main.add_argument("-outdir", type=str, help="output directory", default="processing")

    filters.add_argument("-timelapse", action="store_true", default=False, help="keep all the measurements, assign very high error rather than removing")

    filters.add_argument("-v_check", action="store_true", default=False, help="activate v analysis")
    filters.add_argument("-v_min", type=float, default=1e-5, help="min voltage, V")

    filters.add_argument("-ctc_check", action="store_true", default=False, help="activate ctc analysis")
    filters.add_argument("-ctc_max", type=float, default=1e4, help="max ctc, ohm")

    filters.add_argument("-stk_check", action="store_true", default=False, help="activate stk analysis")
    filters.add_argument("-stk_max", type=float, default=5, help="max stacking err, pct")

    filters.add_argument("-rec_check", action="store_true", default=False, help="activate reciprocal analysis")
    filters.add_argument("-rec_max", type=float, default=10, help="max reciprocal err, pct")
    filters.add_argument(
        "-rec_quantities",
        type=str,
        default=["r"],
        choices=["r", "rhoa"],
        help="quantities for reciprocal err, mean if more than 1",
        nargs="+",
    )
    filters.add_argument("-rec_couple", action="store_true", default=True, help="couple reciprocals")
    filters.add_argument("-rec_unpaired", action="store_true", default=False, help="keep unpaired")

    filters.add_argument("-k_check", action="store_true", default=False, help="activate k analysis")
    filters.add_argument("-k_max", type=float, default=None, help="max geometrical factor, m")
    filters.add_argument("-k_file", type=str, help="file with geom factors and activates k and rhoa; self to keep; 2d or 3d to calc")

    filters.add_argument("-rhoa_check", action="store_true", default=False, help="activate rhoa analysis")
    filters.add_argument("-rhoa_min", default=1, type=float, help="rhoa min")
    filters.add_argument("-rhoa_max", default=5000, type=float, help="rhoa max")

    filters.add_argument("-elecbad_check", action="store_true", default=True, help="activate elecbad analysis")
    filters.add_argument("-elecbad", type=int, help="electrodes to remove", nargs="+")

    err.add_argument("-err", default=5, type=float, help="add this value as a base error")
    err.add_argument("-err_rec", action="store_true", default=False, help="if true, add reciprocal error")
    err.add_argument("-err_stk", action="store_true", help="if true, add stacking error")

    outputs.add_argument("-w_err", action="store_true", default=True, help="write error (base, rec, stk)")
    outputs.add_argument("-w_rhoa", action="store_true", help="if true, write rhoa")
    outputs.add_argument("-w_k", action="store_true", help="if true, write k")
    outputs.add_argument("-w_ip", action="store_true", help="if true, write phase")
    outputs.add_argument("-plot", action="store_true", help="plot data quality figures")
    outputs.add_argument("-cross", action="store_true", help="report on the cross validity of the filters")
    outputs.add_argument(
        "-export",
        type=str,
        default="csv",
        choices=["pygimli", "res2dinv", "simpeg", "csv"],
        help="list of export formats",
        nargs="+",
    )

    adjustments.add_argument("-s_abmn", type=int, default=0, help="shift abmn")
    adjustments.add_argument("-s_meas", type=int, default=0, help="shift measurement number")
    adjustments.add_argument("-s_elec", type=int, default=0, help="shift electrode number")
    adjustments.add_argument("-f_elec", type=str, default=None, help="electrode coordinates:num,x,y,z with header")

    args = parse.parse_args()

    # make suere a list is passed even when a single file is given
    if isinstance(args.fname, str):
        args["fname"] = [args.fname]

    return args


def process(args):
    print("processing")

    for f in args["fname"]:
        print(f)

        if args["ftype"] == "electra":
            ds = read_electra_custom_complete(f, methods=["fft"])

        elif args["ftype"] == "labrecque":
            ds = read_labrecque(f)

        elif args["ftype"] == "bert":
            ds = read_bert(f)

        elif args["ftype"] == "syscal":
            ds = read_syscal(f)

        elif args["ftype"] == "terra":
            ds = read_terra(f)
        elif args["ftype"] == "terrabis":
            ds = read_terrabis(f)

        else:
            raise ValueError(args["ftype"], " have yet to be implemented")

        if args['f_elec'] is not None:
            elec_coords = pd.read_csv(args['f_elec'])
            ds.elec = elec_coords
            print('set elec coordtinates to: ', ds.elec)

        file_path, file_name = os.path.split(f)
        ds.meta["file_name"] = file_name
        ds.meta["file_path"] = file_path
        ds.meta["file_type"] = args["ftype"]
        ds.meta["processing_datetime"] = datetime.datetime.now().isoformat(timespec="seconds")

        # k
        if args["k_file"]:
            if args["k_file"] in ['1d', '2d', '3d']:
                print('calculating geometric factors')
                ds.calc_k_3d()
            else:
                k_vec = pd.read_csv(args["k_file"], index_col=None)
                ds.set_k(k_vec)
            ds.data["rhoa"] = ds.data["k"] * ds.data["r"]  # update rhoa with the new k
        if args["k_check"]:
            ds.data["k_valid"] = ds.data["k"] < float(args["k_max"])

        # rec
        if args["rec_check"]:
            ds.rec_process(rec_quantities=args["rec_quantities"], rec_max=args['rec_max'])

        # bad elecs
        if args["elecbad_check"]:
            print("removing the data associated with these electrodes: ", args["elecbad"])
            ds.data["elec_valid"] = ~np.isin(ds.data[["a", "b", "m", "n"]], args["elecbad"]).any(axis=1)

        # ctc
        if args["ctc_check"] and not any(ds.data["ctc"].isnull()):
            ds.data["ctc_valid"] = ds.data["ctc"] < float(args["ctc_max"])

        # stk
        if args["stk_check"]:
            ds.data["stk_valid"] = ds.data["stk"] < float(args["stk_max"])

        # rhoa
        if args["rhoa_check"]:
            if any(ds.data["rhoa"].isnull()):
                Warning("no rhoa data, cannot check rhoa range. Setting rhoa_valid to True")
                ds.data["rhoa_valid"] = True
            else:
                ds.data["rhoa_valid"] = ds.data["rhoa"].between(args["rhoa_min"], args["rhoa_max"])

        # combine filters
        filters = ["rec_valid", "k_valid", "rhoa_valid", "ctc_valid", "elec_valid"]
        ds.data["valid"] = ds.data[filters].all(axis="columns")
        ds.default_types()

        # err estimate
        if "err" in ds.data.columns:
            ds.data.drop("err", axis=1, inplace=True)
        ds.data.insert(8, "err", 0.0)
        ds.data["err"] += float(args["err"])
        if args["err_rec"]:
            ds.data['err'] += ds.data['rec_err']
            # use 0.75 quantile when no rec_err is present (unpaired)
            expected_rec_err = np.zeros(len(ds.data))
            missings_rec_err = ds.data['rec_err'] == 0
            expected_rec_err[missings_rec_err] = ds.data["rec_err"].quantile(0.75)
            ds.data["err"] += expected_rec_err
        if args["err_stk"]:
            ds.data["err"] += ds.data["stk"]
        if args["w_err"]:
            if any(ds.data["err"].isnull()):
                raise ValueError("err column contains null values")
            if any(ds.data["err"] == 0):
                raise ValueError("err column contains 0s, which may hinder the inversion")

        # metadata of the analysis
        if "meas_tot" not in ds.meta:
            ds.meta["meas_tot"] = len(ds.data)
        ds.meta["meas_out"] = ds.meta["meas_tot"] - sum(ds.data["valid"])
        ds.meta["meas_in"] = sum(ds.data["valid"])
        ds.meta["by_rec"] = ds.meta["meas_tot"] - sum(ds.data["rec_valid"])
        ds.meta["by_rhoa"] = ds.meta["meas_tot"] - sum(ds.data["rhoa_valid"])
        ds.meta["by_k"] = ds.meta["meas_tot"] - sum(ds.data["k_valid"])
        ds.meta["by_ctc"] = ds.meta["meas_tot"] - sum(ds.data["ctc_valid"])
        ds.meta["by_elecbad"] = ds.meta["meas_tot"] - sum(ds.data["elec_valid"])

        # BEFORE COUPLING
        if "csv" in args["export"]:
            fcsv = output_file(f, new_ext=".csv", directory=args["outdir"])
            ds.to_csv(fcsv)

        if args["plot"]:
            plot_columns = []
            # general
            if "err" in ds.data.columns:
                plot_columns.append("err")
            if "rhoa" in ds.data.columns:
                plot_columns.append("rhoa")
            if "r" in ds.data.columns:
                plot_columns.append("r")
            if args["rec_check"]:
                plot_columns.append("rec_err")
            if args["ctc_check"]:
                plot_columns.append("ctc")
            if args["stk_check"]:
                plot_columns.append("stk")
            if args["v_check"]:
                plot_columns.append("v")
            if args["k_check"]:
                plot_columns.append("k")
            plot_columns.append("curr")

            # ELECTRA
            # plot avaialble methods
            if "fft_rhoa" in ds.data.columns:
                plot_columns.append("fft_rhoa")
            elif "fft_r" in ds.data.columns:
                plot_columns.append("fft_r")
            if "fit_rhoa" in ds.data.columns:
                plot_columns.append("fit_rhoa")
            elif "fit_r" in ds.data.columns:
                plot_columns.append("fit_r")
            if "M_m_rhoa" in ds.data.columns:
                plot_columns.append("M_m_rhoa")
            elif "M_m_r" in ds.data.columns:
                plot_columns.append("M_m_r")
            # compare methods
            if "M_m_r" in ds.data.columns and "fft_r" in ds.data.columns:
                ds.data["M_m_r_over_fft_r"] = ds.data["M_m_r"] / ds.data["fft_r"]
                plot_columns.append("M_m_r_over_fft_r")
            if "M_m_r" in ds.data.columns and "fit_r" in ds.data.columns:
                ds.data["M_m_r_over_fit_r"] = ds.data["M_m_r"] / ds.data["fit_r"]
                plot_columns.append("M_m_r_over_fit_r")
            if "fft_r" in ds.data.columns and "fit_r" in ds.data.columns:
                ds.data["fft_r_over_fit_r"] = ds.data["fft_r"] / ds.data["fit_r"]
                plot_columns.append("fft_r_over_fit_r")

            # plot with flexible scale
            ds.plot_together(f, plot_columns, valid_column="valid", outdir=args["outdir"])


        # skip if no check was done because there are no information
        if args["rec_check"]:
            # rec coupling, should stay between error analysis and export
            # 1 is the flag for direct and 0 for unpaired
            if args["rec_couple"]:
                ds.rec_couple(keep_unpaired=args['rec_unpaired'])


        if "simpeg" in args["export"]:
            fubc = output_file(f, new_ext=".xyz", directory=args["outdir"])
            ds.to_ubc_xyz(fubc, meas_col="r", w_err=args["w_err"])
        if "res2dinv" in args["export"]:
            finv = output_file(f, new_ext=".dat", directory=args["outdir"])
            ds.to_res2dinv(finv, meas_col="rhoa", w_err=args["w_err"])
        if "pygimli" in args["export"]:
            finv = output_file(f, new_ext="_pygimli.dat", directory=args["outdir"])
            ds.to_bert(finv, w_err=args["w_err"], w_ip=args["w_ip"], w_rhoa=args['w_rhoa'], w_k=args['w_k'])

        # yield the ertds in case it is needed
        # or to confimir that the processing is done
        yield ds


def main():
    args = vars(general_get_cmd())

    if args["fname"] is None:
        raise ValueError("missing fname argument, no file to process?")

    for ds in process(args):
        pass


if __name__ == "__main__":
    main()
