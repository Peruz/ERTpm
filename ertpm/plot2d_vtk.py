#!/usr/bin/env python

import argparse
import pandas as pd
import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt


def general_get_cmd(dict_args={}):

    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    main = parse.add_argument_group("main")
    option = parse.add_argument_group("option")
    output = parse.add_argument_group("output")

    main.add_argument("-fname", type=str, help="vtk file to plot", nargs="+")
    main.add_argument("-bounds", type=float, help="(xmin, xmax, ymin, ymax, zmin, zmax)", nargs="+")
    main.add_argument("-outdir", type=str, help="data file for difference")
    main.add_argument("-rho", type=str, default="res", help="vtk resistivity name")
    main.add_argument("-cov", type=str, default="cov", help="coverage vtk name")

    option.add_argument("-elec", type=str, help="elec file", default=None)
    option.add_argument("-dHow", type=str, help="how to perform difference", default="pct")
    option.add_argument("-Cmin", type=float, help="min colorbar", default=0)
    option.add_argument("-Cmax", type=float, help="max colorbar", default=2200)
    option.add_argument("-Cgrid", type=str, help="grid color", default=None)
    option.add_argument("-Cedge", type=str, help="edge color", default="darkgrey")
    option.add_argument("-Cmap", type=str, help="colormap name", default="turbo")
    option.add_argument(
        "-clipping",
        help="clipping method",
        default="cov",
        choices=["cov", "bounds", None],
    )
    option.add_argument("-ticks", type=str, help="figure ticks", default="outside")
    option.add_argument("-notes", help="additional notes")
    option.add_argument("-Fsize", type=float, help="float size", default=14)
    option.add_argument("-min_cov", type=float, help="minimum coverage", default=-3)
    output.add_argument("-dir_vista", type=str, help="output directory", default="vista")

    args = vars(parse.parse_args())
    print("ciao")

    if isinstance(args['fname'], str):
        args["fname"] = [args['fname']]

    for key, val in dict_args.items():
        if not hasattr(args, key):
            raise AttributeError("unrecognized option: ", key)
        else:
            setattr(args, key, val)

    return args


def prepare_rho(mesh, rho, outdir, dHow):
    if outdir is not None:
        mesh_diff = pv.read(outdir)
        if dHow == "val":
            mesh[rho] = mesh[rho] - mesh_diff[rho]
        if dHow == "pct":
            mesh[rho] = mesh[rho] / mesh_diff[rho]
    return mesh


def prepare_cov(mesh, covLabel, min_cov=-2, max_cov=-1):
    """in bert the coverage is the log of the absolute values of the jacobian matrix column
    here we limit the values between a minimum and 1,
    this ensures a nice transition and shape, and compatibility with the plotting opacity argument
    """
    cov = mesh[covLabel]
    cov[cov > max_cov] = max_cov
    cov[cov < min_cov] = min_cov
    cov = (cov - min_cov) / (max_cov - min_cov)
    # cov_np[cov_np > max_cov] = 1
    # cov_np[cov_np < min_cov] = 0
    mesh[covLabel] = cov
    return mesh


def _plot_(f, args):
    mesh = pv.read(f)
    mesh = prepare_rho(mesh=mesh, rho=args['rho'], outdir=args['outdir'], dHow=args['dHow'])
    mesh = prepare_cov(mesh=mesh, covLabel=args['cov'], min_cov=args['min_cov'])
    mesh.save('changed.vtk')
    sba = dict(
        title="resistivity ohm m\n",
        width=0.5,
        position_x=0.25,
        height=0.1,
        position_y=0.025,
    )
    plotter = pv.Plotter(window_size=(1300, 800))
    # select region
    # if args['clipping'] == "cov":
    #     mesh = mesh.threshold(args['min_cov'], args['cov'], invert=False)
    # elif args['clipping'] == "bounds":
    mybounds = (0, 80, -40, 1, -40, 1)
    # cell_centers = mesh.cell_centers().points
    # inside = np.where(cell_centers[:, 2] > -4)
    # mesh = mesh.extract_cells(inside)
    mesh = mesh.clip_box(mybounds, invert=False)

    # camera
    length = mesh.GetLength()
    bounds = mesh.GetBounds()
    print(bounds)
    xc = (bounds[1] + bounds[0]) / 2
    # yc = (bounds[3] + bounds[2]) * 2.5 / 4
    yc = (bounds[3] + bounds[2]) / 2
    cam = [[xc, yc, length], [xc, yc, 0], [0, 1, 0]]
    plotter.camera_position = cam

    # mesh and data
    _ = plotter.add_mesh(
        mesh,
        scalars=args['rho'],
        opacity=args['cov'],
        show_edges=False,
        edge_color="k",
        show_scalar_bar=True,
        scalar_bar_args=sba,
        cmap=args['Cmap'],
        clim=(args['Cmin'], args['Cmax']),
    )
    _ = plotter.show_bounds(
        mesh=mesh,
        bounds=mybounds,
        grid=args['Cgrid'],
        location="outer",
        ticks=args['ticks'],
        font_size=12,
        font_family="times",
        # padding=0,
        # xlabel="m",
        # ylabel="m",
        show_zaxis=False,
        show_zlabels=False,
    )

    length = mesh.GetLength()
    bounds = mesh.GetBounds()
    print(bounds)
    xc = ((bounds[1] + bounds[0]) / 2) - (bounds[1] + bounds[0]) / 100 * 5
    # yc = (bounds[3] + bounds[2]) * 2.5 / 4
    yc = (bounds[3] + bounds[2]) / 2
    cam = [[xc, yc, length], [xc, yc, 0], [0, 1, 0]]
    plotter.camera_position = cam

    # electrodes
    if args['elec'] is not None:
        electrodes = pd.read_csv(args['elec'], usecols=['x', 'y', 'z'])
        electrodes = pv.PolyData(electrodes[['x', 'z', 'y']].to_numpy())
        electrodes_labels = ["{}".format(i + 1) for i in range(electrodes.n_points)]
        electrodes["elec_num"] = electrodes_labels
        plotter.add_point_labels(electrodes, "elec_num", font_size=14, point_size=5, shape=None, point_color="k")
        # add optional notes if present
        if args['notes'] is not None:
            for k, v in args['notes'].items():
                coordinates = pv.PolyData(np.array(v))
                coordinates["name"] = [k]
                plotter.add_point_labels(
                    coordinates,
                    "name",
                    font_size=14,
                    point_size=5,
                    shape=None,
                    point_color="k",
                )
    fpng = f.replace(".vtk", ".png")
    plotter.show(screenshot=fpng)
    return fpng


def main():
    args = general_get_cmd()

    if args["fname"] is None:
        raise ValueError("missing fname argument, no file to process?")

    pv.set_plot_theme("document")

    for f in args['fname']:
        print("\n", "-" * 80, "\n", f)
        print(f)
        _ = _plot_(f, args)


if __name__ == "__main__":
    main()
