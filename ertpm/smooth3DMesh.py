import gmsh
import numpy as np
import sys
import math
from scipy.spatial import Delaunay
from IPython import embed
import pandas as pd
import matplotlib.pyplot as plt
import duckdb as db
from scipy.interpolate import griddata

gmsh.initialize()

gmsh.model.add("lp")

model = gmsh.model
geo = model.geo
mesh = model.mesh

cl = 0.05

topo_df = pd.read_csv('electrodesUpdated_topography.csv')
elecs_df = pd.read_csv('electrodesUpdated_resipy.csv')

edges = np.array((
    (-2, 3, 1.5),
    (-2, -2, -1),
    (3, -2, -1),
    (3, 3, 1.3)
))

edgesBottom = edges.copy()
edgesBottom[:, 2] -= 6

edges_df = pd.DataFrame(np.array(edges), columns=['x', 'y', 'z'])
edgesBottom_df = pd.DataFrame(np.array(edgesBottom), columns=['x', 'y', 'z'])

surf_df = db.sql(
    """
    select x, y, z, 1 as flag from elecs_df where buried = 0
    union all
    select x, y, z, 0 as flag from (select x, y, z from topo_df except select x, y, z from elecs_df)
    union all
    select x, y, z, 0 as flag from edges_df
    """
).df()

# grid_x, grid_y = np.meshgrid(np.linspace(-6, 7, 4, endpoint=True), np.linspace(-6, 7, 4, endpoint=True))
# grid_z = griddata((surf_df['x'], surf_df['y']), surf_df['z'], (grid_x, grid_y), method='linear')
# grid_df = pd.DataFrame(data={'x': grid_x.ravel(), 'y': grid_y.ravel(), 'z': grid_z.ravel()})
# surf_df = db.sql("""
#     select x, y, z, flag from surf_df
#     union all
#     select x, y, z, 0 as flag from (select x, y, z from grid_df except select x, y, z from surf_df)
# """).df()

surf_xy = surf_df[['x', 'y']].to_numpy()
surf_tri = Delaunay(surf_xy, qhull_options='QJ')

plt.triplot(surf_xy[:, 0], surf_xy[:, 1], surf_tri.simplices)
plt.show()

# Surface points
elecSurfTags = []  # surface electrode tags
for i, p in surf_df.iterrows():
    if p['flag'] == 1:
        pt = geo.addPoint(p['x'], p['y'], p['z'], 0.1, i + 1)
        elecSurfTags.append(pt)
    else:
        pt = geo.addPoint(p['x'], p['y'], p['z'], 1, i + 1)
    geo.synchronize()

# Bottom points
pointBottomTags = []
for i, p in edgesBottom_df.iterrows():
    pt = geo.addPoint(p['x'], p['y'], p['z'], 1, i + 1000)
    pointBottomTags.append(pt)
    geo.synchronize()

# Surface plane surfaces
surfPlaneSurfTags = []
for i, s in enumerate(surf_tri.simplices):
    s += 1
    l1 = geo.addLine(s[0], s[1])
    l2 = geo.addLine(s[1], s[2])
    l3 = geo.addLine(s[2], s[0])
    curveLoop = geo.addCurveLoop([l1, l2, l3])
    planeSurf = geo.addPlaneSurface([curveLoop])
    surfPlaneSurfTags.append(planeSurf)
    geo.synchronize()

# Vertical edges
elecSurfCnt = len(elecSurfTags)
# assuming the 4 surface corners are consecutive to the electrode tags
# and that the matching bottoms start from 1000
vl1 = geo.addLine(elecSurfCnt + 1, 1000)
vl2 = geo.addLine(elecSurfCnt + 2, 1001)
vl3 = geo.addLine(elecSurfCnt + 3, 1002)
vl4 = geo.addLine(elecSurfCnt + 4, 1003)

# Bottom lines
bl1 = geo.addLine(1000, 1001)
bl2 = geo.addLine(1001, 1002)
bl3 = geo.addLine(1002, 1003)
bl4 = geo.addLine(1003, 1000)
# Bottom loop and surface
bottomCurveLoop = geo.addCurveLoop([bl1, bl2, bl3, bl4])
bottomSurf = geo.addPlaneSurface([bottomCurveLoop])

# Redefine surface edges
sl1 = geo.addLine(elecSurfCnt + 1, elecSurfCnt + 2)
sl2 = geo.addLine(elecSurfCnt + 2, elecSurfCnt + 3)
sl3 = geo.addLine(elecSurfCnt + 3, elecSurfCnt + 4)
sl4 = geo.addLine(elecSurfCnt + 4, elecSurfCnt + 1)

# Side line loops
sideCurveLoop1 = geo.addCurveLoop([vl1, bl1, -vl2, -sl1])
sideCurveLoop2 = geo.addCurveLoop([vl2, bl2, -vl3, -sl2])
sideCurveLoop3 = geo.addCurveLoop([vl3, bl3, -vl4, -sl3])
sideCurveLoop4 = geo.addCurveLoop([vl4, bl4, -vl1, -sl4])
sideSurf1 = geo.addPlaneSurface([sideCurveLoop1])
sideSurf2 = geo.addPlaneSurface([sideCurveLoop2])
sideSurf3 = geo.addPlaneSurface([sideCurveLoop3])
sideSurf4 = geo.addPlaneSurface([sideCurveLoop4])
sideSurfs = [sideSurf1, sideSurf2, sideSurf3, sideSurf4]


# Volume
surfaceLoop = surfPlaneSurfTags + sideSurfs + [bottomSurf]
sl1 = gmsh.model.geo.addSurfaceLoop(surfaceLoop)
v1 = gmsh.model.geo.addVolume([sl1])

# Buried electrodes
elecs_volume = elecs_df.loc[elecs_df['buried'] == 1, ['x', 'y', 'z']]
elecVolumeTags = []
for i, p in elecs_volume.iterrows():
    evt = geo.addPoint(p['x'], p['y'], p['z'], 0.1, 2000 + i)
    elecVolumeTags.append(evt)
geo.synchronize()

mesh.embed(0, elecVolumeTags, 3, v1)
geo.synchronize()

# Physical groups for pygimli
elecTags = elecVolumeTags + elecSurfTags
pgrp = gmsh.model.addPhysicalGroup(0, elecTags, 99)
gmsh.model.setPhysicalName(0, pgrp, "Electrodes")
geo.synchronize()
pgrp = gmsh.model.addPhysicalGroup(2, surfPlaneSurfTags, 1, "Neumann")
geo.synchronize()
pgrp = gmsh.model.addPhysicalGroup(2, [bottomSurf] + sideSurfs, 2, "MixedBC")
geo.synchronize()
pgrp = gmsh.model.addPhysicalGroup(3, [v1], 2)
gmsh.model.setPhysicalName(3, pgrp, "invVol")

# Reorganize the tags to handle the duplicates
geo.removeAllDuplicates()
geo.remove_all_duplicates()
geo.synchronize()

refine = True
if refine:
    mesh.field.add("Distance", 1)
    mesh.field.setNumbers(1, "NodesList", elecTags)
    mesh.field.add("Threshold", 2)
    mesh.field.setNumber(2, "IField", 1)
    mesh.field.setNumber(2, "LcMin", 0.1)
    mesh.field.setNumber(2, "LcMax", 1)
    mesh.field.setNumber(2, "DistMin", 0.1)
    mesh.field.setNumber(2, "DistMax", 3)
    mesh.field.setNumber(2, "StopAtDistMax", 1)
    mesh.field.setAsBackgroundMesh(2)

geo.synchronize()

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

sys.exit()


c1 = geo.addPoint(-1, -1, 0, 0.05, 101)
c2 = geo.addPoint(-1, 2, 0, 0.05, 102)
c3 = geo.addPoint(2, 2, 0, 0.05, 103)
c4 = geo.addPoint(2, -1, 0, 0.05, 104)
geo.synchronize()
l1 = geo.addLine(c1, c2)
l2 = geo.addLine(c2, c3)
l3 = geo.addLine(c3, c4)
l4 = geo.addLine(c4, c1)
geo.synchronize()
curveLoop = geo.addCurveLoop([l1, l2, l3, l4])
planeSurf = geo.addPlaneSurface([curveLoop])

geo.synchronize()

mesh.addNodes(2, planeSurf, tags, points_xyz.flatten())
geo.synchronize()

if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

sys.exit()




# The API can be used to import a mesh without reading it from a file, by
# creating nodes and elements on the fly and storing them in model
# entities. These model entities can be existing CAD entities, or can be
# discrete entities, entirely defined by the mesh.
#
# Discrete entities can be reparametrized (see `t13.py') so that they can be
# remeshed later on; and they can also be combined with built-in CAD entities to
# produce hybrid models.
#
# We combine all these features in this tutorial to perform terrain meshing,
# where the terrain is described by a discrete surface (that we then
# reparametrize) combined with a CAD representation of the underground.

gmsh.initialize()

gmsh.model.add("x2")

# We will create the terrain surface mesh from N x N input data points:
extrapoints = (0.5, 0.5, 0)
N = 100


# Helper function to return a node tag given two indices i and j:
def tag(i, j):
    return (N + 1) * i + j + 1


# The x, y, z coordinates of all the nodes:
coords = []

# The tags of the corresponding nodes:
nodes = []

# The connectivities of the triangle elements (3 node tags per triangle) on the
# terrain surface:
tris = []


# The connectivities of the line elements on the 4 boundaries (2 node tags
# for each line element):
lin = [[], [], [], []]

# The connectivities of the point elements on the 4 corners (1 node tag for each
# point element):
pnt = [tag(0, 0), tag(N, 0), tag(N, N), tag(0, N)]

for i in range(N + 1):
    for j in range(N + 1):
        nodes.append(tag(i, j))
        coords.extend([
            float(i) / N,
            float(j) / N, 0.05 * math.sin(10 * float(i + j) / N)
        ])
        if i > 0 and j > 0:
            tris.extend([tag(i - 1, j - 1), tag(i, j - 1), tag(i - 1, j)])
            tris.extend([tag(i, j - 1), tag(i, j), tag(i - 1, j)])
        if (i == 0 or i == N) and j > 0:
            lin[3 if i == 0 else 1].extend([tag(i, j - 1), tag(i, j)])
        if (j == 0 or j == N) and i > 0:
            lin[0 if j == 0 else 2].extend([tag(i - 1, j), tag(i, j)])

# Create 4 discrete points for the 4 corners of the terrain surface:
for i in range(4):
    gmsh.model.addDiscreteEntity(0, i + 1)
gmsh.model.setCoordinates(1, 0, 0, coords[3 * tag(0, 0) - 1])
gmsh.model.setCoordinates(2, 1, 0, coords[3 * tag(N, 0) - 1])
gmsh.model.setCoordinates(3, 1, 1, coords[3 * tag(N, N) - 1])
gmsh.model.setCoordinates(4, 0, 1, coords[3 * tag(0, N) - 1])

# Create 4 discrete bounding curves, with their boundary points:
for i in range(4):
    gmsh.model.addDiscreteEntity(1, i + 1, [i + 1, i + 2 if i < 3 else 1])

# Create one discrete surface, with its bounding curves:
gmsh.model.addDiscreteEntity(2, 1, [1, 2, -3, -4])

# Add all the nodes on the surface (for simplicity... see below):
gmsh.model.mesh.addNodes(2, 1, nodes, coords)

# Add point elements on the 4 points, line elements on the 4 curves, and
# triangle elements on the surface:
for i in range(4):
    # Type 15 for point elements:
    gmsh.model.mesh.addElementsByType(i + 1, 15, [], [pnt[i]])
    # Type 1 for 2-node line elements:
    gmsh.model.mesh.addElementsByType(i + 1, 1, [], lin[i])
# Type 2 for 3-node triangle elements:
gmsh.model.mesh.addElementsByType(1, 2, [], tris)

# Reclassify the nodes on the curves and the points (since we put them all on
# the surface before with `addNodes' for simplicity)
gmsh.model.mesh.reclassifyNodes()

# Create a geometry for the discrete curves and surfaces, so that we can remesh
# them later on:
gmsh.model.mesh.createGeometry()

# Note that for more complicated meshes, e.g. for on input unstructured STL
# mesh, we could use `classifySurfaces()' to automatically create the discrete
# entities and the topology; but we would then have to extract the boundaries
# afterwards.

# Create other build-in CAD entities to form one volume below the terrain
# surface. Beware that only built-in CAD entities can be hybrid, i.e. have
# discrete entities on their boundary: OpenCASCADE does not support this
# feature.
p1 = gmsh.model.geo.addPoint(0, 0, -0.5)
p2 = gmsh.model.geo.addPoint(1, 0, -0.5)
p3 = gmsh.model.geo.addPoint(1, 1, -0.5)
p4 = gmsh.model.geo.addPoint(0, 1, -0.5)
c1 = gmsh.model.geo.addLine(p1, p2)
c2 = gmsh.model.geo.addLine(p2, p3)
c3 = gmsh.model.geo.addLine(p3, p4)
c4 = gmsh.model.geo.addLine(p4, p1)
c10 = gmsh.model.geo.addLine(p1, 1)
c11 = gmsh.model.geo.addLine(p2, 2)
c12 = gmsh.model.geo.addLine(p3, 3)
c13 = gmsh.model.geo.addLine(p4, 4)
ll1 = gmsh.model.geo.addCurveLoop([c1, c2, c3, c4])
s1 = gmsh.model.geo.addPlaneSurface([ll1])
ll3 = gmsh.model.geo.addCurveLoop([c1, c11, -1, -c10])
s3 = gmsh.model.geo.addPlaneSurface([ll3])
ll4 = gmsh.model.geo.addCurveLoop([c2, c12, -2, -c11])
s4 = gmsh.model.geo.addPlaneSurface([ll4])
ll5 = gmsh.model.geo.addCurveLoop([c3, c13, 3, -c12])
s5 = gmsh.model.geo.addPlaneSurface([ll5])
ll6 = gmsh.model.geo.addCurveLoop([c4, c10, 4, -c13])
s6 = gmsh.model.geo.addPlaneSurface([ll6])
sl1 = gmsh.model.geo.addSurfaceLoop([s1, s3, s4, s5, s6, 1])
v1 = gmsh.model.geo.addVolume([sl1])
gmsh.model.geo.synchronize()

# Set this to True to build a fully hex mesh:
#transfinite = True
transfinite = False
transfiniteAuto = False

if transfinite:
    NN = 30
    for c in gmsh.model.getEntities(1):
        gmsh.model.mesh.setTransfiniteCurve(c[1], NN)
    for s in gmsh.model.getEntities(2):
        gmsh.model.mesh.setTransfiniteSurface(s[1])
        gmsh.model.mesh.setRecombine(s[0], s[1])
        gmsh.model.mesh.setSmoothing(s[0], s[1], 100)
    gmsh.model.mesh.setTransfiniteVolume(v1)
elif transfiniteAuto:
    gmsh.option.setNumber('Mesh.MeshSizeMin', 0.5)
    gmsh.option.setNumber('Mesh.MeshSizeMax', 0.5)
    # setTransfiniteAutomatic() uses the sizing constraints to set the number
    # of points
    gmsh.model.mesh.setTransfiniteAutomatic()
else:
    gmsh.option.setNumber('Mesh.MeshSizeMin', 0.05)
    gmsh.option.setNumber('Mesh.MeshSizeMax', 0.05)

gmsh.model.mesh.generate(3)
gmsh.write('x2.msh')

# Launch the GUI to see the results:
if '-nopopup' not in sys.argv:
    gmsh.fltk.run()

gmsh.finalize()

# EXTRA

# surf_df_check = db.sql("""
# select x, y, z, max(flag) as flag
# from (
#     select x, y, z, 1 as flag from elecs_df where buried = 0
#     union all
#     select x, y, z, 0 as flag from topo_df
#     )
# group by x, y, z
# order by flag, x,  y, z
# """).df()

# assert all(surf_df == surf_df_check)
