/* l = Unique(ListFromFile["elec.csv"]); */
xs[] = {0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10,10.5,11,11.5};
zs[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

etot = #xs[];
elecc = 0.10;
enum = 1;

xmin = xs[0];
xmax = xs[etot -1];

zleft = zs[0];
zright = zs[etot -1];

xlen = Abs(xmax - xmin);
geomc = xlen;

For i In {0: etot - 1}
    x = xs[i];
    z = zs[i];
    Printf("i = %g    x = %g", i, x);
    Point(enum) = {x, z, 0, elecc};
    enum ++;
EndFor

Printf("xmin = %g",xmin);
Printf("xmax = %g",xmax);
Printf("xlen = %g",xlen);

xgeomin = xmin - (2 * xlen);
xgeomax = xmax + (2 * xlen);
ygeominleft = zleft - (2.5 * xlen);
ygeominright = zright - (2.5 * xlen);
ygeomaxleft = zleft;
ygeomaxright = zright;

Point(101) = {xgeomin, ygeomaxleft, 0, geomc};
Point(102) = {xgeomax, ygeomaxright, 0, geomc};
Point(103) = {xgeomax, ygeominright, 0, geomc};
Point(104) = {xgeomin, ygeominleft, 0, geomc};

Field[1] = Distance;
Field[1].PointsList = {1:etot};
Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = elecc * 2;
Field[2].LcMax = geomc / 2;
Field[2].DistMin = elecc * 10;
Field[2].DistMax = geomc;
Field[3] = Min;
Field[3].FieldsList = {2};
Background Field = 3;

Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.MeshSizeExtendFromBoundary = 0;
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;


points = {101, 1: enum - 1, 102, 103, 104, 101};

lenpoints = #points[];
linenum = 1;
For i In {0: lenpoints -2}
    ls = points[i];
    le = points[i + 1];
    Line(linenum) = {ls, le};
    Printf("line %g ls %g le %g", linenum, ls, le);
    linenum ++;
EndFor

Curve Loop(1) = {1:linenum - 1};
Plane Surface(1) = {1};

Physical Line(1) = {1: etot+1};
Physical Line(4) = {etot+2: etot+4};
Physical Surface(2) = {1};
Physical Point(99) = {1: etot};
