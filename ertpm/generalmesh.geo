/* l = Unique(ListFromFile["elec.csv"]); */
xs[] = {0,0.2,0.4,0.6,0.8,1,1.2,1.4,1.6,1.8,2,2.2,2.4,2.6,2.8,3,3.2,3.4,3.6,3.8,4,4.2,4.4,4.6,4.8,5,5.2,5.4,5.6,5.8,6,6.2,6.4,6.6,6.8,7,7.2,7.4,7.6,7.8,8,8.2,8.4,8.6,8.8,9,9.2,9.4,9.6,9.8,10,10.2,10.4,10.6,10.8,11,11.2,11.4,11.6,11.8,12,12.2,12.4,12.6,12.8,13,13.2,13.4,13.6,13.8,14,14.2};
zs[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

etot = #xs[];
elecc = 0.05;
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
Field[2].DistMax = geomc * 3;
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
