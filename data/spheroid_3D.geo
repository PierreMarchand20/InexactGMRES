SetFactory("OpenCASCADE");
Mesh.MshFileVersion = 2.2;

DefineConstant[
    npplo = {10, Name "npplo"},
    k = {5, Name "k"}
];
Printf("npplo : %f",npplo);
Printf("k : %f",k);

meshsize=2*Pi/(npplo*k);
Mesh.CharacteristicLengthMin = meshsize;
Mesh.CharacteristicLengthMax = meshsize;
Printf("Meshsize : %f",meshsize);

// Closed prolate spheroid: the outer shape of elliptic_cavity_bigger_3D,
// without the cavity (convex, no trapping).
Sphere(1) = {0,0,0,1};
Dilate {{0, 0, 0}, {1.3, 0.6, 0.6}} { Volume{1}; }
