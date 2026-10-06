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

Sphere(1) = {0,0,0,1};
Sphere(2) = {0,0,0,1};

Dilate {{0, 0, 0}, {1, 0.5, 0.5}} { Volume{1}; }
Dilate {{0, 0, 0}, {1.3, 0.6, 0.6}} { Volume{2}; }


Box(3)={Cos(9*Pi/10),-5,-5,-5,10,10};


BooleanDifference{Volume{2};Delete;}{Volume{1};Delete;}
BooleanDifference{Volume{2};Delete;}{Volume{3};Delete;}


