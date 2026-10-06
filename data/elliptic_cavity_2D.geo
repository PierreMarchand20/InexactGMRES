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

Disk(1) = {0,0,0,1,0.5};
Disk(2) = {0,0,0,1.3,0.6};

Rectangle(3)={Cos(7*Pi/10),-5,0,-5,10};


BooleanDifference{Surface{2};Delete;}{Surface{1};Delete;}
BooleanDifference{Surface{2};Delete;}{Surface{3};Delete;}


