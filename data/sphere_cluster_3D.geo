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

// 3 x 3 x 2 cluster of spheres of radius 0.25, centres 0.6 apart (gaps of
// 0.1): no cavity, but waves bounce between the spheres (multiple scattering).
r = 0.25;
s = 0.6;
v = 0;
For i In {0:2}
  For j In {0:2}
    For l In {0:1}
      v += 1;
      Sphere(v) = {(i - 1) * s, (j - 1) * s, (l - 0.5) * s, r};
    EndFor
  EndFor
EndFor
