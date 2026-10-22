$offlisting
*  
*  Equation counts
*      Total        E        G        L        N        X        C        B
*         17        8        0        9        0        0        0        0
*  
*  Variable counts
*                   x        b        i      s1s      s2s       sc       si
*      Total     cont   binary  integer     sos1     sos2    scont     sint
*         14       14        0        0        0        0        0        0
*  FX      0
*  
*  Nonzero counts
*      Total    const       NL      DLL
*         59       47       12        0
*
*  Solve m using NLP minimizing objvar;


Variables  objvar,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14;

Positive Variables  x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14;

Equations  e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17;


e1..    objvar - x5 + 5*x6 + 3*x9 + 9*x10 - 6*x11 - 7*x13 - x14 =E= 0;

e2..    x9 + x10 =L= 300;

e3..    x11 + x12 =L= 50;

e4..    x13 + x14 =L= 300;

e5..    x5 + x6 =L= 300;

e6..    x9 + x10 + x11 + x12 + x13 + x14 =L= 300;

e7..    x5 + x9 + x11 + x13 =L= 100;

e8..    x6 + x10 + x12 + x14 =L= 200;

e9..  - 0.5*x5 + 0.5*x9 - 1.5*x11 - 1.5*x13 =L= 0;

e10..    0.5*x6 + 1.5*x10 - 0.5*x12 - 0.5*x14 =L= 0;

e11..    x2 + x3 + x4 =E= 1;

e12.. -x2*x7 + x9 =E= 0;

e13.. -x2*x8 + x10 =E= 0;

e14.. -x3*x7 + x11 =E= 0;

e15.. -x3*x8 + x12 =E= 0;

e16.. -x4*x7 + x13 =E= 0;

e17.. -x4*x8 + x14 =E= 0;

* set non-default bounds
x2.up = 1;
x3.up = 1;
x4.up = 1;
x5.up = 100;
x6.up = 200;
x7.up = 100;
x8.up = 200;
x9.up = 100;
x10.up = 200;
x11.up = 50;
x12.up = 50;
x13.up = 100;
x14.up = 200;

Model m / all /;

m.limrow=0; m.limcol=0;
m.tolproj=0.0;

$if NOT '%gams.u1%' == '' $include '%gams.u1%'

$if not set NLP $set NLP NLP
Solve m using %NLP% minimizing objvar;
