$offlisting
*  
*  Equation counts
*      Total        E        G        L        N        X        C        B
*         35       19        0       16        0        0        0        0
*  
*  Variable counts
*                   x        b        i      s1s      s2s       sc       si
*      Total     cont   binary  integer     sos1     sos2    scont     sint
*         37       37        0        0        0        0        0        0
*  FX      0
*  
*  Nonzero counts
*      Total    const       NL      DLL
*        160      128       32        0
*
*  Solve m using NLP minimizing objvar;


Variables  objvar,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18
          ,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35
          ,x36,x37;

Positive Variables  x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17
          ,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34
          ,x35,x36,x37;

Equations  e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19
          ,e20,e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,e31,e32,e33,e34,e35;


e1..    objvar - x6 + 5*x7 - 4*x8 + 2*x9 + 2*x10 + 8*x11 - x12 + 5*x13 + 3*x22
      + 9*x23 + 6*x25 - 7*x26 - x27 - 10*x28 - 4*x29 + 6*x30 + 12*x31 + 3*x32
      + 9*x33 - 4*x34 + 2*x35 - 7*x36 - x37 =E= 0;

e2..    x22 + x23 + x24 + x25 =L= 600;

e3..    x26 + x27 + x28 + x29 =L= 600;

e4..    x6 + x7 + x8 + x9 =L= 600;

e5..    x30 + x31 + x32 + x33 =L= 600;

e6..    x34 + x35 + x36 + x37 =L= 600;

e7..    x10 + x11 + x12 + x13 =L= 600;

e8..    x22 + x23 + x24 + x25 + x26 + x27 + x28 + x29 =L= 600;

e9..    x30 + x31 + x32 + x33 + x34 + x35 + x36 + x37 =L= 600;

e10..    x6 + x10 + x22 + x26 + x30 + x34 =L= 100;

e11..    x7 + x11 + x23 + x27 + x31 + x35 =L= 200;

e12..    x8 + x12 + x24 + x28 + x32 + x36 =L= 100;

e13..    x9 + x13 + x25 + x29 + x33 + x37 =L= 200;

e14..  - 0.5*x6 + 0.5*x22 - 1.5*x26 + x30 - x34 =L= 0;

e15..    0.5*x7 + x11 + 1.5*x23 - 0.5*x27 + 2*x31 =L= 0;

e16..  - x8 - 0.5*x12 - 2*x28 + 0.5*x32 - 1.5*x36 =L= 0;

e17..    0.5*x13 + x25 - x29 + 1.5*x33 - 0.5*x37 =L= 0;

e18..    x2 + x3 =E= 1;

e19..    x4 + x5 =E= 1;

e20.. -x2*x14 + x22 =E= 0;

e21.. -x2*x15 + x23 =E= 0;

e22.. -x2*x16 + x24 =E= 0;

e23.. -x2*x17 + x25 =E= 0;

e24.. -x3*x14 + x26 =E= 0;

e25.. -x3*x15 + x27 =E= 0;

e26.. -x3*x16 + x28 =E= 0;

e27.. -x3*x17 + x29 =E= 0;

e28.. -x4*x18 + x30 =E= 0;

e29.. -x4*x19 + x31 =E= 0;

e30.. -x4*x20 + x32 =E= 0;

e31.. -x4*x21 + x33 =E= 0;

e32.. -x5*x18 + x34 =E= 0;

e33.. -x5*x19 + x35 =E= 0;

e34.. -x5*x20 + x36 =E= 0;

e35.. -x5*x21 + x37 =E= 0;

* set non-default bounds
x2.up = 1;
x3.up = 1;
x4.up = 1;
x5.up = 1;
x6.up = 100;
x7.up = 200;
x8.up = 100;
x9.up = 200;
x10.up = 100;
x11.up = 200;
x12.up = 100;
x13.up = 200;
x14.up = 100;
x15.up = 200;
x16.up = 100;
x17.up = 200;
x18.up = 100;
x19.up = 200;
x20.up = 100;
x21.up = 200;
x22.up = 100;
x23.up = 200;
x24.up = 100;
x25.up = 200;
x26.up = 100;
x27.up = 200;
x28.up = 100;
x29.up = 200;
x30.up = 100;
x31.up = 200;
x32.up = 100;
x33.up = 200;
x34.up = 100;
x35.up = 200;
x36.up = 100;
x37.up = 200;

Model m / all /;

m.limrow=0; m.limcol=0;
m.tolproj=0.0;

$if NOT '%gams.u1%' == '' $include '%gams.u1%'

$if not set NLP $set NLP NLP
Solve m using %NLP% minimizing objvar;
