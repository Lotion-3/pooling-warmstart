$offlisting
*  
*  Equation counts
*      Total        E        G        L        N        X        C        B
*         50       23        0       27        0        0        0        0
*  
*  Variable counts
*                   x        b        i      s1s      s2s       sc       si
*      Total     cont   binary  integer     sos1     sos2    scont     sint
*         34       34        0        0        0        0        0        0
*  FX      0
*  
*  Nonzero counts
*      Total    const       NL      DLL
*        217      177       40        0
*
*  Solve m using NLP minimizing objvar;


Variables  objvar,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18
          ,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34;

Positive Variables  x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17
          ,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34;

Equations  e1,e2,e3,e4,e5,e6,e7,e8,e9,e10,e11,e12,e13,e14,e15,e16,e17,e18,e19
          ,e20,e21,e22,e23,e24,e25,e26,e27,e28,e29,e30,e31,e32,e33,e34,e35,e36
          ,e37,e38,e39,e40,e41,e42,e43,e44,e45,e46,e47,e48,e49,e50;


e1..    objvar + 9*x15 + 18*x16 + 8*x17 + 3*x18 + 13*x19 + 22*x20 + 12*x21
      + 7*x22 + 14*x23 + 23*x24 + 13*x25 + 8*x26 + 6*x27 + 15*x28 + 5*x29
      + 11*x31 + 20*x32 + 10*x33 + 5*x34 =E= 0;

e2..    x15 + x16 + x17 + x18 =L= 75;

e3..    x19 + x20 + x21 + x22 =L= 75;

e4..    x23 + x24 + x25 + x26 =L= 75;

e5..    x27 + x28 + x29 + x30 =L= 75;

e6..    x31 + x32 + x33 + x34 =L= 75;

e7..    x15 + x16 + x17 + x18 + x19 + x20 + x21 + x22 =L= 75;

e8..    x23 + x24 + x25 + x26 + x27 + x28 + x29 + x30 + x31 + x32 + x33 + x34
      =L= 75;

e9..    x15 + x19 + x23 + x27 + x31 =L= 10;

e10..    x16 + x20 + x24 + x28 + x32 =L= 25;

e11..    x17 + x21 + x25 + x29 + x33 =L= 30;

e12..    x18 + x22 + x26 + x30 + x34 =L= 10;

e13..  - 2*x15 + x19 + x23 - 2*x31 =L= 0;

e14..    3*x15 - 2*x19 + 2.5*x23 - 0.3*x31 =L= 0;

e15..    0.75*x15 - 0.25*x19 - 0.25*x23 - 0.25*x27 + 0.75*x31 =L= 0;

e16..  - 0.25*x15 + 1.25*x19 + 0.15*x23 + 0.25*x27 + 0.85*x31 =L= 0;

e17..  - 3*x16 - x28 - 3*x32 =L= 0;

e18..    3.5*x16 - 1.5*x20 + 3*x24 + 0.5*x28 + 0.2*x32 =L= 0;

e19..    0.5*x16 - 0.5*x20 - 0.5*x24 - 0.5*x28 + 0.5*x32 =L= 0;

e20..  - x16 + 0.5*x20 - 0.6*x24 - 0.5*x28 + 0.1*x32 =L= 0;

e21..  - 0.5*x17 + 2.5*x21 + 2.5*x25 + 1.5*x29 - 0.5*x33 =L= 0;

e22..    0.5*x17 - 4.5*x21 - 2.5*x29 - 2.8*x33 =L= 0;

e23..    0.1*x17 - 0.9*x21 - 0.9*x25 - 0.9*x29 + 0.1*x33 =L= 0;

e24..  - 0.3*x17 + 1.2*x21 + 0.1*x25 + 0.2*x29 + 0.8*x33 =L= 0;

e25..  - 2*x18 + x22 + x26 - 2*x34 =L= 0;

e26..    2*x18 - 3*x22 + 1.5*x26 - x30 - 1.3*x34 =L= 0;

e27..  - x22 - x26 - x30 =L= 0;

e28..  - 1.3*x18 + 0.2*x22 - 0.9*x26 - 0.8*x30 - 0.2*x34 =L= 0;

e29..    x2 + x3 =E= 1;

e30..    x4 + x5 + x6 =E= 1;

e31.. -x2*x7 + x15 =E= 0;

e32.. -x2*x8 + x16 =E= 0;

e33.. -x2*x9 + x17 =E= 0;

e34.. -x2*x10 + x18 =E= 0;

e35.. -x3*x7 + x19 =E= 0;

e36.. -x3*x8 + x20 =E= 0;

e37.. -x3*x9 + x21 =E= 0;

e38.. -x3*x10 + x22 =E= 0;

e39.. -x4*x11 + x23 =E= 0;

e40.. -x4*x12 + x24 =E= 0;

e41.. -x4*x13 + x25 =E= 0;

e42.. -x4*x14 + x26 =E= 0;

e43.. -x5*x11 + x27 =E= 0;

e44.. -x5*x12 + x28 =E= 0;

e45.. -x5*x13 + x29 =E= 0;

e46.. -x5*x14 + x30 =E= 0;

e47.. -x6*x11 + x31 =E= 0;

e48.. -x6*x12 + x32 =E= 0;

e49.. -x6*x13 + x33 =E= 0;

e50.. -x6*x14 + x34 =E= 0;

* set non-default bounds
x2.up = 1;
x3.up = 1;
x4.up = 1;
x5.up = 1;
x6.up = 1;
x7.up = 10;
x8.up = 25;
x9.up = 30;
x10.up = 10;
x11.up = 10;
x12.up = 25;
x13.up = 30;
x14.up = 10;
x15.up = 10;
x16.up = 25;
x17.up = 30;
x18.up = 10;
x19.up = 10;
x20.up = 25;
x21.up = 30;
x22.up = 10;
x23.up = 10;
x24.up = 25;
x25.up = 30;
x26.up = 10;
x27.up = 10;
x28.up = 25;
x29.up = 30;
x30.up = 10;
x31.up = 10;
x32.up = 25;
x33.up = 30;
x34.up = 10;

Model m / all /;

m.limrow=0; m.limcol=0;
m.tolproj=0.0;

$if NOT '%gams.u1%' == '' $include '%gams.u1%'

$if not set NLP $set NLP NLP
Solve m using %NLP% minimizing objvar;
