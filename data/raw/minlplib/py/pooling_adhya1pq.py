# NLP written by GAMS Convert at 02/17/22 17:21:47
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#        49       22        0       27        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        33       33        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#       197      157       40
#
# Reformulation has removed 1 variable and 1 equation

from pyomo.environ import *

model = m = ConcreteModel()

m.x1 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x2 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x3 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x4 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x5 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x6 = Var(within=Reals, bounds=(0,10), initialize=0)
m.x7 = Var(within=Reals, bounds=(0,25), initialize=0)
m.x8 = Var(within=Reals, bounds=(0,30), initialize=0)
m.x9 = Var(within=Reals, bounds=(0,10), initialize=0)
m.x10 = Var(within=Reals, bounds=(0,10), initialize=0)
m.x11 = Var(within=Reals, bounds=(0,25), initialize=0)
m.x12 = Var(within=Reals, bounds=(0,30), initialize=0)
m.x13 = Var(within=Reals, bounds=(0,10), initialize=0)
m.x14 = Var(within=Reals, bounds=(0,10), initialize=0)
m.x15 = Var(within=Reals, bounds=(0,25), initialize=0)
m.x16 = Var(within=Reals, bounds=(0,30), initialize=0)
m.x17 = Var(within=Reals, bounds=(0,10), initialize=0)
m.x18 = Var(within=Reals, bounds=(0,10), initialize=0)
m.x19 = Var(within=Reals, bounds=(0,25), initialize=0)
m.x20 = Var(within=Reals, bounds=(0,30), initialize=0)
m.x21 = Var(within=Reals, bounds=(0,10), initialize=0)
m.x22 = Var(within=Reals, bounds=(0,10), initialize=0)
m.x23 = Var(within=Reals, bounds=(0,25), initialize=0)
m.x24 = Var(within=Reals, bounds=(0,30), initialize=0)
m.x25 = Var(within=Reals, bounds=(0,10), initialize=0)
m.x26 = Var(within=Reals, bounds=(0,10), initialize=0)
m.x27 = Var(within=Reals, bounds=(0,25), initialize=0)
m.x28 = Var(within=Reals, bounds=(0,30), initialize=0)
m.x29 = Var(within=Reals, bounds=(0,10), initialize=0)
m.x30 = Var(within=Reals, bounds=(0,10), initialize=0)
m.x31 = Var(within=Reals, bounds=(0,25), initialize=0)
m.x32 = Var(within=Reals, bounds=(0,30), initialize=0)
m.x33 = Var(within=Reals, bounds=(0,10), initialize=0)

m.obj = Objective(sense=minimize, expr= -9 * m.x14 - 18 * m.x15 - 8 * m.x16 - 3
    * m.x17 - 13 * m.x18 - 22 * m.x19 - 12 * m.x20 - 7 * m.x21 - 14 * m.x22 -
    23 * m.x23 - 13 * m.x24 - 8 * m.x25 - 6 * m.x26 - 15 * m.x27 - 5 * m.x28 -
    11 * m.x30 - 20 * m.x31 - 10 * m.x32 - 5 * m.x33)

m.e1 = Constraint(expr= m.x14 + m.x15 + m.x16 + m.x17 <= 75)
m.e2 = Constraint(expr= m.x18 + m.x19 + m.x20 + m.x21 <= 75)
m.e3 = Constraint(expr= m.x22 + m.x23 + m.x24 + m.x25 <= 75)
m.e4 = Constraint(expr= m.x26 + m.x27 + m.x28 + m.x29 <= 75)
m.e5 = Constraint(expr= m.x30 + m.x31 + m.x32 + m.x33 <= 75)
m.e6 = Constraint(expr= m.x14 + m.x15 + m.x16 + m.x17 + m.x18 + m.x19 + m.x20
    + m.x21 <= 75)
m.e7 = Constraint(expr= m.x22 + m.x23 + m.x24 + m.x25 + m.x26 + m.x27 + m.x28
    + m.x29 + m.x30 + m.x31 + m.x32 + m.x33 <= 75)
m.e8 = Constraint(expr= m.x14 + m.x18 + m.x22 + m.x26 + m.x30 <= 10)
m.e9 = Constraint(expr= m.x15 + m.x19 + m.x23 + m.x27 + m.x31 <= 25)
m.e10 = Constraint(expr= m.x16 + m.x20 + m.x24 + m.x28 + m.x32 <= 30)
m.e11 = Constraint(expr= m.x17 + m.x21 + m.x25 + m.x29 + m.x33 <= 10)
m.e12 = Constraint(expr= -2 * m.x14 + m.x18 + m.x22 - 2 * m.x30 <= 0)
m.e13 = Constraint(expr= 3 * m.x14 - 2 * m.x18 + 2.5 * m.x22 - 0.3 * m.x30
    <= 0)
m.e14 = Constraint(expr= 0.75 * m.x14 - 0.25 * m.x18 - 0.25 * m.x22 - 0.25 *
    m.x26 + 0.75 * m.x30 <= 0)
m.e15 = Constraint(expr= -0.25 * m.x14 + 1.25 * m.x18 + 0.15 * m.x22 + 0.25 *
    m.x26 + 0.85 * m.x30 <= 0)
m.e16 = Constraint(expr= -3 * m.x15 - m.x27 - 3 * m.x31 <= 0)
m.e17 = Constraint(expr= 3.5 * m.x15 - 1.5 * m.x19 + 3 * m.x23 + 0.5 * m.x27 +
    0.2 * m.x31 <= 0)
m.e18 = Constraint(expr= 0.5 * m.x15 - 0.5 * m.x19 - 0.5 * m.x23 - 0.5 * m.x27
    + 0.5 * m.x31 <= 0)
m.e19 = Constraint(expr= -m.x15 + 0.5 * m.x19 - 0.6 * m.x23 - 0.5 * m.x27 + 0.1
    * m.x31 <= 0)
m.e20 = Constraint(expr= -0.5 * m.x16 + 2.5 * m.x20 + 2.5 * m.x24 + 1.5 * m.x28
    - 0.5 * m.x32 <= 0)
m.e21 = Constraint(expr= 0.5 * m.x16 - 4.5 * m.x20 - 2.5 * m.x28 - 2.8 * m.x32
    <= 0)
m.e22 = Constraint(expr= 0.1 * m.x16 - 0.9 * m.x20 - 0.9 * m.x24 - 0.9 * m.x28
    + 0.1 * m.x32 <= 0)
m.e23 = Constraint(expr= -0.3 * m.x16 + 1.2 * m.x20 + 0.1 * m.x24 + 0.2 * m.x28
    + 0.8 * m.x32 <= 0)
m.e24 = Constraint(expr= -2 * m.x17 + m.x21 + m.x25 - 2 * m.x33 <= 0)
m.e25 = Constraint(expr= 2 * m.x17 - 3 * m.x21 + 1.5 * m.x25 - m.x29 - 1.3 *
    m.x33 <= 0)
m.e26 = Constraint(expr= -m.x21 - m.x25 - m.x29 <= 0)
m.e27 = Constraint(expr= -1.3 * m.x17 + 0.2 * m.x21 - 0.9 * m.x25 - 0.8 * m.x29
    - 0.2 * m.x33 <= 0)
m.e28 = Constraint(expr= m.x1 + m.x2 == 1)
m.e29 = Constraint(expr= m.x3 + m.x4 + m.x5 == 1)
m.e30 = Constraint(expr= -m.x1 * m.x6 + m.x14 == 0)
m.e31 = Constraint(expr= -m.x1 * m.x7 + m.x15 == 0)
m.e32 = Constraint(expr= -m.x1 * m.x8 + m.x16 == 0)
m.e33 = Constraint(expr= -m.x1 * m.x9 + m.x17 == 0)
m.e34 = Constraint(expr= -m.x2 * m.x6 + m.x18 == 0)
m.e35 = Constraint(expr= -m.x2 * m.x7 + m.x19 == 0)
m.e36 = Constraint(expr= -m.x2 * m.x8 + m.x20 == 0)
m.e37 = Constraint(expr= -m.x2 * m.x9 + m.x21 == 0)
m.e38 = Constraint(expr= -m.x3 * m.x10 + m.x22 == 0)
m.e39 = Constraint(expr= -m.x3 * m.x11 + m.x23 == 0)
m.e40 = Constraint(expr= -m.x3 * m.x12 + m.x24 == 0)
m.e41 = Constraint(expr= -m.x3 * m.x13 + m.x25 == 0)
m.e42 = Constraint(expr= -m.x4 * m.x10 + m.x26 == 0)
m.e43 = Constraint(expr= -m.x4 * m.x11 + m.x27 == 0)
m.e44 = Constraint(expr= -m.x4 * m.x12 + m.x28 == 0)
m.e45 = Constraint(expr= -m.x4 * m.x13 + m.x29 == 0)
m.e46 = Constraint(expr= -m.x5 * m.x10 + m.x30 == 0)
m.e47 = Constraint(expr= -m.x5 * m.x11 + m.x31 == 0)
m.e48 = Constraint(expr= -m.x5 * m.x12 + m.x32 == 0)
m.e49 = Constraint(expr= -m.x5 * m.x13 + m.x33 == 0)
