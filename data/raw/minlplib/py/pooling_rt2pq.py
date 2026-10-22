# NLP written by GAMS Convert at 02/17/22 17:21:49
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#        52       20        3       29        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        34       34        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#       277      241       36
#
# Reformulation has removed 1 variable and 1 equation

from pyomo.environ import *

model = m = ConcreteModel()

m.x1 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x2 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x3 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x4 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x5 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x6 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x7 = Var(within=Reals, bounds=(0,60.98), initialize=0)
m.x8 = Var(within=Reals, bounds=(0,161.29), initialize=0)
m.x9 = Var(within=Reals, bounds=(0,161.29), initialize=0)
m.x10 = Var(within=Reals, bounds=(0,5), initialize=0)
m.x11 = Var(within=Reals, bounds=(0,12.5), initialize=0)
m.x12 = Var(within=Reals, bounds=(0,12.5), initialize=0)
m.x13 = Var(within=Reals, bounds=(0,12.5), initialize=0)
m.x14 = Var(within=Reals, bounds=(0,17.5), initialize=0)
m.x15 = Var(within=Reals, bounds=(0,17.5), initialize=0)
m.x16 = Var(within=Reals, bounds=(0,17.5), initialize=0)
m.x17 = Var(within=Reals, bounds=(0,12.5), initialize=0)
m.x18 = Var(within=Reals, bounds=(0,12.5), initialize=0)
m.x19 = Var(within=Reals, bounds=(0,12.5), initialize=0)
m.x20 = Var(within=Reals, bounds=(0,17.5), initialize=0)
m.x21 = Var(within=Reals, bounds=(0,17.5), initialize=0)
m.x22 = Var(within=Reals, bounds=(0,17.5), initialize=0)
m.x23 = Var(within=Reals, bounds=(0,12.5), initialize=0)
m.x24 = Var(within=Reals, bounds=(0,12.5), initialize=0)
m.x25 = Var(within=Reals, bounds=(0,12.5), initialize=0)
m.x26 = Var(within=Reals, bounds=(0,17.5), initialize=0)
m.x27 = Var(within=Reals, bounds=(0,17.5), initialize=0)
m.x28 = Var(within=Reals, bounds=(0,17.5), initialize=0)
m.x29 = Var(within=Reals, bounds=(0,5), initialize=0)
m.x30 = Var(within=Reals, bounds=(0,5), initialize=0)
m.x31 = Var(within=Reals, bounds=(0,5), initialize=0)
m.x32 = Var(within=Reals, bounds=(0,5), initialize=0)
m.x33 = Var(within=Reals, bounds=(0,5), initialize=0)
m.x34 = Var(within=Reals, bounds=(0,5), initialize=0)

m.obj = Objective(sense=minimize, expr= -180.8 * m.x7 - 128 * m.x8 - 88 * m.x9
    + 110 * m.x10 - 140.8 * m.x17 - 180.8 * m.x18 - 100.8 * m.x19 - 140.8 *
    m.x20 - 180.8 * m.x21 - 100.8 * m.x22 - 128 * m.x23 - 168 * m.x24 - 88 *
    m.x25 - 128 * m.x26 - 168 * m.x27 - 88 * m.x28 + 110 * m.x29 + 70 * m.x30
    + 150 * m.x31 + 110 * m.x32 + 70 * m.x33 + 150 * m.x34)

m.e1 = Constraint(expr= m.x7 + m.x17 + m.x18 + m.x19 + m.x20 + m.x21 + m.x22
    <= 60.98)
m.e2 = Constraint(expr= m.x8 + m.x9 + m.x23 + m.x24 + m.x25 + m.x26 + m.x27 +
    m.x28 <= 161.29)
m.e3 = Constraint(expr= m.x10 + m.x29 + m.x30 + m.x31 + m.x32 + m.x33 + m.x34
    <= 5)
m.e4 = Constraint(expr= m.x17 + m.x18 + m.x19 + m.x23 + m.x24 + m.x25 + m.x29
    + m.x30 + m.x31 <= 12.5)
m.e5 = Constraint(expr= m.x20 + m.x21 + m.x22 + m.x26 + m.x27 + m.x28 + m.x32
    + m.x33 + m.x34 <= 17.5)
m.e6 = Constraint(expr= m.x8 + m.x10 + m.x17 + m.x20 + m.x23 + m.x26 + m.x29 +
    m.x32 >= 5)
m.e7 = Constraint(expr= m.x7 + m.x18 + m.x21 + m.x24 + m.x27 + m.x30 + m.x33
    >= 5)
m.e8 = Constraint(expr= m.x9 + m.x19 + m.x22 + m.x25 + m.x28 + m.x31 + m.x34
    >= 5)
m.e9 = Constraint(expr= m.x8 + m.x10 + m.x17 + m.x20 + m.x23 + m.x26 + m.x29 +
    m.x32 <= 300)
m.e10 = Constraint(expr= m.x7 + m.x18 + m.x21 + m.x24 + m.x27 + m.x30 + m.x33
    <= 300)
m.e11 = Constraint(expr= m.x9 + m.x19 + m.x22 + m.x25 + m.x28 + m.x31 + m.x34
    <= 300)
m.e12 = Constraint(expr= -0.17 * m.x8 - 0.04 * m.x10 + 0.0299999999999999 *
    m.x17 + 0.0299999999999999 * m.x20 - 0.17 * m.x23 - 0.17 * m.x26 - 0.04 *
    m.x29 - 0.04 * m.x32 <= 0)
m.e13 = Constraint(expr= -3 * m.x8 - 3 * m.x10 - 3 * m.x23 - 3 * m.x26 - 3 *
    m.x29 - 3 * m.x32 <= 0)
m.e14 = Constraint(expr= -26.1 * m.x8 - 14.8 * m.x17 - 14.8 * m.x20 - 26.1 *
    m.x23 - 26.1 * m.x26 <= 0)
m.e15 = Constraint(expr= -15.2 * m.x8 - 8.2 * m.x17 - 8.2 * m.x20 - 15.2 *
    m.x23 - 15.2 * m.x26 <= 0)
m.e16 = Constraint(expr= 0.12 * m.x8 - 0.01 * m.x10 - 0.08 * m.x17 - 0.08 *
    m.x20 + 0.12 * m.x23 + 0.12 * m.x26 - 0.01 * m.x29 - 0.01 * m.x32 <= 0)
m.e17 = Constraint(expr= 7.09999999999999 * m.x8 - 19 * m.x10 - 4.2 * m.x17 -
    4.2 * m.x20 + 7.09999999999999 * m.x23 + 7.09999999999999 * m.x26 - 19 *
    m.x29 - 19 * m.x32 <= 0)
m.e18 = Constraint(expr= 1.5 * m.x8 - 13.7 * m.x10 - 5.5 * m.x17 - 5.5 * m.x20
    + 1.5 * m.x23 + 1.5 * m.x26 - 13.7 * m.x29 - 13.7 * m.x32 <= 0)
m.e19 = Constraint(expr= 0.0299999999999999 * m.x7 + 0.0299999999999999 * m.x18
    + 0.0299999999999999 * m.x21 - 0.17 * m.x24 - 0.17 * m.x27 - 0.04 * m.x30
    - 0.04 * m.x33 <= 0)
m.e20 = Constraint(expr= 2.1 * m.x7 + 2.1 * m.x18 + 2.1 * m.x21 - 0.9 * m.x24
    - 0.9 * m.x27 - 0.9 * m.x30 - 0.9 * m.x33 <= 0)
m.e21 = Constraint(expr= -14.8 * m.x7 - 14.8 * m.x18 - 14.8 * m.x21 - 26.1 *
    m.x24 - 26.1 * m.x27 <= 0)
m.e22 = Constraint(expr= -8.2 * m.x7 - 8.2 * m.x18 - 8.2 * m.x21 - 15.2 * m.x24
    - 15.2 * m.x27 <= 0)
m.e23 = Constraint(expr= -0.08 * m.x7 - 0.08 * m.x18 - 0.08 * m.x21 + 0.12 *
    m.x24 + 0.12 * m.x27 - 0.01 * m.x30 - 0.01 * m.x33 <= 0)
m.e24 = Constraint(expr= -3.2 * m.x7 - 3.2 * m.x18 - 3.2 * m.x21 +
    8.09999999999999 * m.x24 + 8.09999999999999 * m.x27 - 18 * m.x30 - 18 *
    m.x33 <= 0)
m.e25 = Constraint(expr= -2.5 * m.x7 - 2.5 * m.x18 - 2.5 * m.x21 + 4.5 * m.x24
    + 4.5 * m.x27 - 10.7 * m.x30 - 10.7 * m.x33 <= 0)
m.e26 = Constraint(expr= -0.17 * m.x9 + 0.0299999999999999 * m.x19 +
    0.0299999999999999 * m.x22 - 0.17 * m.x25 - 0.17 * m.x28 - 0.04 * m.x31 -
    0.04 * m.x34 <= 0)
m.e27 = Constraint(expr= -3 * m.x9 - 3 * m.x25 - 3 * m.x28 - 3 * m.x31 - 3 *
    m.x34 <= 0)
m.e28 = Constraint(expr= -26.1 * m.x9 - 14.8 * m.x19 - 14.8 * m.x22 - 26.1 *
    m.x25 - 26.1 * m.x28 <= 0)
m.e29 = Constraint(expr= -15.2 * m.x9 - 8.2 * m.x19 - 8.2 * m.x22 - 15.2 *
    m.x25 - 15.2 * m.x28 <= 0)
m.e30 = Constraint(expr= 0.12 * m.x9 - 0.08 * m.x19 - 0.08 * m.x22 + 0.12 *
    m.x25 + 0.12 * m.x28 - 0.01 * m.x31 - 0.01 * m.x34 <= 0)
m.e31 = Constraint(expr= 3.09999999999999 * m.x9 - 8.2 * m.x19 - 8.2 * m.x22 +
    3.09999999999999 * m.x25 + 3.09999999999999 * m.x28 - 23 * m.x31 - 23 *
    m.x34 <= 0)
m.e32 = Constraint(expr= -7 * m.x19 - 7 * m.x22 - 15.2 * m.x31 - 15.2 * m.x34
    <= 0)
m.e33 = Constraint(expr= m.x1 + m.x3 + m.x5 == 1)
m.e34 = Constraint(expr= m.x2 + m.x4 + m.x6 == 1)
m.e35 = Constraint(expr= -m.x1 * m.x11 + m.x17 == 0)
m.e36 = Constraint(expr= -m.x1 * m.x12 + m.x18 == 0)
m.e37 = Constraint(expr= -m.x1 * m.x13 + m.x19 == 0)
m.e38 = Constraint(expr= -m.x2 * m.x14 + m.x20 == 0)
m.e39 = Constraint(expr= -m.x2 * m.x15 + m.x21 == 0)
m.e40 = Constraint(expr= -m.x2 * m.x16 + m.x22 == 0)
m.e41 = Constraint(expr= -m.x3 * m.x11 + m.x23 == 0)
m.e42 = Constraint(expr= -m.x3 * m.x12 + m.x24 == 0)
m.e43 = Constraint(expr= -m.x3 * m.x13 + m.x25 == 0)
m.e44 = Constraint(expr= -m.x4 * m.x14 + m.x26 == 0)
m.e45 = Constraint(expr= -m.x4 * m.x15 + m.x27 == 0)
m.e46 = Constraint(expr= -m.x4 * m.x16 + m.x28 == 0)
m.e47 = Constraint(expr= -m.x5 * m.x11 + m.x29 == 0)
m.e48 = Constraint(expr= -m.x5 * m.x12 + m.x30 == 0)
m.e49 = Constraint(expr= -m.x5 * m.x13 + m.x31 == 0)
m.e50 = Constraint(expr= -m.x6 * m.x14 + m.x32 == 0)
m.e51 = Constraint(expr= -m.x6 * m.x15 + m.x33 == 0)
m.e52 = Constraint(expr= -m.x6 * m.x16 + m.x34 == 0)
