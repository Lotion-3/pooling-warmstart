# NLP written by GAMS Convert at 02/17/22 17:21:48
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#        34       18        0       16        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        36       36        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#       136      104       32
#
# Reformulation has removed 1 variable and 1 equation

from pyomo.environ import *

model = m = ConcreteModel()

m.x1 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x2 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x3 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x4 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x5 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x6 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x7 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x8 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x9 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x10 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x11 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x12 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x13 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x14 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x15 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x16 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x17 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x18 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x19 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x20 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x21 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x22 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x23 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x24 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x25 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x26 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x27 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x28 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x29 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x30 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x31 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x32 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x33 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x34 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x35 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x36 = Var(within=Reals, bounds=(0,200), initialize=0)

m.obj = Objective(sense=minimize, expr= m.x5 - 5 * m.x6 + 4 * m.x7 - 2 * m.x8
    - 2 * m.x9 - 8 * m.x10 + m.x11 - 5 * m.x12 - 3 * m.x21 - 9 * m.x22 - 6 *
    m.x24 + 7 * m.x25 + m.x26 + 10 * m.x27 + 4 * m.x28 - 6 * m.x29 - 12 * m.x30
    - 3 * m.x31 - 9 * m.x32 + 4 * m.x33 - 2 * m.x34 + 7 * m.x35 + m.x36)

m.e1 = Constraint(expr= m.x21 + m.x22 + m.x23 + m.x24 <= 600)
m.e2 = Constraint(expr= m.x25 + m.x26 + m.x27 + m.x28 <= 600)
m.e3 = Constraint(expr= m.x5 + m.x6 + m.x7 + m.x8 <= 600)
m.e4 = Constraint(expr= m.x29 + m.x30 + m.x31 + m.x32 <= 600)
m.e5 = Constraint(expr= m.x33 + m.x34 + m.x35 + m.x36 <= 600)
m.e6 = Constraint(expr= m.x9 + m.x10 + m.x11 + m.x12 <= 600)
m.e7 = Constraint(expr= m.x21 + m.x22 + m.x23 + m.x24 + m.x25 + m.x26 + m.x27
    + m.x28 <= 600)
m.e8 = Constraint(expr= m.x29 + m.x30 + m.x31 + m.x32 + m.x33 + m.x34 + m.x35
    + m.x36 <= 600)
m.e9 = Constraint(expr= m.x5 + m.x9 + m.x21 + m.x25 + m.x29 + m.x33 <= 100)
m.e10 = Constraint(expr= m.x6 + m.x10 + m.x22 + m.x26 + m.x30 + m.x34 <= 200)
m.e11 = Constraint(expr= m.x7 + m.x11 + m.x23 + m.x27 + m.x31 + m.x35 <= 100)
m.e12 = Constraint(expr= m.x8 + m.x12 + m.x24 + m.x28 + m.x32 + m.x36 <= 200)
m.e13 = Constraint(expr= -0.5 * m.x5 + 0.5 * m.x21 - 1.5 * m.x25 + m.x29 -
    m.x33 <= 0)
m.e14 = Constraint(expr= 0.5 * m.x6 + m.x10 + 1.5 * m.x22 - 0.5 * m.x26 + 2 *
    m.x30 <= 0)
m.e15 = Constraint(expr= -m.x7 - 0.5 * m.x11 - 2 * m.x27 + 0.5 * m.x31 - 1.5 *
    m.x35 <= 0)
m.e16 = Constraint(expr= 0.5 * m.x12 + m.x24 - m.x28 + 1.5 * m.x32 - 0.5 *
    m.x36 <= 0)
m.e17 = Constraint(expr= m.x1 + m.x2 == 1)
m.e18 = Constraint(expr= m.x3 + m.x4 == 1)
m.e19 = Constraint(expr= -m.x1 * m.x13 + m.x21 == 0)
m.e20 = Constraint(expr= -m.x1 * m.x14 + m.x22 == 0)
m.e21 = Constraint(expr= -m.x1 * m.x15 + m.x23 == 0)
m.e22 = Constraint(expr= -m.x1 * m.x16 + m.x24 == 0)
m.e23 = Constraint(expr= -m.x2 * m.x13 + m.x25 == 0)
m.e24 = Constraint(expr= -m.x2 * m.x14 + m.x26 == 0)
m.e25 = Constraint(expr= -m.x2 * m.x15 + m.x27 == 0)
m.e26 = Constraint(expr= -m.x2 * m.x16 + m.x28 == 0)
m.e27 = Constraint(expr= -m.x3 * m.x17 + m.x29 == 0)
m.e28 = Constraint(expr= -m.x3 * m.x18 + m.x30 == 0)
m.e29 = Constraint(expr= -m.x3 * m.x19 + m.x31 == 0)
m.e30 = Constraint(expr= -m.x3 * m.x20 + m.x32 == 0)
m.e31 = Constraint(expr= -m.x4 * m.x17 + m.x33 == 0)
m.e32 = Constraint(expr= -m.x4 * m.x18 + m.x34 == 0)
m.e33 = Constraint(expr= -m.x4 * m.x19 + m.x35 == 0)
m.e34 = Constraint(expr= -m.x4 * m.x20 + m.x36 == 0)
