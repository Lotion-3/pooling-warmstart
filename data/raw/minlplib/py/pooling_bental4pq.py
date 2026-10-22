# NLP written by GAMS Convert at 02/17/22 17:21:47
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#        16        7        0        9        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        13       13        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        51       39       12
#
# Reformulation has removed 1 variable and 1 equation

from pyomo.environ import *

model = m = ConcreteModel()

m.x1 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x2 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x3 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x4 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x5 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x6 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x7 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x8 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x9 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x10 = Var(within=Reals, bounds=(0,50), initialize=0)
m.x11 = Var(within=Reals, bounds=(0,50), initialize=0)
m.x12 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x13 = Var(within=Reals, bounds=(0,200), initialize=0)

m.obj = Objective(sense=minimize, expr= m.x4 - 5 * m.x5 - 3 * m.x8 - 9 * m.x9
    + 6 * m.x10 + 7 * m.x12 + m.x13)

m.e1 = Constraint(expr= m.x8 + m.x9 <= 300)
m.e2 = Constraint(expr= m.x10 + m.x11 <= 50)
m.e3 = Constraint(expr= m.x12 + m.x13 <= 300)
m.e4 = Constraint(expr= m.x4 + m.x5 <= 300)
m.e5 = Constraint(expr= m.x8 + m.x9 + m.x10 + m.x11 + m.x12 + m.x13 <= 300)
m.e6 = Constraint(expr= m.x4 + m.x8 + m.x10 + m.x12 <= 100)
m.e7 = Constraint(expr= m.x5 + m.x9 + m.x11 + m.x13 <= 200)
m.e8 = Constraint(expr= -0.5 * m.x4 + 0.5 * m.x8 - 1.5 * m.x10 - 1.5 * m.x12
    <= 0)
m.e9 = Constraint(expr= 0.5 * m.x5 + 1.5 * m.x9 - 0.5 * m.x11 - 0.5 * m.x13
    <= 0)
m.e10 = Constraint(expr= m.x1 + m.x2 + m.x3 == 1)
m.e11 = Constraint(expr= -m.x1 * m.x6 + m.x8 == 0)
m.e12 = Constraint(expr= -m.x1 * m.x7 + m.x9 == 0)
m.e13 = Constraint(expr= -m.x2 * m.x6 + m.x10 == 0)
m.e14 = Constraint(expr= -m.x2 * m.x7 + m.x11 == 0)
m.e15 = Constraint(expr= -m.x3 * m.x6 + m.x12 == 0)
m.e16 = Constraint(expr= -m.x3 * m.x7 + m.x13 == 0)
