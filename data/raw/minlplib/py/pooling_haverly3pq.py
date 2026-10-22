# NLP written by GAMS Convert at 02/17/22 17:21:49
#
# Equation counts
#     Total        E        G        L        N        X        C        B
#        13        5        0        8        0        0        0        0
#
# Variable counts
#                  x        b        i      s1s      s2s       sc       si
#     Total     cont   binary  integer     sos1     sos2    scont     sint
#        10       10        0        0        0        0        0        0
# FX      0
#
# Nonzero counts
#     Total    const       NL
#        36       28        8
#
# Reformulation has removed 1 variable and 1 equation

from pyomo.environ import *

model = m = ConcreteModel()

m.x1 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x2 = Var(within=Reals, bounds=(0,1), initialize=0)
m.x3 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x4 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x5 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x6 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x7 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x8 = Var(within=Reals, bounds=(0,200), initialize=0)
m.x9 = Var(within=Reals, bounds=(0,100), initialize=0)
m.x10 = Var(within=Reals, bounds=(0,200), initialize=0)

m.obj = Objective(sense=minimize, expr= m.x3 - 5 * m.x4 - 3 * m.x7 - 9 * m.x8
    + 4 * m.x9 - 2 * m.x10)

m.e1 = Constraint(expr= m.x7 + m.x8 <= 300)
m.e2 = Constraint(expr= m.x9 + m.x10 <= 300)
m.e3 = Constraint(expr= m.x3 + m.x4 <= 300)
m.e4 = Constraint(expr= m.x7 + m.x8 + m.x9 + m.x10 <= 300)
m.e5 = Constraint(expr= m.x3 + m.x7 + m.x9 <= 100)
m.e6 = Constraint(expr= m.x4 + m.x8 + m.x10 <= 200)
m.e7 = Constraint(expr= -0.5 * m.x3 + 0.5 * m.x7 - 1.5 * m.x9 <= 0)
m.e8 = Constraint(expr= 0.5 * m.x4 + 1.5 * m.x8 - 0.5 * m.x10 <= 0)
m.e9 = Constraint(expr= m.x1 + m.x2 == 1)
m.e10 = Constraint(expr= -m.x1 * m.x5 + m.x7 == 0)
m.e11 = Constraint(expr= -m.x1 * m.x6 + m.x8 == 0)
m.e12 = Constraint(expr= -m.x2 * m.x5 + m.x9 == 0)
m.e13 = Constraint(expr= -m.x2 * m.x6 + m.x10 == 0)
