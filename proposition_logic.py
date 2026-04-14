from sympy import symbols
from sympy.logic.boolalg import Implies, And
from sympy.logic.inference import satisfiable

P,Q,R = symbols('P Q R')

rule1= Implies(P,Q)
rule2= Implies(Q,R)
print("Rule 1: If P then Q", rule1)
print("Rule 2: If Q then R", rule2)



knowledge_base= And(rule1,rule2, P)

test = satisfiable(And(knowledge_base, ~R))
print("Can R be False", test)
