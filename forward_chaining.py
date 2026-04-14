facts = {"rain"}

rules =[
    ({"rain"}, "wet_ground"),
    ({"wet ground"}, "slippery")

]
print("Initial Facts:", facts)

while True:
  new_fact = False

  for condition,conclusion in rules:
    if condition.issubset(facts) and conclusion not in facts:
      facts.add(conclusion)
      print("new fact delivered", conclusion)
      new_fact = True
  if not new_fact:
    break
print("Final Facts:", facts)
