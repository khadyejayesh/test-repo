def loan_approval(income, credit_score):
  if income > 50000 and credit_score > 700:
    return "Loan Approved"
  elif income > 30000 and credit_score > 650:
    return "Loan under reviewed"
  else:
    return "Loan Denied"

income = 45000
credit_score = 680

print("Income:", income)
print("Credit Score:", credit_score)
print("Decision:", loan_approval(income, credit_score))
