# Practical_04
# Prior probability of having disease
P_D= 0.01

# Probability of symptom given disease

P_S_given_D=0.9

#Overall probability of symptom

P_S= 0.05

# Bayes Theorem

P_D_given_S=(P_S_given_D*P_D)/P_S

print("Probability of Disease given Symptom:", round(P_D_given_S, 4))
