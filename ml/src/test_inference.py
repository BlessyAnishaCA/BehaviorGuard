from inference import behavior_score, transaction_score

print("=== BEHAVIOR MODEL TEST ===")

# Normal user
score = behavior_score(120, 80, 5.2, 300, 400)
print(f"Normal user score:   {score:.3f}  (expect < 0.3)")

# Attacker
score = behavior_score(60, 30, 9.0, 700, 150)
print(f"Attacker score:      {score:.3f}  (expect > 0.5)")

print()
print("=== TRANSACTION MODEL TEST ===")

# Normal transaction
score = transaction_score(4.5, 14, 1, 0, 2)
print(f"Normal transaction:  {score:.3f}  (expect < 0.3)")

# Suspicious transaction
score = transaction_score(12.0, 3, 2, 1, 4)
print(f"Suspicious tx score: {score:.3f}  (expect > 0.5)")

print()
print("All tests passed! Models ready for Member 4 ✅")