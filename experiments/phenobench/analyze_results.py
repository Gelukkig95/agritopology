from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

dates = ["05-15", "05-26", "06-05"]
weeds = [2862 + 1249, 2410 + 1097, 2869 + 1580]
risky = [80 + 32, 290 + 131, 886 + 494]

safe = [w - r for w, r in zip(weeds, risky)]
chi2, p, dof, expected = chi2_contingency([risky, safe])
print(f"Total weeds: {sum(weeds)}")
print(f"Chi-square: {chi2:.2f}, p-value: {p:.2e}")

pct = [r / w * 100 for r, w in zip(risky, weeds)]

plt.figure(figsize=(6, 4))
plt.bar(dates, pct, color=["green", "orange", "red"])
plt.ylabel("Weed instances contacting crop (%)")
plt.xlabel("Acquisition date (growth stage)")
plt.title("RCC8 Override Risk by Growth Stage\n(Train+Val, n=12,067)")
for i, v in enumerate(pct):
    plt.text(i, v + 1, f"{v:.1f}%", ha="center")
plt.savefig("risk_by_date.png", dpi=150)
plt.show()