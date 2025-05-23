import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")

df = df[["Country", "Salary"]].dropna()
avg_salary_by_country = (
    df.groupby("Country")["Salary"].mean().sort_values(ascending=False).head(20)
)

plt.figure(figsize=(12, 8))
sns.barplot(x=avg_salary_by_country.values, y=avg_salary_by_country.index)
plt.xlabel("Średnie zarobki brutton (USD)")
plt.ylabel("Kraj")
plt.title("Top 20 krajów wg średnich zarobków")
plt.tight_layout()
plt.show()
