import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figtext
from scipy.stats import spearmanr

df = pd.read_csv("data.csv")

# Filter by professional developers
df = df[df["Professional"] == "Professional developer"]
# Make sure Salary is not N/A
df = df[df["Salary"].notna()]

# Convert numeric salary into nominal categories
def salary_category(s):
    if s < 15000:
        return "< $15k"
    elif s < 30000:
        return "<$15k, $30k)"
    elif s < 45000:
        return "<$30k, $45k)"
    elif s < 60000:
        return "<$45k, $60k)"
    elif s < 75000:
        return "<$60k, $75k)"
    elif s < 90000:
        return "<$75k, $90k)"
    elif s < 105000:
        return "<$90k, $105k)"
    elif s < 120000:
        return "<$105k, $120k)"
    elif s < 135000:
        return "<$120k, $135k)"
    else:
        return ">= 135k$"
df['SalaryClass'] = df['Salary'].apply(salary_category)

# Nominal attributes used for data analysis
nominal_attrs = [
    "SalaryClass",
    "Gender",
    "Country",
    "YearsCodedJob",
    "YearsProgram",
    "FormalEducation",
    "MajorUndergrad",
    "DeveloperType",
    "CompanySize",
    "EmploymentStatus",
    "HaveWorkedLanguage",
    "HaveWorkedFramework",
    "HaveWorkedDatabase",
    "HaveWorkedPlatform",
    "HomeRemote",
    "CompanyType",
    "Race"
]

df_nominal = df[nominal_attrs].dropna()

# Draw histograms (attribute values distribution)
for attr in nominal_attrs:
    attr_counts = df_nominal[attr].value_counts().head(10)
    plt.figure(figsize = (12,8))
    plt.bar(attr_counts.index, attr_counts.values)
    plt.xlabel(attr)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Liczba próbek")
    plt.title(f"Histogram atrybutu {attr}")
    plt.tight_layout()
    plt.savefig(f"figures/histograms/{attr}.png")
    plt.show()

# Calculate salary outliers
valid_idx = df_nominal.dropna().index.intersection(df.index)
salary = df.loc[valid_idx, ["Salary"]]["Salary"]
plt.figure(figsize=(12, 8))
sns.boxplot(x=salary.values)
plt.xlabel("Salary")
plt.title(f"Punkty oddalone atrybutu Salary")
plt.savefig(f"figures/boxplot_Salary.png")
plt.show()

Q1 = salary.quantile(0.25)
Q3 = salary.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = salary[(salary < lower) | (salary > upper)]
print(f"Salary - punkty oddalone\n\t{salary.describe()}\n\tLiczba outlierów = {len(outliers)}")

# Encode values of nominal attributes as integers
df_encoded = df_nominal.apply(lambda x: pd.factorize(x)[0])
corr_matrix = spearmanr(df_encoded)[0]
corr_df = pd.DataFrame(corr_matrix, index=nominal_attrs, columns=nominal_attrs)

plt.figure(figsize = (12,12))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Macierz korelacji metodą Spearmana")
plt.tight_layout()
plt.savefig(f"figures/spearman_corr.png")
plt.show()
