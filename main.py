import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

df = pd.read_csv("data.csv")

# Filter by professional developers
df = df[df["Professional"] == "Professional developer"]
# Make sure Salary is not N/A
df = df[df["Salary"].notna()]

# Convert numeric salary into nominal categories
def salary_category(s):
    if s < 15000:
        return "< 15e3$"
    elif s < 30000:
        return "< 30e3$"
    elif s < 45000:
        return "< 45e3$"
    elif s < 60000:
        return "< 60e3$"
    elif s < 75000:
        return "< 75e3$"
    elif s < 90000:
        return "< 90e3$"
    elif s < 105000:
        return "< 105e3$"
    elif s < 120000:
        return "< 120e3$"
    elif s < 135000:
        return "< 135e3$"
    else:
        return ">= 135e3$"
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
    "HaveWorkedPlatform"
]

df_nominal = df[nominal_attrs].dropna()

# Draw histograms (attribute values distribution)
for attr in nominal_attrs:
    attr_counts = df_nominal[attr].value_counts().head(10)
    plt.figure(figsize = (16,12))
    plt.bar(attr_counts.index, attr_counts.values)
    plt.xlabel(attr)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Liczba próbek")
    plt.title(f"Histogram atrybutu {attr}")
    plt.tight_layout()
    plt.savefig(f"figures/histograms/{attr}.png")
    plt.show()

# Encode values of nominal attributes as integers
df_encoded = df_nominal.apply(lambda x: pd.factorize(x)[0])
corr_matrix = spearmanr(df_encoded)[0]
corr_df = pd.DataFrame(corr_matrix, index=nominal_attrs, columns=nominal_attrs)

plt.figure(figsize = (16,16))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Macierz korelacji metodą Spearmana")
plt.tight_layout()
plt.savefig(f"figures/spearman_corr.png")
plt.show()
