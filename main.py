import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

df = pd.read_csv("data.csv")

# Filter by professional developers
df = df[df["Professional"] == "Professional developer"]

# Nominal attributes used for data analysis
attributes = [
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

df_nominal = df[attributes].dropna()

# Draw histograms (attribute values distribution)
for attr in attributes:
    attr_counts = df_nominal[attr].value_counts().head(20)
    plt.figure(figsize = (20,20))
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
corr_df = pd.DataFrame(corr_matrix, index=attributes, columns=attributes)

plt.figure(figsize = (16,16))
sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Macierz korelacji metodą Spearmana")
plt.tight_layout()
plt.savefig(f"figures/spearman_corr.png")
plt.show()
