import pandas as pd
import matplotlib.pyplot as plt

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

# Draw histograms (attribute values distribution)
for attr in attributes:
    attr_counts = df[attr].value_counts().head(20)
    plt.figure(figsize = (20,20))
    plt.bar(attr_counts.index, attr_counts.values)
    plt.xlabel(attr)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel("Liczba próbek")
    plt.title(f"Histogram atrybutu {attr}")
    plt.tight_layout()
    plt.savefig(f"figures/histograms/{attr}.png")
    plt.show()

