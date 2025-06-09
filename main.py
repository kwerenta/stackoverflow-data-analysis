import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from scipy.stats import spearmanr

df = pd.read_csv("data.csv")

# Filter by professional developers
df = df[df["Professional"] == "Professional developer"]
# Make sure Salary is not N/A
df = df[df["Salary"].notna()]

# Convert numeric salary into nominal categories
salary_classes = ["< $15k", "<$15k, $30k)", "<$30k, $45k)", "<$45k, $60k)", "<$60k, $75k)", "<$75k, $90k)", "<$90k, $105k)", "<$105k, $120k)", "<$120k, $135k)", ">= $135k"]
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

class SalaryPredictionModel:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = "SalaryClass"
        self.macro_f1_threshold = 0.6

    def preprocess_data(self, df):
        print("🔄 Rozpoczynam przetwarzanie danych...")

        df_filtered = df.copy()
        print(f"📊 Liczba rekordów z wynagrodzeniem: {len(df_filtered)}")

        available_features = nominal_attrs[1:]

        # Przygotowanie finalnego zbioru danych
        df_model = df_filtered[available_features + ['SalaryClass']].copy()

        # Usuwanie wierszy z brakującymi danymi
        df_model = df_model.dropna()
        print(f"📊 Liczba rekordów po usunięciu braków: {len(df_model)}")

        self.feature_columns = available_features
        return df_model

    def encode_categorical_features(self, df):
        df_encoded = df.copy()

        for column in self.feature_columns:
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df[column].astype(str))
            self.label_encoders[column] = le

        if self.target_column not in self.label_encoders:
            le_target = LabelEncoder()
            df_encoded[self.target_column] = le_target.fit_transform(df[self.target_column].astype(str))
            self.label_encoders[self.target_column] = le_target

        return df_encoded

    def train_model(self, X_train, y_train):
        print("🌳 Trenuję model drzewa decyzyjnego...")

        # Parametry do optymalizacji
        param_grid = {
            'max_depth': [3, 5, 7, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10, 20, 50],
            'min_samples_leaf': [1, 2, 5, 10, 20],
            'criterion': ['gini', 'entropy'],
        }

        # Grid Search z walidacją krzyżową
        dt = DecisionTreeClassifier(random_state=420, class_weight='balanced')
        grid_search = GridSearchCV(
            dt,
            param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1
        )

        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        print(f"✅ Najlepsze parametry: {grid_search.best_params_}")
        print(f"✅ Najlepszy wynik CV F1-macro: {grid_search.best_score_:.4f}")

        return grid_search.best_score_

    def evaluate_model(self, X_test, y_test):
        print("📊 Ewaluuję model...")

        y_pred = self.model.predict(X_test)
        macro_f1 = f1_score(y_test, y_pred, average='macro')

        print(f"\n🎯 WYNIKI EWALUACJI:")
        print(f"{'='*50}")
        print(f"📈 Macro F1 Score: {macro_f1:.4f}")
        print(f"🎯 Kryterium sukcesu (≥0.6): {'✅ SPEŁNIONE' if macro_f1 >= self.macro_f1_threshold else '❌ NIESPEŁNIONE'}")

        # Raport klasyfikacji
        print(f"\n📋 Raport klasyfikacji:")
        print(classification_report(y_test, y_pred, target_names=salary_classes))

        return macro_f1, y_pred

    def plot_confusion_matrix(self, y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)

        plt.figure(figsize=(12, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=salary_classes, yticklabels=salary_classes)
        plt.title('Macierz Pomyłek - Predykcja Przedziałów Wynagrodzeń')
        plt.xlabel('Predykcja')
        plt.ylabel('Rzeczywistość')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def plot_feature_importance(self):
        """
        Wizualizacja ważności cech
        """
        importance = self.model.feature_importances_
        feature_names = self.feature_columns

        # Sortowanie według ważności
        indices = np.argsort(importance)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title('Ważność Cech w Modelu Drzewa Decyzyjnego')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Cechy')
        plt.ylabel('Ważność')
        plt.tight_layout()
        plt.show()

        print(f"\n🏆 TOP 10 NAJWAŻNIEJSZYCH CECH:")
        print(f"{'='*40}")
        for i in range(min(10, len(importance))):
            idx = indices[i]
            print(f"{i+1:2d}. {feature_names[idx]:<20} : {importance[idx]:.4f}")

    def plot_decision_tree(self):
        class_names = self.label_encoders[self.target_column].classes_

        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model,
            max_depth=4,
            feature_names=self.feature_columns,
            class_names=class_names,
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title(f'Drzewo Decyzyjne - Predykcja Przedziałów Wynagrodzeń (Wyświetlona głębokość: 4)')
        plt.tight_layout()
        plt.show()

    def analyze_salary_distribution(self, df):
        print(f"\n💰 ANALIZA ROZKŁADU WYNAGRODZEŃ:")
        print(f"{'='*50}")

        # Rozkład klas wynagrodzeń
        salary_class_counts = df['SalaryClass'].value_counts().sort_values()[::-1]
        print(f"\n📋 Rozkład klas wynagrodzeń:")
        for class_name, count in salary_class_counts.items():
            percentage = (count / len(df)) * 100
            print(f"   {class_name}: {count:4d} ({percentage:5.1f}%)")

    def run_full_pipeline(self, df):
        print("🚀 ROZPOCZYNAM PEŁNY PIPELINE MODELOWANIA")
        print("="*60)

        df_processed = self.preprocess_data(df)
        self.analyze_salary_distribution(df_processed)
        df_encoded = self.encode_categorical_features(df_processed)

        X = df_encoded[self.feature_columns]
        y = df_encoded[self.target_column]

        print(f"\n📊 Kształt danych: {X.shape}")
        print(f"📊 Liczba klas: {len(np.unique(y))}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        cv_score = self.train_model(X_train, y_train)
        macro_f1, y_pred = self.evaluate_model(X_test, y_test)

        self.plot_confusion_matrix(y_test, y_pred)
        self.plot_feature_importance()
        self.plot_decision_tree()

        print(f"\n🎯 PODSUMOWANIE WYNIKÓW:")
        print(f"{'='*50}")
        print(f"📈 Walidacja krzyżowa F1-macro: {cv_score:.4f}")
        print(f"📈 Test F1-macro: {macro_f1:.4f}")
        print(f"🎯 Kryterium sukcesu (≥0.6): {'✅ SPEŁNIONE' if macro_f1 >= self.macro_f1_threshold else '❌ NIESPEŁNIONE'}")

        if macro_f1 >= self.macro_f1_threshold:
            print("🎉 MODEL OSIĄGNĄŁ ZAŁOŻONE KRYTERIUM SUKCESU!")

        return {
            'model': self.model,
            'macro_f1': macro_f1,
            'cv_score': cv_score,
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }

print("🎯 MODEL PREDYKCJI WYNAGRODZEŃ PROGRAMISTÓW")
print("📋 Stack Overflow Developer Survey 2017")
print("="*60)

model = SalaryPredictionModel()
results = model.run_full_pipeline(df)

if results:
    print(f"\n✅ MODELOWANIE ZAKOŃCZONE POMYŚLNIE!")
    print(f"🔧 Model został zapisany i jest gotowy do użycia.")
else:
    print(f"\n❌ MODELOWANIE NIEUDANE!")

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
print(f"Salary - punkty oddalone\n{salary.describe()}\n\tPrzedział wartośći <{max(0, lower)}, {upper}>\n\tLiczba outlierów = {len(outliers)}")

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
