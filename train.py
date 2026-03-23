import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import joblib

# Загрузка данных
train = pd.read_csv("train_energy_data.csv")
test  = pd.read_csv("test_energy_data.csv")

# Кодировка текста в числа
le1 = LabelEncoder()
for df in [train, test]:
    df["Building Type"] = le1.fit_transform(df["Building Type"])

X_train = train.drop("Energy Consumption", axis=1)
y_train = train["Energy Consumption"]
X_test  = test.drop("Energy Consumption", axis=1)
y_test  = test["Energy Consumption"]

# обуч
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = r2_score(y_test, y_pred) * 100

# сохр. модели
joblib.dump({"model": model, "le1": le1}, "model.pkl")
print(f"Accuracy: {accuracy:.2f}%")
print("model.pkl сохранён")

plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, alpha=0.4, color="steelblue", label="Предсказания")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2, label="Идеал")
plt.xlabel("Реальное потребление (кВт·ч)")
plt.ylabel("Предсказанное потребление (кВт·ч)")
plt.title(f"Линейная регрессия — Energy Consumption\nAccuracy: {accuracy:.2f}%")
plt.legend()
plt.tight_layout()
plt.show()