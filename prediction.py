import joblib
import numpy as np

bundle = joblib.load("model.pkl")
model = bundle["model"]
le1   = bundle["le1"]

print("=" * 40)
print("  Предсказание потребления энергии")
print("=" * 40)

# Тип здания
while True:
    building = input("\nТип здания (Residential / Commercial / Industrial): ").strip().capitalize()
    try:
        float(building)
        print("Ошибка: введите слово, а не число!")
        continue
    except ValueError:
        pass
    if building not in le1.classes_:
        print(f"Ошибка: введите одно из {list(le1.classes_)}")
    else:
        break

# Площадь
while True:
    try:
        sqft = float(input("Площадь (кв. футы):            "))
        break
    except ValueError:
        print("Ошибка: введите число!")

# Количество жителей
while True:
    try:
        occupants = float(input("Количество жителей:            "))
        break
    except ValueError:
        print("Ошибка: введите число!")

# Количество приборов
while True:
    try:
        appliances = float(input("Количество приборов:           "))
        break
    except ValueError:
        print("Ошибка: введите число!")

# Температура
while True:
    try:
        temp = float(input("Средняя температура (°C):      "))
        break
    except ValueError:
        print("Ошибка: введите число!")

# Предсказание
b = le1.transform([building])[0]
X = np.array([[b, sqft, occupants, appliances, temp]])
result = model.predict(X)[0]

print("\n" + "=" * 40)
print("  РЕЗУЛЬТАТ")
print("=" * 40)
print(f"  Тип здания   : {building}")
print(f"  Площадь      : {sqft:.0f} кв. футов")
print(f"  Жители       : {occupants:.0f} чел.")
print(f"  Приборы      : {appliances:.0f} шт.")
print(f"  Температура  : {temp:.1f} °C")
print("-" * 40)
print(f"  ⚡ Потребление: {result:.2f} кВт·ч")
print("=" * 40)