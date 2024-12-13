#!/usr/bin/env python
# coding: utf-8

# In[1]:





# In[ ]:





# In[2]:


# Імпортуємо необхідні модулі
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Завантажуємо дані Iris
data = load_iris()

# Поділяємо дані на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, random_state=0
)

# Виводимо розміри отриманих наборів
print(f"Навчальна вибірка: {X_train.shape}, Мітки: {y_train.shape}")
print(f"Тестова вибірка: {X_test.shape}, Мітки: {y_test.shape}")

# Створюємо модель k-ближчих сусідів і навчаємо її
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

# Введення характеристик нової квітки для передбачення
print("\nВведіть характеристики нової квітки:")
sepal_len = float(input("Довжина чашолистка (см): "))
sepal_wid = float(input("Ширина чашолистка (см): "))
petal_len = float(input("Довжина пелюстки (см): "))
petal_wid = float(input("Ширина пелюстки (см): "))

# Формуємо новий зразок
new_sample = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
print(f"Форма нового зразка: {new_sample.shape}")

# Виконуємо передбачення для введених даних
predicted_class = model.predict(new_sample)
print(f"\nПередбачений клас: {predicted_class[0]}")
print(f"Вид квітки: {data.target_names[predicted_class]}")

# Оцінюємо модель на тестовій вибірці
accuracy = model.score(X_test, y_test)
print(f"\nТочність моделі на тестовій вибірці: {accuracy:.2f}")


# In[3]:


# Імпортуємо необхідні модулі
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Завантажуємо дані Iris
data = load_iris()

# Поділяємо дані на навчальну та тестову вибірки
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, random_state=0
)

# Виводимо розміри отриманих наборів
print(f"Навчальна вибірка: {X_train.shape}, Мітки: {y_train.shape}")
print(f"Тестова вибірка: {X_test.shape}, Мітки: {y_test.shape}")

# Створюємо модель k-ближчих сусідів і навчаємо її
model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_train, y_train)

# Введення характеристик нової квітки для передбачення
print("\nВведіть характеристики нової квітки:")
sepal_len = float(input("Довжина чашолистка (см): "))
sepal_wid = float(input("Ширина чашолистка (см): "))
petal_len = float(input("Довжина пелюстки (см): "))
petal_wid = float(input("Ширина пелюстки (см): "))

# Формуємо новий зразок
new_sample = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
print(f"Форма нового зразка: {new_sample.shape}")

# Виконуємо передбачення для введених даних
predicted_class = model.predict(new_sample)
print(f"\nПередбачений клас: {predicted_class[0]}")
print(f"Вид квітки: {data.target_names[predicted_class]}")

# Оцінюємо модель на тестовій вибірці
accuracy = model.score(X_test, y_test)
print(f"\nТочність моделі на тестовій вибірці: {accuracy:.2f}")


# In[ ]:




