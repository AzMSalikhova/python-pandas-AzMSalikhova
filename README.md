[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YJUfykA1)
# Python_pandas
Задание на агрегацию и обработку файлов с данными

 # Задание
**Медицинские диагностические устройства в клиниках.**
У медицинских клиник существуют чувствительные устройства, для которых важно плановое и своевременное обслуживание, для чего необходимо отслеживать статусы устройств.

**Необходимо считать данные из файла xlsx и произвести следующие действия:**
* Отфильтровать данные по гарантии
* Найти клиники с наибольшим количеством проблем
* Построить отчёт по срокам калибровки
* Сагрегировать данные по клиникам и оборудованию и составить сводную таблицу

Поля в таблице:
* device_id – уникальный идентификатор устройства
* clinic_id – уникальный идентификатор клиники, где установлено или планируется установить устройство
* clinic_name – название клиники, в которой используется оборудование
* city – город, где установлен аппарат (всего 15 городов)
* department – медицинское отделение клиники, в котором используется оборудование
* model – модель устройства (всего 6 моделей)
* serial_number – серийный номер устройства
* install_date – дата установки оборудования в клинике (Если устройство ещё не установлено, дата может быть в будущем, форматы дат могут отличаться)
* status – текущий статус устройства (planned_installation – устройство запланировано к установке, operational – устройство работает, maintenance_scheduled – запланировано техническое обслуживание, faulty – устройство неисправно). В данных могут встречаться варианты написания (например OK, op, broken), которые нужно нормализовать.
* warranty_until – дата окончания гарантии производителя, после этой даты ремонт может выполняться на платной основе.
* last_calibration_date – дата последней калибровки оборудования (значение может отсутствовать, дата может быть ошибочной (раньше даты установки))
* last_service_date – дата последнего технического обслуживания (когда проводилось обслуживание, какие устройства требуют сервисной проверки)
* issues_reported_12mo – количество зарегистрированных проблем за последние 12 месяцев
* failure_count_12mo – количество отказов устройства за последние 12 месяцев
* uptime_pct – процент времени, в течение которого устройство было работоспособным
* issues_text – текстовое описание некоторых проблем, зарегистрированных в работе устройства

# Задача
- В качестве практической работы необходимо нарисовать блок-схемы работы алгоритма решения задач и загрузить в свой репозиторий в течение дня.
- В качестве домашней работы необходимо реализовать программы по ранее созданным алгоритмам решения задач на языке Python и загрузить в свой репозиторий до крайнего срока.

Даты сдачи оговариваются в канале группы.

# Теоретическая справка

# 1. Что такое XLSX

XLSX — это формат таблиц Microsoft Excel.

Таблица состоит из:

- строк (rows)
- столбцов (columns)
- ячеек (cells)
- листов (sheets)

Пример таблицы:

| Name | Age | Salary |
|-----|-----|------|
| Alex | 25 | 3000 |
| Maria | 30 | 4200 |

В **pandas** такая структура называется **DataFrame**.

---

# 2. Установка библиотек

```bash
pip install pandas numpy openpyxl
```

Библиотеки:

- **pandas** — работа с таблицами
- **numpy** — быстрые математические операции
- **openpyxl** — чтение и запись Excel файлов

---

# 3. Основные структуры pandas

## DataFrame

Таблица данных.

```python
import pandas as pd


df = pd.DataFrame({
    "Name": ["Alex", "Maria"],
    "Age": [25, 30]
})
```

Результат:

```
    Name   Age
0   Alex   25
1  Maria   30
```

## Series

Отдельный столбец таблицы.

```python
df["Age"]
```

---

# 4. Чтение Excel файла

Основная функция:

```python
pd.read_excel()
```

Пример:

```python
import pandas as pd


df = pd.read_excel("data.xlsx")

print(df)
```

---

# 5. Чтение конкретного листа

```python
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
```

или

```python
df = pd.read_excel("data.xlsx", sheet_name=0)
```

---

# 6. Чтение всех листов

```python
sheets = pd.read_excel("data.xlsx", sheet_name=None)
```

Результат — словарь:

```
{
 "Sheet1": DataFrame,
 "Sheet2": DataFrame
}
```

Использование:

```python
df1 = sheets["Sheet1"]
```

---

# 7. Запись Excel файла

```python
df.to_excel("result.xlsx")
```

Без индекса:

```python
df.to_excel("result.xlsx", index=False)
```

---

# 8. Запись нескольких листов

```python
with pd.ExcelWriter("file.xlsx") as writer:
    df1.to_excel(writer, sheet_name="Users")
    df2.to_excel(writer, sheet_name="Products")
```

---

# 9. Просмотр данных

## Первые строки

```python
df.head()
```

```python
df.head(10)
```

## Последние строки

```python
df.tail()
```

## Информация о таблице

```python
df.info()
```

## Статистика

```python
df.describe()
```

---

# 10. Выбор данных

## Один столбец

```python
df["Age"]
```

## Несколько столбцов

```python
df[["Name", "Age"]]
```

## По строкам

```python
df.iloc[0]
```

---

# 11. Фильтрация данных

```python
df[df["Age"] > 25]
```

Несколько условий:

```python
df[(df["Age"] > 25) & (df["Salary"] > 3000)]
```

---

# 12. Изменение данных

Добавление столбца:

```python
df["Bonus"] = df["Salary"] * 0.1
```

Изменение значений:

```python
df["Age"] = df["Age"] + 1
```

---

# 13. Работа с пропущенными значениями

Проверка:

```python
df.isnull()
```

Удаление строк:

```python
df.dropna()
```

Замена:

```python
df.fillna(0)
```

---

# 14. Использование NumPy

```python
import numpy as np
```

Пример вычислений:

```python
df["LogSalary"] = np.log(df["Salary"])
```

Преобразование столбца в массив:

```python
arr = df["Salary"].to_numpy()
```

Основные функции:

```python
np.mean(arr)
np.max(arr)
np.min(arr)
np.std(arr)
```

---

# 15. Группировка данных

Пример таблицы:

| Department | Salary |
|------|------|
| IT | 4000 |
| IT | 5000 |
| HR | 3000 |

Группировка:

```python
df.groupby("Department").mean()
```

---

# 16. Сортировка

```python
df.sort_values("Salary")
```

По убыванию:

```python
df.sort_values("Salary", ascending=False)
```

---

# 17. Практический пример

Excel файл `sales.xlsx`:

| Product | Price | Quantity |
|--------|------|------|
| Laptop | 1000 | 5 |
| Mouse | 20 | 50 |

Python код:

```python
import pandas as pd
import numpy as np


df = pd.read_excel("sales.xlsx")


df["Revenue"] = df["Price"] * df["Quantity"]


total = np.sum(df["Revenue"])


print(df)
print("Total revenue:", total)


df.to_excel("report.xlsx", index=False)
```

---

# 18. Типичный workflow

Обычно работа с Excel выглядит так:

1. Загрузка файла

```
pd.read_excel()
```

2. Очистка данных

```
dropna()
fillna()
```

3. Обработка

```
filter
sort
groupby
```

4. Вычисления

```
numpy
```

5. Сохранение

```
to_excel()
```

---

# Полезные функции pandas

```
read_excel()
to_excel()
head()
info()
describe()
groupby()
sort_values()
dropna()
fillna()
merge()
concat()
```

