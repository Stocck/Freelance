import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# №1 Считываю с файла titanic.csv
data = pd.read_csv('titanic.csv')

# №2 Оставляю в выборке признаки и итоговое значиение (Итоговое значение нужно для дальнейшей обработки DataFrame)
data = data.loc[:, ['Pclass', 'Fare', 'Age', 'Sex', 'Survived']]

# №3 Обращаю внимание, что Sex строковый тип данных
# №3.1 разбиваю с помощью pd.get_dummies(data) на два столбца со значениями True/False
data = pd.get_dummies(data)

# №3.2 Так как Sex_male являются обратным столбцом Sex_female, можно избавится от него
data = data.drop('Sex_male', 1)
data = data.rename(columns={'Sex_female': 'Sex'})

# №5 (пришлось решить 5 раньше 4,так проще) Удаляем строки с nan
data = data.dropna()

# №4 Отделяю выборку от целевой переменной
X_data = data.drop('Survived', 1)
y_data = data.Survived

# №6 Создаю и Обучаю дерево
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X_data, y_data)

# №7.1 С помощью метода feature_importances_ нахожу важность признаков
feat_importance = clf.feature_importances_

# №7.2 Обединяю в один массив важность признаков и их названия
arr = [[feat_importance[i], X_data.columns[i]] for i in range(len(X_data.columns))]

# №7.3 Сортиую по убыванию и вывожу список самых важных признаков
arr.sort(key=lambda x: -x[0])
for i in arr:
    print(i[1], i[0])
