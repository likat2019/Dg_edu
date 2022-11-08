#создай здесь свой индивидуальный проект!
import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv('train.csv')
df.drop(['id', 'bdate', 'has_photo', 'followers_count', 'graduation', 'relation', 'life_main', 'people_main', 'city', 'last_seen', 'occupation_name', 'career_start', 'career_end'], axis=1, inplace=True)
df.info()
print(df.head())

print('Гипотеза №1 - телефонов у девушек больше чем у мужчин/ (1-девушки, 2-парни)')
ph = round(df.groupby('sex')['has_mobile'].mean(), 2)
print(ph)
ph.plot(kind = 'pie')
plt.show()

print('Гипотеза №2- девушки брали курс чаще чем мужчины/ (1-девушки, 2-парни)')
lessons = round(df.groupby('sex')['result'].mean(), 2)
print(lessons)
lessons.plot(kind = 'pie')
plt.show()

def sex_apply(sex):
    if sex == 2:
        return 0
    return 1
df['sex'] = df['sex'].apply(sex_apply)

def edu_form_apply(edu_form):
    if edu_form == 'Distance Learning':
        return 'Distance Learning'
    elif edu_form == 'Part-time':
        return 'Part-time'
    else:
        return 'Full-time'

df['education_form'] = df['education_form'].apply(edu_form_apply)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop(['education_form'], axis=1, inplace=True)


def edu_status_apply(edu_status):
    if edu_status == 'Undergraduate applicant':
        return 0
    elif edu_status == "Student (Master's)" or edu_status == "Student (Bachelor's)" or edu_status == "Student (Specialist)":
        return 1
    elif edu_status == "Alumnus (Specialist)" or edu_status == "Alumnus (Master's)" or edu_status == "Alumnus (Bachelor's)":
        return 2
    else:
        return 3
df['education_status'] = df['education_status'].apply(edu_status_apply)

def langs_apply(langs):
    langs = langs.split(';')
    if 'Русский' in langs:
        return 1
    else:
        return 0
df['langs'] = df['langs'].apply(langs_apply)

def ocu_type_apply(ocu_type):
    if ocu_type == 'work':
        return 0
    else:
        return 1
df['occupation_type'] = df['occupation_type'].apply(ocu_type_apply)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

x = df.drop('result', axis = 1)
y = df['result']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(y_test)
print(y_pred)
print('Процент правильно предсказанных исходов:', round(accuracy_score(y_test, y_pred)*100, 2))