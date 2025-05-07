
'''ОЧЕНЬ ОЧЕНЬ ВАЖНО
ЗНАЧЕНИЯ НАЧАЛА И КОНЦА ВЫБОРКИ ДОЛЖНЫ ОТЛИЧАТЬСЯ НА ШАГ ВЫБОРКИ
ТО ЕСТЬ ПЕРВЫЙ 0 ПОСЛЕДНИЙ 128 И ШАГ 128
ИЛИ НАЧАЛО 0 КОНЕЦ 1000 ШАГ 128
ТО ЕСТЬ ВСЯ ВЫБОРКА ДОЛЖНА БЫТЬ БОЛЬШЕ ЧЕМ ШАГ'''




import os
os.makedirs("saved_models", exist_ok=True)

from Bio import PDB
from Bio.PDB.Residue import Residue

import csv
import pandas as pd

import numpy as np
from sklearn.linear_model import SGDClassifier

try:
    with open("path.txt") as f:
        lines=f.read()
        pdbset_dir=lines.split(sep=',')[0]
        data_file_path=lines.split(sep=',')[1]
except(FileNotFoundError):
    print('Укажите путь к пдб и датабазе с обратным слэшэм /')
    pdbset_dir=str(input())
    data_file_path=str(input())
    with open("path.txt", "a") as f:      ### СОХРАНЕНИЕ ПУТИ К ДАТАБАЗЕ И ПДБ ДИРЕКТОРИИ ДЛЯ УДОБСТВА ПРИ ПЕВРОМ ЗАПУСКЕ
        f.write(str(pdbset_dir))
        f.write(str(','))
        f.write(str(data_file_path))

print('ОЧЕНЬ ОЧЕНЬ ВАЖНО ЗНАЧЕНИЯ НАЧАЛА И КОНЦА ВЫБОРКИ ДОЛЖНЫ ОТЛИЧАТЬСЯ НА ШАГ ВЫБОРКИ\nТО ЕСТЬ ПЕРВЫЙ 0 ПОСЛЕДНИЙ 128 И ШАГ 128 ИЛИ НАЧАЛО 0 КОНЕЦ 1000 ШАГ 128\nТО ЕСТЬ ВСЯ ВЫБОРКА ДОЛЖНА БЫТЬ БОЛЬШЕ ЧЕМ ШАГ')
print('Выберите модель: BINARY_MODEL(), SVM_MODEL(), SVM_MODEL_LOAD(), SGDClassifier_model(), NeuralNetwork()')
modelle=str(input())

print('первый')
upper_row =(int(input()))
print('последний')
lower_row =(int(input()))
print('шаг')
how_many_prot_for_1iter = (int(input()))
print('с какого начать валидацию')
testnum=int(input())
print('сколько проверить')
test_iter=int(input())
print('Задать максимальную длину белка?')
maximum_sequence_of_all=0
if str(input()) == 'yes':
    print('Введите максимальную длину белка = 3907, это замедляет обучение, но вы никогда не получите ошибку размерностей векторов\n'
          'максимальная длина комплекса = 3900, +5 начальных признаков места мутации, +2 chain id (который ни на что особо не влияет)')
    Ubermaximum_length=int(input())
    maximum_sequence_of_all=Ubermaximum_length





# ПАРАМЕТРЫ ДЛЯ ПРОВЕРКИ НА ПУСТОТЫ В ДАТАБАЗЕ, А ПУСТОТ В НЕЙ МНОГО
parser = PDB.PDBParser(QUIET=True)
data = pd.read_csv(data_file_path, sep=';') ### ВЫГРУЗКА ДАТАБАЗЫ
T=(np.array(([f'{str(x)[0:3]}' for x in [x[0] for x in np.array(data.iloc[upper_row:lower_row, [13]])]])).reshape(len(data.iloc[upper_row:lower_row, [13]]), 1))   ### ВЕКТОР С ТЕМПЕРАТУРАМИ
R = np.array([[8.314 / 4184] for i in T]) ### ВЕКТОР С ГАЗОВОЙ ПОСТОЯННОЙ
affinity_mut = (np.array(data.iloc[upper_row:lower_row, [7]]))   ### АФФИННОСТЬ КОМПЛЕКСА С МУТАЦИЕЙ
affinity_wt = (np.array(data.iloc[upper_row:lower_row, [9]]))    ### АФФИНОСТЬ ОБЫЧНОГО КОМПЛЕКСА

# фУНКЦИЯ РАЗНИЦЫ СВОБОДНОЙ ЭНЕРГИИ СВЯЗЫВАНИЯ ДЛЯ ПРОВЕРКИ НА НЕКЛАССИФИЦИРУЕМЫЕ СЛУЧАИ ТО ЕСТЬ ДДГ = 0
ddg = lambda affinity_mut, affinity_wt, T, R: R * T * (np.log(affinity_mut)) - R * T * (np.log(affinity_wt))

# СЛОВАРИКИ
mut_importance={"SUR":25, "INT":50, "SUP":75, "RIM":100, "COR":125} #весовые коэфф места мутации
oneletterresidues={"A":'ALA', "R":'ARG', "N":'ASN', "D":'ASP', "C":'CYS', "E":'GLU', "Q":'GLN', "G":'GLY', "H":'HIS', "I":'ILE', "L":'LEU', "K":'LYS', "M":'MET', "F":'PHE', "P":'PRO', "S":'SER', "T":'THR', "W":'TRP', "Y":'TYR', "V":'VAL'}
residues={'ALA':1, 'ARG':2, 'ASN':3, 'ASP':4, 'CYS':5, 'GLU':6, 'GLN':7, 'GLY':8, 'HIS':9, 'ILE':10, 'LEU':11, 'LYS':12, 'MET':13, 'PHE':14, 'PRO':15, 'SER':16, 'THR':17, 'TRP':18, 'TYR':19, 'VAL':20, 'HOH':0}


parser = PDB.PDBParser(QUIET=True)


'''

ПРОВЕДЕНИЕ АНАЛИЗА ДАТАБАЗЫ ДЛЯ ПОИСКА УДАЧНЫХ ВЫБОРОК ДЛЯ ТЕСТА БИНАРНОЙ КЛАССИФИКАЦИИ И SVM, ну и просто полезную информацию извлечь

dg=0
a=[]
b=[]
c=[]
e=[]

print(np.array(data.iloc[0:7000, [0]]))
print(len(np.array(data.iloc[0:7000, [0]])))
for i in range(len(np.array(data.iloc[0:7000, [0]]))):
    if (np.array(data.iloc[dg, [7]]))[0]<(np.array(data.iloc[dg+1, [9]]))[0]:
        e.append(dg)
    if (np.array(data.iloc[dg, [0]]))[0]==(np.array(data.iloc[dg+1, [0]]))[0]:
        a.append(dg)
    else:
        b.append(len(a))
        c.append(a)
        a=[]
    dg+=1
lengs=structures_extractor(pdbset_dir, data_file_path, upper_row, lower_row, how_many_prot_for_1iter)
with open("parsing.txt", "a") as f:
    f.write('[')
for i in lengs:
    with open("parsing.txt", "a") as f:
        f.write(str(i))
    with open("parsing.txt", "a") as f:
        f.write(',')
with open("parsing.txt", "a") as f:
    f.write(']')


with open("parsing.txt", "a") as f:
    f.write('[')
for i in e:
    with open("parsing.txt", "a") as f:
        f.write(str(i))
    with open("parsing.txt", "a") as f:
        f.write(',')
with open("parsing.txt", "a") as f:
    f.write(']')

print(c[b.index(max(b))])
print(e)
print(lengs)

'''
# МОДЕЛЬКИ


# БИНАРНАЯ КЛАССИФИКАЦИЯ

def BINARY_MODEL():
    x_train = code_vector
    x_train = [x + [1] for x in x_train]
    x_train = np.array(x_train).astype('float64')
    y_train = np.array(class_vector)
    pt = np.sum([x * y for x, y in zip(x_train, y_train)], axis=0)   ### СУММА ЧИСЛОВЫХ ВЕКТОРОВ СИКВЕНСОВ БЕЛКОВ УМНОЖЕННЫХ НА ИХ МЕТКИ КЛАССОВ
    xxt = np.sum([np.outer(x, x) for x in x_train], axis=0)          ### МАТРИЦА ПОЛУЧЕННАЯ ИЗ СУММЫ ПРОИЗВЕДЕНИй ВЕКТОРОВ СИКВЕНСОВ БЕЛКОВ НА ТРАНСПОНИРОВАННЫЕ ВЕКТОРА СИКВЕНСОВ БЕЛКОВ
    w = np.dot(pt, np.linalg.pinv(np.matrix(xxt), rcond=1e-15))      ### СТРОГОЕ НАХОЖДЕНИЕ ВЕКТОРА ВЕСОВ w ЧЕРЕЗ СКАЛЯРНОЕ ПРОИЗВЕДЕНИЕ
    # print(xxt.shape)
    # print(np.linalg.det(xxt))
    print(w.shape)
    testcodevect=testcode.tolist()
    print(np.array(testcodevect).shape)
    testcodevect=[x +[1] for x in testcodevect]
    testcodevect=np.array(testcodevect).astype('float64')
    print('testcode shape', testcodevect.shape)
    y_pred = np.array([np.sign(np.dot(w.astype('float64'), x)) for x in testcodevect])    ### СКАЛЯРНОЕ ПРОИЗВДЕНИЕ КАЖДОГО ВЕКТОРА СИКВЕНСА НА ВЕКТОР ВЕСОВ
    y_pred=y_pred.astype('int32').ravel()
    print('ПРЕДСКАЗАНИЕ МОДЕЛИ', y_pred)
    print('accuracy',
          [1 == x * y for x, y in zip(y_pred.reshape(len(y_pred)), np.array(testclasses).reshape(len(testclasses)))].count(np.True_),
          "of", len([1 == x * y for x, y in zip(y_pred.reshape(len(y_pred)), np.array(testclasses).reshape(len(testclasses)))]))

# SVM МОДЕЛЬ ДЛЯ ОБУЧЕНИЯ

def SVM_MODEL():
    import sklearn as skl
    import joblib
    data_x = code_vector
    data_y = (np.array(class_vector)).ravel()
    model = skl.svm.SVC(kernel='linear')              ### ИМПОРТИРУЕМ МОДЕЛЬ НА ОПОРНЫХ ВЕКТОРАХ С ЛИНЕЙНЫМ ЯДРОМ
    model.fit(data_x, data_y)              ### ОБУЧЕНИЕ
    filename = 'svm_model.joblib'
    joblib.dump(model, filename)              ### СОХРАНЕНИЕ ОБУЧЕННОЙ МОДЕЛИ
    y_pred = model.predict(testcode)              ### ВАЛИДАЦИЯ
    print(np.array(testclasses).astype('int32').ravel())
    print('ПРЕДСКАЗАНИЕ МОДЕЛИ', y_pred)
    print('accuracy',
          [1 == x * y for x, y in zip(y_pred.reshape(len(y_pred)), np.array(testclasses).reshape(len(testclasses)))].count(np.True_),
          "of", len([1 == x * y for x, y in zip(y_pred.reshape(len(y_pred)), np.array(testclasses).reshape(len(testclasses)))]))
### ЭТА СЛОЖНАЯ СТРОЧКА СЧИТАЕТ КОЛИЧЕСТВО СООТВЕТСТВИЙ ПРЕДСКАЗАНИЯ МОДЕЛИ И ВЕКТОРА КЛАССОВ ВАЛИДАЦИИ

###

# SVM МОДЕЛЬ ДЛЯ ПРОВЕРКИ, ПО ФАЙЛУ С ВЕКТОРАМИ

def SVM_MODEL_LOAD():
    import sklearn as skl
    import joblib
    print('Укажите путь к пресету модели')
    model_preset=str(input())
    # Load the model from the saved file
    loaded_model = joblib.load(str(model_preset))              ### ЗАГРУЗКА МОДЕЛИ
    y_pred = loaded_model.predict(testcode)
    print(testclasses)
    print('ПРЕДСКАЗАНИЕ МОДЕЛИ', y_pred)
    print('accuracy',
          [1 == x * y for x, y in zip(y_pred.reshape(len(y_pred)), np.array(test).reshape(len(test)))].count(np.True_),
          "of", len([1 == x * y for x, y in zip(y_pred.reshape(len(y_pred)), np.array(test).reshape(len(test)))]))


# ФУНКЦИИ ПРОГРАММЫ

def max_sequence(sequences_pylist):
    global maximum_sequence_of_all
    a = [1, 1]
    for structure in sequences_pylist:              ### ФУНКЦИЯ СЧИТАЕТ ДЛИНУ КАЖДОГО ВЕКТОРА С СИКВЕНСОМ БЕЛКА И ЗАПОМИНАЕТ МАКСИМУМАЛЬНУЮ
        a.append(len(structure))                    ### ЕСЛИ МАКСИМАЛЬНАЯ НЕ УКАЗАНА
    return (max(a))


# СОЗДАНИЕ ВЕКТОРА СИКВЕНСОВ БЕЛКОВ ПО ИМЕНАМ ИЗ ДАТАБАЗЫ И ПДБ ФАЙЛАМ

checklength=[]
structures_of_protein=0
parsed_structures_row_ids=0
parser = PDB.PDBParser(QUIET=True)
def structures_extractor(pdbset_dir, data_file_path, upper_row, lower_row, how_many_prot_for_1iter):
    global maximum_sequence_of_all
    print(maximum_sequence_of_all)
    global checklength
    checklength.append(maximum_sequence_of_all)             ### ЗАПОМИНАНИЕ МАКСИМАЛЬНОЙ ДЛИНЫ КАК ОБУЧЕНИЯ ТАК И ВАЛИДАЦИИ
    leng=[]
    def myparser(pdbset_dir, data_file_path, upper_row, lower_row):
        data = pd.read_csv(data_file_path, sep=';')
        structures = (np.array(data.iloc[upper_row:lower_row, [0]])) ### ИМЕНА СТРУКТУР КОМПЛЕКСОВ В ФОРМАТЕ ПДБ
        names = lambda listnames, null: np.append(null, listnames[0:len(listnames), 0:3], axis=None)
        Pdbchains = (names(structures, np.array([])))  # удаление индекса цепей для обращения к файлам пдб
        # print(Pdbchains)
        Pdb = (np.array([f'{x[0:4]}' for x in Pdbchains])).reshape(len(Pdbchains), 1)  # индекс пдб комплекса н массив
        # print(Pdb)
        # print(Pdb)
        file = (Pdb.reshape(1, len(Pdb))).reshape(len(Pdb))
        # print(path+file+'.pdb')
        parser = PDB.PDBParser(QUIET=True)
        return file
    for i in range(upper_row,lower_row-(lower_row%how_many_prot_for_1iter), how_many_prot_for_1iter):  ### ЦИКЛ ДЛЯ РАЗБИЕНИЯ ИЗВЛЕЧЕНИЯ СТРУКТУР НА ЧАСТИ И ЭКОНОМИИ ПАМЯТИ
        parser = PDB.PDBParser(QUIET=True)
        file=myparser(pdbset_dir, data_file_path, upper_row, lower_row)

        def get_structure(pdbset_dir, files): ### ФУНКЦИЯ КОТОРАЯ СОБИРАЕТ ВСЕ ПДБ СТРУКТУРЫ БЕЛКОВ
            a = []
            for i in files:
                a.append((parser.get_structure(str(i), pdbset_dir + str(i) + '.pdb')))  ### ОБРАЩЕНИЕ К ПДБ ФАЙЛАМ ПО НАЗВАНИЮ
            return a
        global structures_of_protein
        structures_of_protein = get_structure(pdbset_dir, file)

        def SequenceReader(structures_of_protein):          ### ФУНКЦИЯ ЧИТАЮЩАЯ СИКВЕНС КАЖДОЙ СТРУКТУРЫ КОМПЛЕКСА
            b = []
            for structure in structures_of_protein:
                a = []
                for model in structure:
                    for chain in model:
                        for residue in chain:
                            a.append(residue.get_resname())
                b.append(a)                                 ### ДОБАВЛЕНИЕ КАЖДОГО СИКВЕНСА В ОБЩИЙ СПИСОК (ПОЗЖЕ ВЕКТОР) СИКВЕНСОВ БЕЛКОВ
            return (b)

        sequences_pylist = SequenceReader(structures_of_protein) ### СПИСОК СИКВЕНСОВ

        def max_sequence(sequences_pylist):             ### ФУНКЦИЯ СЧИТАЮЩАЯ КАЖДЫЙ СИКВЕНС ИЗ СПИСКА И ЗАПОМИНАЮЩАЯ МАКСИМАЛЬНЫЙ
            global maximum_sequence_of_all
            a = [1,1]
            for structure in sequences_pylist:
                a.append(len(structure))
            return (max(a))
        leng.append(max_sequence(sequences_pylist))   ### СПИСОК МАКСИМАЛЬНЫХ СИКВЕНСОВ ПО ВСЕМ ИТЕРАЦИЯМ ПАРСЕРА (how many prot for 1 iter)
    def get_structure(pdbset_dir, files):
        parser = PDB.PDBParser(QUIET=True)
        a = []
        for i in files:
            a.append((parser.get_structure(str(i), pdbset_dir + str(i) + '.pdb')))
        return a

    def SequenceReader(structures_of_protein):
        b = []
        for structure in structures_of_protein:
            a = []
            for model in structure:
                for chain in model:
                    for residue in chain:
                        a.append(residue.get_resname())
            b.append(a)
        return (b)

    def max_sequence(sequences_pylist):
        global maximum_sequence_of_all
        a = [1, 1]
        for structure in sequences_pylist: ### ЭТИ ФУНКЦИИ ПРОДУБЛИРОВАНЫ ВНЕ ЦИКЛА ПОТОМУ ЧТО ОШИБКА "ФУНКЦИИ НЕ ОПРЕДЕЛЕНЫ"
            a.append(len(structure))
        return (max(a))
    leng.append(max_sequence(SequenceReader(get_structure(pdbset_dir, myparser(pdbset_dir, data_file_path, lower_row-(lower_row%how_many_prot_for_1iter), lower_row)))))
    ### ВСЕ ЭТО БЫЛО НУЖНО ЧТОБЫ ПОСЧИТАТЬ МАКСИМУМ СИКВЕНСОВ,ЭТО ОЧЕНЬ ВАЖНЫЙ ПАРАМЕТР, СТРОЧКА ВЫШЕ ДОБАВЛЯЕТ ОСТАВШИЕСЯ МАКСИМУМЫ ИЗ lower_row-(lower_row%how_many_prot_for_1iter),
    ### КОТОРЫЕ НЕ ВОШЛИ В БОЛЬШИЕ ШАГИ: ШАГ УСЛОВНО 250, ОСТАТОК 33

    if max(checklength) < max(leng):     ### ФУНКЦИЯ КОТОРАЯ ОБНОВЛЯЕТ ГЛОБАЛЬНЫЙ МАКСИМУМ СИКВЕНСА ЕСЛИ НАХОДИТ НОВЫЙ МАКСИМУМ
        maximum_sequence_of_all=max(leng)
    parser = PDB.PDBParser(QUIET=True)
    print(structures_of_protein)
    def VectorMaker(structures_of_protein, upper_row, lower_row):
            data = pd.read_csv(data_file_path, sep=';')
            affinity_mut = (np.array(data.iloc[upper_row:lower_row, [7]]))
            affinity_wt = (np.array(data.iloc[upper_row:lower_row, [9]]))
            # print(np.array([data.iloc[upper_row:lower_row, [7]]]))
            T = (np.array(([f'{str(x)[0:3]}' for x in [x[0] for x in np.array(data.iloc[upper_row:lower_row, [13]])]])).reshape(len(data.iloc[upper_row:lower_row, [13]]), 1))
            # print(T)
            R = np.array([[8.314 / 4184] for i in T])
            b = []
            global parsed_structures_row_ids    ### ЗАПОМИНАНИЕ НОРМАЛЬНЫХ СТРОК БЕЗ ПУСТОТ
            parsed_structures_row_ids = []
            e=-1
            lenlen=[]
            chainindices=[]
            for structure in structures_of_protein:
                a = []
                e += 1
                try:
                    T[e].astype('float64')
                except(ValueError): ### ЭТО ПРОВЕРКА НА ОТСУТВУЮЩИЕ ЗНАЧЕНИЯ В СТОБЛЦЕ affinity_mut, В СТОЛБЦЕ affinity_wt (да и там тоже есть пустые места :) ИЛИ ЕСЛИ ПОЛУЧАЕТСЯ НОЛЬ В РАЗНИЦЕ DDG
                    continue
                if (str((np.array(affinity_mut[e]).astype("float64"))[0]))=="nan" or (str((np.array(affinity_wt[e]).astype("float64"))[0]))=="nan" or (np.sign(ddg(affinity_mut[e], affinity_wt[e], T[e].astype('float64'), R[e]))==float(0)):
                    continue
                parsed_structures_row_ids.append(upper_row+e)                                ############## ПАРАМЕТР ВАЖНЫЙ КОТОРЫЙ ЗАПИСЫВАЕТ ИНДЕКСЫ БЕЛКОВ ПО СТРОКАМ
                for model in structure:
                    for chain in model:
                        a.append('chain_id:'+chain.id)
                        for residue in chain:
                            a.append(residue.get_resname())           ### ТЕПЕРЬ ПО ВСЕМ НОРМАЛЬНЫМ СТРУКТУРАМ БЕЛКОВ И НАЙДЕННОМУ МАКСИМУМУ ДЛИНЫ
                c=len(a)                                              ### ДЕЛАЕМ ВЕКТОР С СИКВЕНСАМИ, ГДЕ ХВОСТ БЕЛКА ЗАПОЛНЕН АЛАНИНОМ ДЛЯ СООТВЕТСТВИЯ
                for i in range(maximum_sequence_of_all - c):          ### ДЛИНЕ МАКСИМАЛЬНОГО БЕЛКА
                    a.append('ALA')
                b.append(a)
                print(len(a))
            return (b)
            try:
                check_shape = np.array(b)                             ### НА СЛУЧАЙ РАЗМЕРНОСТИ ВЕКТОРА ВАЛИДАЦИИ БОЛЬШЕ ВЕКТОРА ОБУЧЕНИЯ
            except(ValueError):
                print("максимальная длина белка обучения меньше длины белка валидации")
    return VectorMaker(structures_of_protein, upper_row, lower_row)

# СОЗДАНИЕ ВЕКТОР КЛАССОВ ПО ДАННЫМ АФФИННОСТИ, +1 - DDG > 0, -1 - DDG < 0

def class_vector_maker(data_file_path, upper_row, lower_row, maximum_sequence_of_all):
    data = pd.read_csv(data_file_path, sep=';')
    affinity_mut=(np.array(data.iloc[upper_row:lower_row, [7]]))
    affinity_wt=(np.array(data.iloc[upper_row:lower_row, [9]]))
    #print(np.array([data.iloc[upper_row:lower_row, [7]]]))
    T = (np.array(([f'{str(x)[0:3]}' for x in [x[0] for x in np.array(data.iloc[upper_row:lower_row, [13]])]])).reshape(len(data.iloc[upper_row:lower_row, [13]]), 1))
    #print(T)
    R = np.array([[8.314 / 4184] for i in T])
    #print(affinity_wt, affinity_mut)
    ddg = lambda affinity_mut, affinity_wt, T, R: R * T * (np.log(affinity_mut)) - R * T * (np.log(affinity_wt))
    class_vector=[]
    e=-1
    for i in range(lower_row-upper_row):    ### ДЛЯ КАЖДОГО БЕЛКА В ОБОЗНАЧЕННЫХ ГРАНИЦАХ СЧИТАЕМ DDG и ОПРЕДЕЛЯЕМ КЛАСС -1 +1
        e+=1
        if i == len(R):
            break
        print(e)
        try:
            T[e].astype('float64')
        except(ValueError):
            continue              ### ЭТО ПРОВЕРКА НА ОТСУТВУЮЩИЕ ЗНАЧЕНИЯ В СТОБЛЦЕ affinity_mut ИЛИ ЕСЛИ ПОУЛЧАЕТСЯ НОЛЬ В РАЗНИЦЕ DDG
        if (str((np.array(affinity_mut[e]).astype("float64"))[0]))=="nan" or (str((np.array(affinity_wt[e]).astype("float64"))[0]))=="nan" or (np.sign(ddg(affinity_mut[e], affinity_wt[e], T[e].astype('float64'), R[e]))==float(0)):
                    continue
        a=np.sign(ddg(affinity_mut[i], affinity_wt[i], T[i].astype('float64'), R[i]))      ### ДОБАВЛЕНИЕ В ВЕКТОР КЛАССОВ МЕТОК -1 ИЛИ 1 ДЛЯ КАЖДОГО БЕЛКА
        class_vector.append(a)
    #print(np.array(class_vector).shape)
    return class_vector



# ВНЕСЕНИЕ МУТАЦИЙ В ВЕКТОР СИКВЕНСОВ БЕЛКОВ

def change_seq(data_file_path, protein_vector, upper_row, lower_row):
    data = pd.read_csv(data_file_path, sep=';')
    mutations=[]
    for q in parsed_structures_row_ids:
        mutations.append(data.iloc[q, [2]])              ### ВЫГРУЗКА ДАННЫХ О МУТАЦИИ
    i=0
    for x in (np.append([], mutations[0:len(mutations)], axis=None)):                     ###УЧЕСТЬ МНОЖЕСТВНЕННЫЕ МУТАЦИИ LI39A,EI41L  RI39D,EI41L
        try:
            #check=(str(protein_vector[int(i), int((protein_vector[i].tolist()).index(str('chain_id:' + x[1])) + int(x[2:len(x) - 1]))])==oneletterresidues[x[0]])                                                                                  ###СДЕЛАНО
            protein_vector[int(i), int((protein_vector[i].tolist()).index(str('chain_id:'+x[1]))+int(x[2:len(x) - 1]))] = oneletterresidues[x[len(x) - 1]]   ###  ЕСЛИ НЕ ПОЛУЧАЕТСЯ СРАЗУ
            #print(check)         ### В НАЧАЛЕ ЦЕПИ ЕСТЬ CHAIN_ID: ПО КОТОРОМУ МЫ НАХОДИМ НАЧАЛО ЦЕПИ И ЕЕ НАЗВАНИЕ
        except(ValueError):      ### МУТАЦИЯ СОВЕРШАЕТСЯ ПО ИНДЕКСУ CHAIN_ID: + НОМЕР ОСТАТКА
            a = list(str(x).split(sep=','))    #####3 a = ['RA156K,DA160H,IA168V,QA170K']     ### ТО СОЗДАЕМ СПИСОК
            #print(a)                     ##### c = 'RA156K'
            #print("x", x)                                                                     ### ПРОВОДИМ ЗАМЕНУ ДЛЯ КАЖДОЙ МУТАЦИИ ИЗ СПИСКА
            for c in a:
               # for x in protein_vector[i]:
                    #count=[]
                    #if x != 'HOH':
                        #count.append
                protein_vector[int(i), int((protein_vector[i].tolist()).index(str('chain_id:'+c[1]))+int(c[2:len(c) - 1]))] = oneletterresidues[c[len(c) - 1]]        ### ЗАМЕНА АМИНОКИСЛОТЫ НА МЕСТЕ МУТАЦИИ
        #protein_vector[i]
        i += 1
    return protein_vector



# ПЕРЕВОД СИКВЕНСОВ В ЧИСЛОВОЙ ВЕКТОР ПО СЛОВАРЮ

def SequenceEncoder(protein_vector):
    b=[]
    i=0
    for structure in protein_vector:
        a = []
        c=0
        for acid in structure:                              ### ДЛЯ КАЖДОГО БЕЛКА В ВЕКТОРЕ МЕНЯЕМ АМИНОКИСЛОТЫ НА ЧИСЛОВОЙ КОД ПО СЛОВАРЮ
            try:
                protein_vector[i,c]=residues[acid]           #########  ЗАМЕНИТЬ TRY НА ### dictionary.get(acid, 0.01)
            except (KeyError):
                protein_vector[i,c]=0.01                       ### ЕСЛИ НЕТ АМИНОКИСЛОТЫ, ТО HOH, MG и тд заменяем на 0,01
            c+=1
        i+=1
    return (protein_vector)



# ДОБАВЛЕНИЕ К ВЕКТОРУ ДОПАОЛНИТЕЛЬНЫХ ПРИЗНАКОВ В НАЧАЛО ВЕКТОРА

def AddOtherAttributesToVector(code_vector, parsed_structures_row_ids, upper_row, lower_row):
    #mutations_place = ((np.array(data.iloc[upper_row:lower_row, [3]])))
    code_vector=code_vector.tolist()
    mut_importance = {"SUR": 25, "INT": 50, "SUP": 75, "RIM": 100, "COR": 125}  # весовые коэфф места мутации
    mut_characteristics = np.zeros((len(parsed_structures_row_ids), 5))         ### ВЫГРУЗКА ДАННЫХ О МЕСТЕ МУТАЦИИ ПО КАЖДОМУ ИНДЕКСУ СТРОКИ БЕЛКА
    for i in range((len(parsed_structures_row_ids))):
        mutations_place=(np.array(data.iloc[parsed_structures_row_ids[i], [3]]))
        mut_str=(mutations_place[0]).split(sep=',')                             ### ДЕЛАЕМ СПИСОК МЕСТ МУТАЦИИ
        mut_place_vect=[mut_str.count("SUR")*mut_importance["SUR"],             ###SUR
                                mut_str.count("INT")*mut_importance["INT"],    ###INT
                                mut_str.count("SUP")*mut_importance["SUP"],    ###SUP ### СЧИТАЕМ И ЗАМЕНЯЕМ НА ЧИСЛОВОЙ КОЭФФ ПО СЛОВАРЮ
                                mut_str.count("RIM")*mut_importance["RIM"],    ###RIM
                                mut_str.count("COR")*mut_importance["COR"]]   ###COR
        for m in mut_place_vect:
            code_vector[i].insert(0, m)
    return (code_vector)



# ОБУЧАЮЩАЯ ВЫБОРКА      ### ПОСЛЕДОВАТЕЛЬНОЕ ПРИМЕНЕНИЕ ФУНКЦИЙ К ВЕКТОРУ, ЧТОБЫ ПОЛУЧИЛСЯ ЧИСЛОВОЙ ВЕКТОР ОБУЧЕНИЯ И ВЕКТОР МЕТОК КЛАССОВ

if modelle != 'NeuralNetwork()':
    if modelle != 'SGDClassifier()':
        protein_vector=structures_extractor(pdbset_dir, data_file_path, upper_row, lower_row, how_many_prot_for_1iter)
        print(protein_vector)
        protein_vector=np.array(protein_vector)
        #print('chain',(protein_vector[0].tolist()).index('chain_id:E'))
        #print("форма вектора белков", protein_vector.shape)
        protein_vector=change_seq(data_file_path, protein_vector, upper_row, lower_row)
        code_vector=SequenceEncoder(protein_vector)
        code_vector=AddOtherAttributesToVector(code_vector, parsed_structures_row_ids, upper_row, lower_row)
        print("форма вектора обучения", np.array(code_vector).shape)
        class_vector=class_vector_maker(data_file_path, upper_row, lower_row, maximum_sequence_of_all)
        print("форма вектора классов", np.array(class_vector).shape)

    # ВАЛИДАЦИОННАЯ ВЫБОРКА

    testseq=np.array(structures_extractor(pdbset_dir, data_file_path, testnum, testnum+test_iter, test_iter))
    print("форма вектора белков валидации", testseq.shape)
    testseq=change_seq(data_file_path, testseq, testnum, testnum+test_iter)
    testcode=SequenceEncoder(testseq)
    testcode=AddOtherAttributesToVector(testcode, parsed_structures_row_ids, testnum, testnum+test_iter)
    print("форма вектора валидации", np.array(testcode).shape)
    testclasses=class_vector_maker(data_file_path, testnum, testnum+test_iter, maximum_sequence_of_all)
    print("форма вектора классов валидации", np.array(testclasses).shape)

    testcode = np.array(testcode, dtype=np.float64)
    print(testcode)



### СТОХАСТИЧЕСКИЙ ГРАДИЕНТНЫЙ СПУСК С ДООБУЧЕНИЕМ

def SGDClassifier_model(code_vector, class_vector, testcode, testclasses):
    import sklearn as skl
    import joblib
    global maximum_sequence_of_all
    print('До какого белка делаем выборку?')
    x_train_end=int(input())
    maximum_sequence_of_all = Ubermaximum_length
    model = skl.linear_model.SGDClassifier(loss='hinge')  # Hinge-loss ДАЕТ МОЕЛИ ПОВЕДЕНИЕ КАК У SVM
    print('Обучить модель?')
    if str(input()) == 'yes':
        x_batch, y_batch = code_vector, (np.array(class_vector)).ravel()
        model.partial_fit(x_batch, y_batch, classes=np.array([-1, 1]))    ### ВЫБОРКА ИЗНАЧАЛЬНАЯ НА ОСНОВЕ upper_row, lower_row
        for i in range((x_train_end-lower_row)//how_many_prot_for_1iter):
            if i*how_many_prot_for_1iter+lower_row > x_train_end:
                break               ### ФОРМИРОВАНИЕ НОВОЙ ОБУЧАЮЩЕЙ ВЫБОРКИ ДЛЯ КАЖДОЙ ЭПОХИ С начала lower_row до конца x_train_end по batch = how_many_prot_for_1ite
            print('Epochs ', i,' of ',(x_train_end-lower_row)//how_many_prot_for_1iter)
            protein_vector = np.array(structures_extractor(pdbset_dir, data_file_path, lower_row+i*how_many_prot_for_1iter, lower_row+(i+1)*how_many_prot_for_1iter, how_many_prot_for_1iter))
            print("форма вектора белков", protein_vector.shape)
            protein_vector = change_seq(data_file_path, protein_vector, lower_row+i*how_many_prot_for_1iter, lower_row+(i+1)*how_many_prot_for_1iter)
            code_vector = SequenceEncoder(protein_vector)
            code_vector = AddOtherAttributesToVector(code_vector, parsed_structures_row_ids, lower_row+i*how_many_prot_for_1iter, lower_row+(i+1)*how_many_prot_for_1iter)
            print("форма вектора обучения", np.array(code_vector).shape)
            class_vector = class_vector_maker(data_file_path, lower_row+i*how_many_prot_for_1iter, lower_row+(i+1)*how_many_prot_for_1iter, maximum_sequence_of_all)
            print("форма вектора классов", np.array(class_vector).shape)
            x_batch, y_batch = code_vector, (np.array(class_vector)).ravel()
            model.partial_fit(x_batch, y_batch)                     ###### ДООБУЧЕНИЕ МОДЕЛИ КАЖДЫЙ РАЗ НА ВЫБОРКЕ ДЛИНЫ batcjh
    print('Модель обучена, сохранить?')
    if str(input()) == 'yes':
        filename = 'sgd_svm_model.joblib'
        joblib.dump(model, filename)
    print('Загрузить модель?')
    if str(input()) == 'yes':
        model_preset='sgd_svm_model.joblib'
        loaded_model=joblib.load(model_preset)
        y_pred = loaded_model.predict(np.array(testcode, dtype=np.float64))
        print(np.array(testclasses).astype('int32').ravel())
        print('ПРЕДСКАЗАНИЕ МОДЕЛИ', y_pred)

        print('accuracy',
              [1 == x * y for x, y in zip(y_pred.reshape(len(y_pred)), np.array(testclasses).reshape(len(testclasses)))].count(np.True_),
              "of", len([1 == x * y for x, y in zip(y_pred.reshape(len(y_pred)), np.array(testclasses).reshape(len(testclasses)))]))


if modelle == 'BINARY_MODEL()':
    BINARY_MODEL()
elif modelle == 'SVM_MODEL()':
    SVM_MODEL()
elif modelle == 'SGDClassifier_model()':
    SGDClassifier_model(code_vector, class_vector, testcode, testclasses)
elif modelle == 'SVM_MODEL_LOAD()':
    SVM_MODEL_LOAD()
elif modelle == 'NeuralNetwork()':
    NeuralNetwork()
'''
# СОХРАНИТЬ МОДЕЛЬ В ВИДЕ .TXT

w = model.coef_[0]
w0 = model.intercept_[0]
v_support = model.support_vectors_
w = np.insert(w, 0, w0)
vectors=[w,v_support]

with open("vector_w.txt", "a") as f:
    f.write('w')

for i in w:
  with open("vector_w.txt", "a") as f:
    f.write((str(i)+','))

with open("vector_w.txt", "a") as f:
    f.write('support')

for i in v_support:
    with open("vector_w.txt", "a") as f:
        f.write('[')
    for c in i:
        with open("vector_w.txt", "a") as f:
            f.write((str(c)+','))
    with open("vector_w.txt", "a") as f:
      f.write('],')
'''