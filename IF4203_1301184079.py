import pandas
import math
from numba import jit

# Fungsi untuk membuat dataset awal

def makeDataSet(pregnancies, glucose, bloodPressure, skinThickness, insulin, bmi, diabetesPedigreeFunction, age, outcome):
    i = 0
    dataSet = []
    while i<len(pregnancies):
        data = [pregnancies[i], glucose[i], bloodPressure[i], skinThickness[i], insulin[i], bmi[i], diabetesPedigreeFunction[i], age[i], outcome[i]]
        dataSet.append(data)
        i += 1
    return dataSet

# Fungsi untuk menghitung jarak menggunakan metode euclide

def euclideanDistance(dataTrain, dataTest):
    hasil = (float(dataTrain[0])-float(dataTest[0]))**2 + (float(dataTrain[1])-float(dataTest[1]))**2 + (float(dataTrain[2])-float(dataTest[2]))**2 + (float(dataTrain[3])-float(dataTest[3]))**2 + (float(dataTrain[4])-float(dataTest[4]))**2 + (float(dataTrain[5])-float(dataTest[5]))**2 + (float(dataTrain[6])-float(dataTest[6]))**2 + (float(dataTrain[7])-float(dataTest[7]))**2
    return math.sqrt(hasil)

# Fungsi untuk menghitung jarak menggunakan metode manhattan

def manhattanDistance(dataTrain, dataTest):
    hasil = abs(float(dataTrain[0]-dataTest[0])) + abs(float(dataTrain[1]-dataTest[1])) + abs(float(dataTrain[2]-dataTest[2])) + abs(float(dataTrain[3]-dataTest[3])) + abs(float(dataTrain[4]-dataTest[4])) + abs(float(dataTrain[5]-dataTest[5])) + abs(float(dataTrain[6]-dataTest[6])) + abs(float(dataTrain[7]-dataTest[7]))
    return hasil

# Fungsi untuk melakukan proses kNN dengan nilai k tertentu dan mereturn akurasinya

def kNN(k, train, test):
    predict = []
    for i in range(0, len(test)):
        label = [0, 0]
        distance = []
        for j in range(0, len(train)):
            # jarak = euclideanDistance(test[i], train[j])
            jarak = manhattanDistance(test[i], train[j])
            distance.append([jarak, train[j][8]])
        distance.sort(key = lambda x: x[0])
        for l in range(0, k):
            if distance[l][1] == 0:
                label[0] += 1
            elif distance[l][1] == 1:
                label[1] += 1
        predict.append(label.index(max(label)))
    i = 0
    error = 0
    while i<len(predict):
        if predict[i] != test[i][8]:
            error += 1
        i += 1
    accuracy = round((len(test)-error)/len(test)*100, 2)
    return accuracy

# Fungsi untuk membuat array akurasi untuk kNN = 1-k dari suatu datatest

def accuracyPerDataset(k, train, test):
    i = 1
    accuracyArray = []
    while i<k:
        accuracy = kNN(i, train, test)
        accuracyArray.append(accuracy)
        i += 1
    return accuracyArray

# Fungsi untuk mencari akurasi rata-rata dari setiap array akurasi kNN

def averageAkurasi(dataset1, dataset2, dataset3, dataset4, dataset5):
    i = 0
    arrayTotal = []
    while i<len(dataset1):
        total = dataset1[i] + dataset2[i] + dataset3[i] + dataset4[i] + dataset5[i]
        arrayTotal.append(total)
        i += 1
    arrayAverage = []
    j = 0
    while j<len(arrayTotal):
        average = round(arrayTotal[j]/5,2)
        arrayAverage.append(average)
        j += 1
    return arrayAverage

# Load data beserta seluruh kolomnya
data = pandas.read_csv("Diabetes.csv")
dataPregnancies = data["Pregnancies"]
dataGlucose = data["Glucose"]
dataBloodPressure = data["BloodPressure"]
dataSkinThickness = data["SkinThickness"]
dataInsulin = data["Insulin"]
dataBMI = data["BMI"]
dataDiabetesPedigreeFunction = data["DiabetesPedigreeFunction"]
dataAge = data["Age"]
dataOutcome = data["Outcome"]

# Buat dataset awal
dataSet = makeDataSet(dataPregnancies, dataGlucose, dataBloodPressure, dataSkinThickness, dataInsulin, dataBMI, dataDiabetesPedigreeFunction, dataAge, dataOutcome)

# Bagi dataset menjadi 5-fold cross validation
# train = 1-614, test = 615-768
dataTrain1 = dataSet[0:614]
dataTest1 = dataSet[614:768]

# train = 1-461 dan 642-768, test = 462-641
dataTrain2 = dataSet[0:461]+dataSet[615:768]
dataTest2 = dataSet[461:615]

# train = 1-307 dan 462-768, test = 308-461
dataTrain3 = dataSet[0:307]+dataSet[462:768]
dataTest3 = dataSet[307:462]

# train = 1-154 dan 308-768, test = 155-307
dataTrain4 = dataSet[0:154]+dataSet[308:768]
dataTest4 = dataSet[154:308]

# train = 155-768, test = 1-154
dataTrain5 = dataSet[154:768]
dataTest5 = dataSet[0:154]

k = 154
accuracyDataset1 = accuracyPerDataset(k, dataTrain1, dataTest1)
# print(accuracyDataset1)
# print("======================")
accuracyDataset2 = accuracyPerDataset(k, dataTrain2, dataTest2)
# print(accuracyDataset2)
# print("======================")
accuracyDataset3 = accuracyPerDataset(k, dataTrain3, dataTest3)
# print(accuracyDataset3)
# print("======================")
accuracyDataset4 = accuracyPerDataset(k, dataTrain4, dataTest4)
# print(accuracyDataset4)
# print("======================")
accuracyDataset5 = accuracyPerDataset(k, dataTrain5, dataTest5)
# print(accuracyDataset5)
# print("======================")
averageDataset = averageAkurasi(accuracyDataset1, accuracyDataset2, accuracyDataset3, accuracyDataset4, accuracyDataset5)
# print("INI ARRAY AVERAGE")
# print(averageDataset)
print("NILAI K MAKSIMUM = "+str(averageDataset.index(max(averageDataset))+1))
print("RATA RATA AKURASINYA = "+str(max(averageDataset))+"%")