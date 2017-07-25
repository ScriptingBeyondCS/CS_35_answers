import csv
from collections import defaultdict
import matplotlib.pyplot as plt

def getData(filename='city_populations.csv'):

    row_list = []
    try:
        csvfile = open('city_populations.csv', newline='')
        csvrows = csv.reader(csvfile)

        for row in csvrows:
            row_list.append(row)
        
        del csvrows
        csvfile.close()

    except FileNotFoundError as e:
        print("File not found")

    return row_list

# MAKE DATA NICE

# def makeNewCSV():
#     data = getData('cityPops.csv')
#     nums = ['1','2','3','4','5','6','7','8','9']
#     d = defaultdict(float)
#     newData = []

#     for row in data:
#         # print (repr(row[2]))
#         if row[2] != '':
#             first_digit = row[2][0]
#             if first_digit in nums:
#                 try:
#                     newData.append(row)
#                 except IndexError:
#                     pass

#     try: 
#         csvfile = open('city_populations.csv', 'w')
#         filewriter = csv.writer(csvfile, delimiter=",")
#         for row in newData:
#             filewriter.writerow(row)
#         csvfile.close()
#     except:
#         print('File could not be opened')


def makeDict():
    data = getData()
    nums = ['1','2','3','4','5','6','7','8','9']
    d = defaultdict(float)
    totalEntries = 0

    for x in nums:
        d[x] = 0

    for row in data:
        first_digit = row[2][0]
        d[first_digit] += 1
        totalEntries += 1

    for key in d:
        d[key] /= totalEntries

    return d

def plotDict():

    d = makeDict()
    X = [1,2,3,4,5,6,7,8,9]
    Y = []
    for num in X:
        Y += [d[str(num)]]
   
    plt.bar(X, Y, align='center')
    plt.xticks(X)
    plt.title('World Population First Digits')
    plt.xlabel('First Digit')
    plt.ylabel('Percentage')
    plt.show()