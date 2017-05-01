import matplotlib.pyplot as plt
import pandas
import re

def graphComparison(allinputValues):
    for inputValues in allinputValues:
        accuracy = {}
        accuracy['10-p'] = 0
        accuracy['10-i'] = 0
        accuracy['10+p'] = 0
        accuracy['10+i'] = 0

        for i in range(0,len(inputValues)):
            inputVal=inputValues.iloc[i]
            prediction=str(inputVal['3']);
            prediction=float(re.sub(r"[^0-9\.]","",prediction))

            if prediction>0.5:
                prediction=0
            else:
                prediction=1


            if len(str(inputVal['0']).split(" "))<10 or len(str(inputVal['1']).split(" "))<10:
                if prediction==int(inputVal['2']):
                    accuracy['10-p']=accuracy['10-p']+1
                else:
                    accuracy['10-i'] = accuracy['10-i'] + 1
            else:
                if prediction==int(inputVal['2']):
                    accuracy['10+p']=accuracy['10+p']+1
                else:
                    accuracy['10+i'] = accuracy['10+i'] + 1
        plt.figure()
        plt.hist([accuracy['10-p'], accuracy['10-i'],accuracy['10+p'],accuracy['10+i']])
        plt.ylabel('predictions')
        plt.show()


data1=pandas.read_csv('test/output_test.csv',sep="\t")
data2=pandas.read_csv('test/output_test.csv',sep="\t")
#data2=pandas.read_csv('data/output/output100.csv')
#data3=pandas.read_csv('data/output/output100.csv')
data=[data1]
graphComparison(data)