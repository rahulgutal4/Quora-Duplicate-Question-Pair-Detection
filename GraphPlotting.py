import matplotlib as plt
import pandas

def graphComparison(allinputValues):
    accuracy={}
    accuracy['15-p']=0
    accuracy['15-i']=0
    accuracy['15+p']=0
    accuracy['15+i']=0
    for inputValues in allinputValues:
        for inputVal in inputValues:
            if len(str(inputVal['question1']).split(" "))<15 or len(str(inputVal['question1']).split(" "))<15:
                if inputVal['predicted']==inputVal['acutal']:
                    accuracy['15-p']=accuracy['15-p']+1
                else:
                    accuracy['15-i'] = accuracy['15-i'] + 1
            else:
                if inputVal['predicted']==inputVal['acutal']:
                    accuracy['15-p']=accuracy['15-p']+1
                else:
                    accuracy['15-i'] = accuracy['15-i'] + 1

    plt.plot([accuracy['15-p'], accuracy['15-i'],accuracy['15+p'],accuracy['15+i']])
    plt.ylabel('predictions')
    plt.show()


data1=pandas.read_csv('data/output/output100.csv')
data2=pandas.read_csv('data/output/output100.csv')
data3=pandas.read_csv('data/output/output100.csv')
data=[data1,data2,data3]
graphComparison(data)