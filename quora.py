import sys
import os
import numpy as np
import pandas as pd
import numpy as np
import gensim
import csv
from siamese import *
from keras.optimizers import RMSprop, SGD, Adam
from tqdm import tqdm
from utils import TfidfEmbeddingVectorizer
import gensim
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
global dim
dim = 300

class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                line = unicode(str(line),"utf-8")
                yield line.split()
                

def generateLabelledSentences(data, label):
    i=0
    labelledData = []
    for line in data:
        labelledData.append(gensim.models.doc2vec.LabeledSentence(words=line, tags=[label+str(i)]))
        i+=1
    return labelledData    
                
def word2vec():
    
    model = None
    path = 'models/w2v_model' + str(dim)
    if os.path.exists(path):
        model = gensim.models.word2vec.Word2Vec.load(path)
        # trim memory
        model.init_sims(replace=True)
    else:
        # train model
        model = gensim.models.Word2Vec(size=dim, workers=16, iter=10, negative=20)
        questions = MySentences('data/')
        model.build_vocab(questions)
        model.train(questions)
        # trim memory
        model.init_sims(replace=True)
        # save model
        model.save(path)
        del questions
    
    questions = MySentences('test/')
    model.build_vocab(questions, update = True)
    model.train(questions)
    # creta a dict 
    w2v = dict(zip(model.index2word, model.syn0))
    print "Number of tokens in Word2Vec:", len(w2v.keys())
    return w2v

def doc2vec(df, dft):

    model = None
    path = 'models/d2v_model' + str(dim)
    if os.path.exists(path):
        model = gensim.models.Doc2Vec.load(path)
        # trim memory
        model.init_sims(replace=True)
    else:
        # train model
        model = gensim.models.Doc2Vec(size=dim, workers=16, iter=10, negative=20)
        questions = list(generateLabelledSentences(df['question1'], 'question1')) + list(generateLabelledSentences(df['question2'], 'question2')) 
        #+ list(generateLabelledSentences(dft['question1'], 't_question1'))  +  list(generateLabelledSentences(dft['question2'], 't_question2'))
        model.build_vocab(questions)
        model.train(questions)
        # trim memory
        model.init_sims(replace=True)
        # save model
        model.save(path)
        del questions
    
    questions = list(generateLabelledSentences(dft['question1'], 't_question1')) + list(generateLabelledSentences(dft['question2'], 't_question2')) 
    model.build_vocab(questions, update = True)
    model.train(questions)
    # creta a dict 
    d2v = dict(zip(model.index2word, model.syn0))
    print "Number of tokens in Word2Vec:", len(d2v.keys())
    return d2v

def transformVectors(w2v, df, dft, path):

    from utils import TfidfEmbeddingVectorizer

    if os.path.exists(path):
        df = pd.read_pickle(path)
    else:
        # gather all questions
        questions = list(df['question1']) + list(df['question2'])
    
        #tokenize questions
        c = 0
        for question in tqdm(questions):
            questions[c] = list(gensim.utils.tokenize(question, deacc=True))
            c += 1

        me = TfidfEmbeddingVectorizer(w2v)
        me.fit(questions)
        # exctract word2vec vectors
        vecs1 = me.transform(df['question1'])
        df['q1_feats'] = list(vecs1)
    
        vecs2 = me.transform(df['question2'])
        df['q2_feats'] = list(vecs2)
    
        # save features
        pd.to_pickle(df, path)

    me = TfidfEmbeddingVectorizer(w2v)
    questions = list(dft['question1']) + list(dft['question2'])
    me.fit(questions)
    # exctract word2vec vectors
    vecs1 = me.transform(dft['question1'])
    dft['q1_feats'] = list(vecs1)
    
    vecs2 = me.transform(dft['question2'])
    dft['q2_feats'] = list(vecs2)
    return df, dft
 
def main():

    df = pd.read_csv("data/quora_duplicate_questions.tsv",delimiter='\t')
    dft = pd.read_csv("test/test.csv", delimiter=',')

    # encode questions to unicode
    df['question1'] = df['question1'].apply(lambda x: unicode(str(x),"utf-8"))
    df['question2'] = df['question2'].apply(lambda x: unicode(str(x),"utf-8"))
    dft['question1'] = dft['question1'].apply(lambda x: unicode(str(x),"utf-8"))
    dft['question2'] = dft['question2'].apply(lambda x: unicode(str(x),"utf-8"))

    val=input("press 1 for word2vec and 2 for doc2vec\n")
    if val==1:
        w2v = word2vec()
    else:
        w2v = doc2vec(df, dft)
    path = 'models/w2v_vectors' + str(dim)
    df, dft = transformVectors(w2v, df, dft, path)

    # shuffle df
    df = df.reindex(np.random.permutation(df.index))

    # set number of train and test instances
    num_train = int(df.shape[0] * 0.88)
    num_test = df.shape[0] - num_train                 
    print("Number of training pairs: %i"%(num_train))
    print("Number of testing pairs: %i"%(num_test))

    # init data data arrays
    X_train = np.zeros([num_train, 2, dim])
    X_test  = np.zeros([num_test, 2, dim])
    Y_train = np.zeros([num_train]) 
    Y_test = np.zeros([num_test])

    # format data 
    b = [a[None,:] for a in list(df['q1_feats'].values)]
    q1_feats = np.concatenate(b, axis=0)

    b = [a[None,:] for a in list(df['q2_feats'].values)]
    q2_feats = np.concatenate(b, axis=0)

    # fill data arrays with features
    X_train[:,0,:] = q1_feats[:num_train]
    X_train[:,1,:] = q2_feats[:num_train]
    Y_train = df[:num_train]['is_duplicate'].values
            
    X_test[:,0,:] = q1_feats[num_train:]
    X_test[:,1,:] = q2_feats[num_train:]
    Y_test = df[num_train:]['is_duplicate'].values
    
    del b
    del q1_feats
    del q2_feats

    # create model
    print 'siamese model'
    net = create_network(dim)

    # train
    #optimizer = SGD(lr=0.01, momentum=0.8, nesterov=True, decay=0.004)
    optimizer = Adam(lr=0.001)
    net.compile(loss=contrastive_loss, optimizer=optimizer)

    for epoch in range(10):
        net.fit([X_train[:,0,:], X_train[:,1,:]], Y_train,
              validation_data=([X_test[:,0,:], X_test[:,1,:]], Y_test),
              batch_size=128, nb_epoch=1, shuffle=True)
    
        # compute final accuracy on training and test sets
        pred = net.predict([X_test[:,0,:], X_test[:,1,:]])
        te_acc = compute_accuracy(pred, Y_test)
    
        print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))
    
    rows = zip(df[num_train:]['question1'], df[num_train:]['question2'], df[num_train:]['is_duplicate'], pred)
    res = pd.DataFrame(data=rows)
    res.to_csv(path_or_buf='test/output_test.csv', sep='\t', encoding='utf-8')
    
    # format data 
    b = [a[None,:] for a in list(dft['q1_feats'].values)]
    q1_feats = np.concatenate(b, axis=0)

    b = [a[None,:] for a in list(dft['q2_feats'].values)]
    q2_feats = np.concatenate(b, axis=0)
    #print q1_feats
    X_test  = np.zeros([dft.shape[0], 2, dim])
    X_test[:,0,:] = q1_feats[0:]
    X_test[:,1,:] = q2_feats[0:]

    pred = net.predict([X_test[:,0,:], X_test[:,1,:]])
    rows = zip(dft['question1'], dft['question2'], pred)
    with open('test/output.csv','wb') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
        
if __name__ == "__main__":
    global dim
    dim=input("Enter no. of Dimentions (Recommended:300 approximate time taken 2mins)\n")
    main()
