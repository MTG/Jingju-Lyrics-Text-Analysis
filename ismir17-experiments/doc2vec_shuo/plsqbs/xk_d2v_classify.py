# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
from sklearn import preprocessing
import sys
# random shuffle
from random import shuffle

# numpy
import numpy

# classifier
from sklearn.linear_model import LogisticRegression
import logging
import sys
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from sklearn.metrics import f1_score


log = logging.getLogger()
log.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

class TaggedLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(TaggedDocument(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


        
def count_num_lines(file_name):
    f=open(file_name,'r').read()
    return len([i for i in f if i=='\n'])





def build_train_data(sources):
    """build data 2d array using doc2vec repr for classification, and labels, for training and testing"""
    #for each document, loop through the lines, and append the doc2vec representation of each line into the output data file.
    data=[]
    labels=[]
    classes=['MB','KB','YAB','YB']
    
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    for k in sources.keys():
        #print k
        #training data only, four sets
        if 'train' not in k:
            continue
        print k
        size = count_num_lines(k)
        print 'size:',size
        prefix=sources[k]
        for i in range(size):
            lab=[j for j in classes if j in prefix][0]
            #print 'label:',lab
            prefix_train = prefix + "_" + str(i)
            data.append(model.docvecs[prefix_train])
            labels.append(lab) #this is nominal label, later to be transformed
    data=numpy.array(data)
    labels=le.transform(labels)
    return data,labels


def build_test_data(sources):
    """build data 2d array using doc2vec repr for classification, and labels, for training and testing"""
    #for each document, loop through the lines, and append the doc2vec representation of each line into the output data file.
    data=[]
    labels=[]
    classes=['MB','KB','YAB','YB']
    
    le = preprocessing.LabelEncoder()
    le.fit(classes)
    for k in sources.keys():
        #print k
        #training data only, four sets
        if 'test' not in k:
            continue
        print k
        size = count_num_lines(k)
        print 'size:',size
        prefix=sources[k]
        for i in range(size):
            lab=[j for j in classes if j in prefix][0]
            #print 'label:',lab
            prefix_test = prefix + "_" + str(i)
            data.append(model.docvecs[prefix_test])
            labels.append(lab) #this is nominal label, later to be transformed
    data=numpy.array(data)
    labels=le.transform(labels)
    return data,labels


train_doc2vec = bool(int(sys.argv[1]))

log.info('source load')
#sources = {'test-neg.txt':'TEST_NEG', 'test-pos.txt':'TEST_POS', 'train-neg.txt':'TRAIN_NEG', 'train-pos.txt':'TRAIN_POS', 'train-unsup.txt':'TRAIN_UNS'}

sources = {'manban_all_test.txt':'TEST_MB', 'manban_all_train.txt':'TRAIN_MB','kuaiban_all_test.txt':'TEST_KB','kuaiban_all_train.txt':'TRAIN_KB','yaoban_all_test.txt':'TEST_YAB','yaoban_all_train.txt':'TRAIN_YAB','yuanban_all_test.txt':'TEST_YB','yuanban_all_train.txt':'TRAIN_YB'}


log.info('TaggedDocument')
sentences = TaggedLineSentence(sources)

if train_doc2vec:
    print 'prepare for training doc2vec from scratch...'
    log.info('D2V')
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=7)
    model.build_vocab(sentences.to_array())

    log.info('Epoch')
    for epoch in range(10):
        log.info('EPOCH: {}'.format(epoch))
        model.train(sentences.sentences_perm())

    log.info('Model Save')
    model.save('./jingju.d2v')




model = Doc2Vec.load('./jingju.d2v')

log.info('Sentiment')

train_arrays,train_labels=build_train_data(sources)
test_arrays,test_labels=build_test_data(sources)


log.info('Fitting')


classifier = LogisticRegression()
classifier = sklearn.naive_bayes.MultinomialNB()
classifier = sklearn.svm.LinearSVC()

classifier = sklearn.svm.SVC()

classifier.fit(train_arrays, train_labels)
y_pred = classifier.predict(test_arrays)

print 'f1:',f1_score(test_labels, y_pred)

#LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

print 'accuracy:',classifier.score(test_arrays, test_labels)
