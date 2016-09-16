import h5py
from os.path import join,exists
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Flatten,Merge
from keras.layers.convolutional import Convolution2D,MaxPooling2D
from keras.optimizers import Adadelta,RMSprop
from hyperas.distributions import choice, uniform, conditional
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm
from random import randint
from sklearn.cross_validation import train_test_split

def reportAcc(acc,score,bestaccfile):
    print('Hyperas:valid accuracy:', acc,'valid loss',score)
    if not exists(bestaccfile):
        current = float("inf")
    else:
        with open(bestaccfile) as f:
            current = float(f.readline().strip())
    if score < current:
        with open(bestaccfile,'w') as f:
            f.write('%f\n' % score)
            f.write('%f\n' % acc)

def model(X_train, Y_train, X_test, Y_test):
    W_maxnorm = 3
    DROPOUT = {{choice([0.3,0.5,0.7])}}
                                                                                                                                            
    model = Sequential()
    model.add(Convolution2D(64, 1, 5, border_mode='same', input_shape=(4, 1, DATASIZE),activation='relu',W_constraint=maxnorm(W_maxnorm)))
    model.add(MaxPooling2D(pool_size=(1, 5),strides=(1,3)))
    model.add(Flatten())
                                                                                                                                            
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(32,activation='relu'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(2))
    model.add(Activation('softmax'))
                                                                                                                                            
    myoptimizer = RMSprop(lr={{choice([0.01,0.001,0.0001])}}, rho=0.9, epsilon=1e-06)
    mylossfunc = 'binary_crossentropy'
    model.compile(loss=mylossfunc, optimizer=myoptimizer,metrics=['accuracy'])
    model.fit(X_train, Y_train, batch_size=100, nb_epoch=5,validation_split=0.1)

    score, acc = model.evaluate(X_test,Y_test)
    model_arch = 'MODEL_ARCH'
    bestaccfile = join('TOPDIR',model_arch,model_arch+'_hyperbestacc')
    reportAcc(acc,score,bestaccfile)
    
    return {'loss': score, 'status': STATUS_OK,'model':(model.to_json(),myoptimizer,mylossfunc)}

def data():
    myprefix = join('TOPDIR','DATACODE' + 'PREFIX')
    X_train,Y_train = getdata(myprefix + '.train.h5.batch')
    X_test,Y_test = getdata(myprefix + '.valid.h5.batch')
    return X_train, Y_train, X_test, Y_test

def BatchGenerator(batchnum,cls,topdir,data_code):
    data1prefix = join(topdir,data_code + 'PREFIX' +'.'+cls+'.h5.batch')
    for i in range(batchnum):
        data1f = h5py.File(data1prefix+str(i+1),'r')
        data1 = data1f['data']
        label = data1f['label']
        yield (data1,label)

def BatchGenerator2(minibatch_size,batchnum,cls,topdir,data_code):
    data1prefix = join(topdir,data_code + 'PREFIX' +'.'+cls+'.h5.batch')
    while True:
        for i in range(batchnum):
            data1f = h5py.File(data1prefix+str(i+1),'r')
            data1 = data1f['data']
            label = data1f['label']
            datalen = len(data1)
            idx = 0
            while idx+minibatch_size <= datalen:
                idx += minibatch_size
                yield ([data1[(idx-minibatch_size):idx],label[(idx-minibatch_size):idx]])
            if idx < datalen:
                yield ([ data1[idx:],label[idx:]    ])

def getdata(data1prefix):
    data1f = h5py.File(data1prefix+'1','r')
    return (data1f['data'],data1f['label'])
