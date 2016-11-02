from __future__ import print_function
import theano,time,numpy as np,sys,h5py,cPickle,argparse,subprocess
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from os.path import join,dirname,basename,exists,realpath
from os import system,chdir,getcwd,makedirs
from keras.models import model_from_json
from tempfile import mkdtemp
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import accuracy_score,roc_auc_score

cwd = dirname(realpath(__file__))

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands on EC2.")
    parser.add_argument("-y", "--hyper", dest="hyper", default=False, action='store_true',help="Perform hyper-parameter tuning")
    parser.add_argument("-t", "--train", dest="train", default=False, action='store_true',help="Train on the training set with the best hyper-params")
    parser.add_argument("-e", "--eval", dest="eval", default=False, action='store_true',help="Evaluate the model on the test set")
    parser.add_argument("-p", "--predit", dest="infile", default='', help="Path to data to predict on (up till batch number)")
    parser.add_argument("-d", "--topdir", dest="topdir",help="The data directory")
    parser.add_argument("-s", "--datasize", dest="datasize",help="The length of input sequence")
    parser.add_argument("-c", "--datacode", dest="datacode",default='data',help="The prefix of each data file")
    parser.add_argument("-m", "--model", dest="model",help="Path to the model file")
    parser.add_argument("-o", "--outdir", dest="outdir",default='',help="Output directory for the prediction on new data")
    parser.add_argument("-x", "--prefix", dest="prefix",default='',help="Additional prefix appended after datacode")
    parser.add_argument("-hi", "--hyperiter", dest="hyperiter",default=9,type=int,help="Num of hyper-param combination to try")
    parser.add_argument("-te", "--trainepoch",default=20,type=int,help="The number of epochs to train for")
    parser.add_argument("-bs", "--batchsize",default=100,type=int,help="Batchsize in SGD-based training")
    parser.add_argument("-w", "--weightfile",default=None,help="Weight file for the best model")
    parser.add_argument("-l", "--lweightfile",default=None,help="Weight file after training")
    parser.add_argument("-r", "--retrain",default=None,help="codename for the retrain run")
    parser.add_argument("-rw", "--rweightfile",default='',help="Weight file to load for retraining")
    return parser.parse_args()

def probedata(dataprefix):
    allfiles = subprocess.check_output('ls '+dataprefix+'*', shell=True).split('\n')[:-1]
    cnt = 0
    samplecnt = 0
    for x in allfiles:
        if  x.split(dataprefix)[1].isdigit():
            cnt += 1
            data = h5py.File(x,'r')
            samplecnt += len(data['label'])
    return (cnt,samplecnt)

if __name__ == "__main__":

    args = parse_args()
    topdir = args.topdir
    model_arch = basename(args.model)
    model_arch = model_arch[:-3] if model_arch[-3:] == '.py' else model_arch
    data_code = args.datacode

    outdir = join(topdir,model_arch)
    if not exists(outdir):
        makedirs(outdir)

    architecture_file = join(outdir,model_arch+'_best_archit.json')
    optimizer_file = join(outdir,model_arch+'_best_optimer.pkl')
    weight_file = join(outdir,model_arch+'_bestmodel_weights.h5') if args.weightfile is None else args.weightfile
    last_weight_file = join(outdir,model_arch+'_lastmodel_weights.h5') if args.lweightfile is None else args.lweightfile
    data1prefix = join(topdir,data_code+args.prefix)
    evalout = join(outdir,model_arch+'_eval.txt')

    tmpdir = mkdtemp()
    with open(args.model) as f,open(join(tmpdir,'mymodel.py'),'w') as fout:
        for x in f:
            newline = x.replace('DATACODE',data_code)
            newline = newline.replace('TOPDIR',topdir)
            newline = newline.replace('DATASIZE',str(args.datasize))
            newline = newline.replace('MODEL_ARCH',model_arch)
            newline = newline.replace('PREFIX',args.prefix)
            fout.write(newline)

    sys.path.append(tmpdir)
    from mymodel import *
    import mymodel

    if args.hyper:
        ## Hyper-parameter tuning
        best_run, best_model = optim.minimize(model=mymodel.model,data=mymodel.data,algo=tpe.suggest,max_evals=int(args.hyperiter),trials=Trials())
        best_archit,best_optim,best_lossfunc = best_model
        open(architecture_file, 'w').write(best_archit)
        cPickle.dump((best_optim,best_lossfunc),open(optimizer_file,'wb') )

    if args.train:
        ### Training
        model = model_from_json(open(architecture_file).read())
        best_optim,best_lossfunc = cPickle.load(open(optimizer_file,'rb'))
        model.compile(loss=best_lossfunc, optimizer=best_optim,metrics=['accuracy'])

        checkpointer = ModelCheckpoint(filepath=weight_file, verbose=1, save_best_only=True)
        trainbatch_num,train_size = probedata(data1prefix+'.train.h5.batch')
        validbatch_num,valid_size = probedata(data1prefix+'.valid.h5.batch')
        history_callback = model.fit_generator(mymodel.BatchGenerator2(args.batchsize,trainbatch_num,'train',topdir,data_code)\
        		    ,train_size,args.trainepoch,validation_data=mymodel.BatchGenerator2(args.batchsize,validbatch_num,'valid',topdir,data_code)\
        			    ,nb_val_samples=valid_size,callbacks = [checkpointer])

        model.save_weights(last_weight_file, overwrite=True)
        system('touch '+join(outdir,model_arch+'.traindone'))
        myhist = history_callback.history
        all_hist = np.asarray([myhist["loss"],myhist["acc"],myhist["val_loss"],myhist["val_acc"]]).transpose()
        np.savetxt(join(outdir,model_arch+".training_history.txt"), all_hist,delimiter = "\t",header='loss\tacc\tval_loss\tval_acc')

    if args.retrain:
        ### Resume training
        model = model_from_json(open(architecture_file).read())
        model.load_weights(args.rweightfile)
        best_optim,best_lossfunc = cPickle.load(open(optimizer_file,'rb'))
        model.compile(loss=best_lossfunc, optimizer=best_optim,metrics=['accuracy'])

        new_weight_file = weight_file + '.'+args.retrain
        new_last_weight_file = last_weight_file + '.'+args.retrain

        checkpointer = ModelCheckpoint(filepath=new_weight_file, verbose=1, save_best_only=True)
        trainbatch_num,train_size = probedata(data1prefix+'.train.h5.batch')
        validbatch_num,valid_size = probedata(data1prefix+'.valid.h5.batch')
        history_callback = model.fit_generator(mymodel.BatchGenerator2(args.batchsize,trainbatch_num,'train',topdir,data_code)\
        		    ,train_size,args.trainepoch,validation_data=mymodel.BatchGenerator2(args.batchsize,validbatch_num,'valid',topdir,data_code)\
        			    ,nb_val_samples=valid_size,callbacks = [checkpointer])

        model.save_weights(new_last_weight_file, overwrite=True)
        system('touch '+join(outdir,model_arch+'.traindone'))
        myhist = history_callback.history
        all_hist = np.asarray([myhist["loss"],myhist["acc"],myhist["val_loss"],myhist["val_acc"]]).transpose()
        np.savetxt(join(outdir,model_arch+".training_history."+ args.retrain + ".txt"), all_hist,delimiter = "\t",header='loss\tacc\tval_loss\tval_acc')

    if args.eval:
        ## Evaluate
        model = model_from_json(open(architecture_file).read())
        model.load_weights(weight_file)
        best_optim,best_lossfunc = cPickle.load(open(optimizer_file,'rb'))
        model.compile(loss=best_lossfunc, optimizer=best_optim,metrics=['accuracy'])

        pred = np.asarray([])
        y_true = np.asarray([])
        testbatch_num = int(subprocess.check_output('ls '+data1prefix+'.test.h5.batch* | wc -l', shell=True).split()[0])
        for X1_train,Y_train in mymodel.BatchGenerator(testbatch_num,'test',topdir,data_code):
            pred = np.append(pred,[x[0] for x in model.predict(X1_train)])
            y_true = np.append(y_true,[x[0] for x in Y_train])

        t_auc = roc_auc_score(y_true,pred)
        t_acc = accuracy_score(y_true,[1 if x>0.5 else 0 for x in pred])
        print('Test AUC:',t_auc)
        print('Test accuracy:',t_acc)
        np.savetxt(evalout,[t_auc,t_acc])

    if args.infile != '':
        ## Predict on new data
        model = model_from_json(open(architecture_file).read())
        model.load_weights(weight_file)
        best_optim = cPickle.load(open(optimizer_file,'rb'))
        model.compile(loss='binary_crossentropy', optimizer=best_optim,metrics=['accuracy'])

        predict_batch_num = len([ 1  for x in subprocess.check_output('ls '+args.infile+'*', shell=True).split('\n')[:-1] if args.infile in x  if x.split(args.infile)[1].isdigit()])
        print('Total number of batch to predict:',predict_batch_num)

        outdir = join(dirname(args.infile),'.'.join(['pred',model_arch,basename(args.infile)])) if args.outdir == '' else args.outdir
        if exists(outdir):
            print('Output directory',outdir,'exists! Overwrite? (yes/no)')
            if raw_input().lower() == 'yes':
                system('rm -r ' + outdir)
            else:
                print('Quit predicting!')
                sys.exit(1)
        for i in range(predict_batch_num):
            print(i)
            data1 = h5py.File(args.infile+str(i+1),'r')['data']
            time1 = time.time()
            pred = model.predict(data1,batch_size=1280)
            time2 = time.time()
            print('predict took %0.3f ms' % ((time2-time1)*1000.0))

            t_outdir = join(outdir,'batch'+str(i+1))
            makedirs(t_outdir)
            for label_dim in range(pred.shape[1]):
                with open(join(t_outdir,str(label_dim)+'.pkl'),'wb') as f:
                    cPickle.dump(pred[:,label_dim],f)

    system('rm -r ' + tmpdir)
