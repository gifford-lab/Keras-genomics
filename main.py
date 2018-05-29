from __future__ import print_function

import numpy as np
import sys
import time
from os import system, makedirs
from os.path import join, dirname, basename, exists, realpath
from pprint import pprint
from tempfile import mkdtemp

import argparse
try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle
import h5py
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import model_from_json
from sklearn.metrics import accuracy_score, roc_auc_score

from hyperband import Hyperband

cwd = dirname(realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Keras + Hyperband for genomics")
    parser.add_argument("-y", "--hyper", dest="hyper", default=False,
                        action='store_true',
                        help="Perform hyper-parameter tuning")
    parser.add_argument("-t", "--train", dest="train", default=False,
                        action='store_true',
                        help="Train on the training set with the best "
                             "hyper-params")
    parser.add_argument("-e", "--eval", dest="eval", default=False,
                        action='store_true',
                        help="Evaluate the model on the test set")
    parser.add_argument("-p", "--predit", dest="infile", default='',
                        help="Path to data to predict on (up till "
                             "batch number)")
    parser.add_argument("-d", "--topdir", dest="topdir",
                        help="The data directory")
    parser.add_argument("-m", "--model", dest="model",
                        help="Path to the model file")
    parser.add_argument("-o", "--outdir", dest="outdir", default='',
                        help="Output directory for the prediction on new data")
    parser.add_argument("-hi", "--hyperiter", dest="hyperiter", default=20,
                        type=int,
                        help="Num of max iteration for each hyper-param "
                             "config")
    parser.add_argument("-te", "--trainepoch", default=20, type=int,
                        help="The number of epochs to train for")
    parser.add_argument("-pa", "--patience", default=10, type=int,
                        help="number of epochs with no improvement after which"
                             " training will be stopped.")
    parser.add_argument("-bs", "--batchsize", default=100, type=int,
                        help="Batchsize in SGD-based training")
    parser.add_argument("-w", "--weightfile", default=None,
                        help="Weight file for the best model")
    parser.add_argument("-l", "--lweightfile", default=None,
                        help="Weight file after training")
    parser.add_argument("-r", "--retrain", default=None,
                        help="codename for the retrain run")
    parser.add_argument("-rw", "--rweightfile", default='',
                        help="Weight file to load for retraining")
    parser.add_argument("-dm", "--datamode", default='memory',
                        help="whether to load data into memory ('memory') or "
                             "using a generator('generator')")
    parser.add_argument("-ei", "--evalidx", dest='evalidx', default=0,
                        type=int,
                        help="which output neuron (0-based) to calculate "
                             "2-class auROC for")
    parser.add_argument("--epochratio", default=1, type=float,
                        help="when training with data generator, optionally "
                             "shrink each epoch size by this factor to enable "
                             "more frequen evaluation on the valid set")
    parser.add_argument("-shuf", default=1, type=int,
                        help="whether to shuffle the data at the begining of "
                             "each epoch (1/0)")

    return parser.parse_args()


def train_func(model, weightfile2save):
    checkpointer = ModelCheckpoint(filepath=weightfile2save, verbose=1,
                                   save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=args.patience,
                                   verbose=0)
    if args.datamode == 'generator':
        trainbatch_num, train_size = hb.probedata(
            join(args.topdir, 'train.h5.batch'))
        validbatch_num, valid_size = hb.probedata(
            join(args.topdir, 'valid.h5.batch'))
        history_callback = model.fit_generator(
            hb.BatchGenerator(args.batchsize,
                              join(args.topdir, 'train.h5.batch'),
                              shuf=args.shuf == 1),
            train_size / args.batchsize * args.epochratio,
            args.trainepoch,
            validation_data=hb.BatchGenerator(args.batchsize, join(
                args.topdir, 'valid.h5.batch'),
                                              shuf=args.shuf == 1),
            validation_steps=np.ceil(float(valid_size) / args.batchsize),
            callbacks=[checkpointer, early_stopping])
    else:
        Y_train, traindata = hb.readdata(join(args.topdir, 'train.h5.batch'))
        Y_valid, validdata = hb.readdata(join(args.topdir, 'valid.h5.batch'))
        history_callback = model.fit(
            traindata,
            Y_train,
            batch_size=args.batchsize,
            epochs=args.trainepoch,
            validation_data=(validdata, Y_valid),
            callbacks=[checkpointer, early_stopping],
            shuffle=args.shuf == 1)
    return model, history_callback


def load_model(weightfile2load=None):
    model = model_from_json(open(architecture_file).read())
    if weightfile2load:
        model.load_weights(weightfile2load)
    best_optim, best_optim_config, best_lossfunc = pickle.load(
        open(optimizer_file, 'rb'))
    model.compile(loss=best_lossfunc,
                  optimizer=best_optim.from_config(best_optim_config),
                  metrics=['categorical_accuracy'])
    return model


if __name__ == "__main__":

    args = parse_args()
    model_arch = basename(args.model)
    model_arch = model_arch[:-3] if model_arch[-3:] == '.py' else model_arch

    outdir = join(args.topdir, model_arch)
    if not exists(outdir):
        makedirs(outdir)

    architecture_file = join(outdir, model_arch + '_best_archit.json')
    optimizer_file = join(outdir, model_arch + '_best_optimer.pkl')
    weight_file = join(outdir,
                       model_arch + '_bestmodel_weights.h5') \
        if args.weightfile is None else args.weightfile
    last_weight_file = join(outdir,
                            model_arch + '_lastmodel_weights.h5') \
        if args.lweightfile is None else args.lweightfile
    evalout = join(outdir, model_arch + '_eval.txt')

    tmpdir = mkdtemp()
    system(' '.join(['cp', args.model, join(tmpdir, 'mymodel.py')]))
    sys.path.append(tmpdir)
    import mymodel

    hb = Hyperband(mymodel.get_params, mymodel.try_params, args.topdir,
                   max_iter=args.hyperiter, datamode=args.datamode)

    if args.hyper:
        ## Hyper-parameter tuning
        results = hb.run(skip_last=1)

        best_result = sorted(results, key=lambda x: x['loss'])[0]
        pprint(best_result['params'])

        best_archit, best_optim, best_optim_config, best_lossfunc = \
        best_result['model']
        open(architecture_file, 'w').write(best_archit)
        pickle.dump((best_optim, best_optim_config, best_lossfunc),
                     open(optimizer_file, 'wb'))

    if args.train:
        ### Training
        model = load_model()
        model, history_callback = train_func(model, weight_file)

        model.save_weights(last_weight_file, overwrite=True)
        system('touch ' + join(outdir, model_arch + '.traindone'))
        myhist = history_callback.history
        all_hist = np.asarray([myhist["loss"], myhist["categorical_accuracy"],
                               myhist["val_loss"],
                               myhist["val_categorical_accuracy"]]).transpose()
        np.savetxt(join(outdir, model_arch + ".training_history.txt"),
                   all_hist, delimiter="\t",
                   header='loss\tacc\tval_loss\tval_acc')

    if args.retrain:
        ### Resume training
        new_weight_file = weight_file + '.' + args.retrain
        new_last_weight_file = last_weight_file + '.' + args.retrain

        model = load_model(args.rweightfile)
        model, history_callback = train_func(model, new_weight_file)

        model.save_weights(new_last_weight_file, overwrite=True)
        system('touch ' + join(outdir, model_arch + '.traindone'))
        myhist = history_callback.history
        all_hist = np.asarray([myhist["loss"], myhist["categorical_accuracy"],
                               myhist["val_loss"],
                               myhist["val_categorical_accuracy"]]).transpose()
        np.savetxt(join(outdir,
                        model_arch + ".training_history." + args.retrain + ".txt"),
                   all_hist, delimiter="\t",
                   header='loss\tacc\tval_loss\tval_acc')

    if args.eval:
        ## Evaluate
        model = load_model(weight_file)

        pred_for_evalidx = []
        pred_bin = []
        y_true_for_evalidx = []
        y_true = []
        testbatch_num, _ = hb.probedata(join(args.topdir, 'test.h5.batch'))
        test_generator = hb.BatchGenerator(None,
                                           join(args.topdir, 'test.h5.batch'),
                                           shuf=args.shuf == 1)
        for _ in range(testbatch_num):
            try:
                X_test, Y_test = test_generator.next()
            except AttributeError:
                X_test, Y_test = next(test_generator)
            t_pred = model.predict(X_test)
            pred_for_evalidx += [x[args.evalidx] for x in t_pred]
            pred_bin += [np.argmax(x) for x in t_pred]
            y_true += [np.argmax(x) for x in Y_test]
            y_true_for_evalidx += [x[args.evalidx] for x in Y_test]

        t_auc = roc_auc_score(y_true_for_evalidx, pred_for_evalidx)
        t_acc = accuracy_score(y_true, pred_bin)
        print('Test AUC for output neuron {}:'.format(args.evalidx),
                       t_auc)
        print('Test categorical accuracy:', t_acc)
        np.savetxt(evalout, [t_auc, t_acc])

    if args.infile != '':
        ## Predict on new data
        model = load_model(weight_file)

        predict_batch_num, _ = hb.probedata(args.infile)
        print('Total number of batch to predict:', predict_batch_num)

        outdir = join(dirname(args.infile), '.'.join(['pred', model_arch,
                                                      basename(
                                                          args.infile)])) if args.outdir == '' else args.outdir
        if exists(outdir):
            print('Output directory', outdir,
                           'exists! Overwrite? (yes/no)')
            if raw_input().lower() == 'yes':
                system('rm -r ' + outdir)
            else:
                print('Quit predicting!')
                sys.exit(1)

        for i in range(predict_batch_num):
            print('predict on batch', i)
            batch_data = h5py.File(args.infile + str(i + 1), 'r')['data']

            time1 = time.time()
            pred = model.predict(batch_data)
            time2 = time.time()
            print(
                'predict took %0.3f ms' % ((time2 - time1) * 1000.0))

            t_outdir = join(outdir, 'batch' + str(i + 1))
            makedirs(t_outdir)
            for label_dim in range(pred.shape[1]):
                with open(join(t_outdir, str(label_dim) + '.pkl'), 'wb') as f:
                    pickle.dump(pred[:, label_dim], f)

    system('rm -r ' + tmpdir)
