from __future__ import print_function

import glob
import subprocess
from math import log, ceil
from os.path import join
from random import random
from time import time, ctime

import h5py
import numpy as np


class Hyperband:

    def __init__(self, get_params_function, try_params_function, datadir,
                 max_iter=81, eta=3, datamode='memory'):
        self.get_params = get_params_function
        self.try_params = try_params_function

        if datamode == 'memory':
            Y_train, X_train = self.readdata(join(datadir, 'train.h5.batch'))
            Y_test, X_test = self.readdata(join(datadir, 'valid.h5.batch'))
            self.data = {'train': (X_train, Y_train),
                         'valid': (X_test, Y_test)}
        else:
            self.data = {
                'train': {
                    'gen_func': self.BatchGenerator,
                    'path': join(datadir, 'train.h5.batch'),
                    'n_sample':
                        self.probedata(join(datadir, 'train.h5.batch'))[1]},
                'valid': {
                    'gen_func': self.BatchGenerator,
                    'path': join(datadir, 'valid.h5.batch'),
                    'n_sample':
                        self.probedata(join(datadir, 'valid.h5.batch'))[1]},
            }

        self.datamode = datamode
        self.max_iter = max_iter  # maximum iterations per configuration
        self.eta = eta  # defines configuration downsampling rate (default = 3)

        self.logeta = lambda x: log(x) / log(self.eta)
        self.s_max = int(self.logeta(self.max_iter))
        self.B = (self.s_max + 1) * self.max_iter

        self.results = []  # list of dicts
        self.counter = 0
        self.best_loss = np.inf
        self.best_counter = -1

    # can be called multiple times
    def run(self, skip_last=0, dry_run=False):

        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s + 1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            T = [self.get_params() for i in range(n)]

            for i in range((s + 1) - int(skip_last)):  # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)

                print(
                    "\n*** {} configurations x {:.1f} iterations each".format(
                        n_configs, n_iterations))

                val_losses = []
                early_stops = []

                for t in T:

                    self.counter += 1
                    print(
                        "\n{} | {} | lowest loss so far: {:.4f} (run {})\n".format(
                            self.counter, ctime(), self.best_loss,
                            self.best_counter))

                    start_time = time()

                    if dry_run:
                        result = {'loss': random(), 'log_loss': random(),
                                  'auc': random()}
                    else:
                        result = self.try_params(n_iterations, t, self.data,
                                                 self.datamode)  # <---

                    assert (type(result) == dict)
                    assert ('loss' in result)

                    seconds = int(round(time() - start_time))
                    print("\n{} seconds.".format(seconds))

                    loss = result['loss']
                    val_losses.append(loss)

                    early_stop = result.get('early_stop', False)
                    early_stops.append(early_stop)

                    # keeping track of the best result so far (for display only)
                    # could do it be checking results each time, but hey
                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_counter = self.counter

                    result['counter'] = self.counter
                    result['seconds'] = seconds
                    result['params'] = t
                    result['iterations'] = n_iterations

                    self.results.append(result)

                # select a number of best configurations for the next loop
                # filter out early stops, if any
                indices = np.argsort(val_losses)
                T = [T[i] for i in indices if not early_stops[i]]
                T = T[0:int(n_configs / self.eta)]

        return self.results

    def readdata(self, dataprefix):
        globber = dataprefix + '*'
        allfiles = glob.iglob(globber)
        cnt = 0
        samplecnt = 0
        for x in allfiles:
            print("Reading " + x + " ...")
            if x.split(dataprefix)[1].isdigit():
                cnt += 1
                dataall = h5py.File(x, 'r')
                if cnt == 1:
                    label = np.asarray(dataall.get('label'))
                    data = np.asarray(dataall.get('data'))
                else:
                    label = np.vstack((label, np.asarray(dataall.get('label'))))
                    data = np.vstack((data, np.asarray(dataall.get('data'))))
        if cnt == 0:
            raise ValueError("No files found starting with " + dataprefix)
        return (label, data)

    def BatchGenerator(self, mb_size, fileprefix, shuf=True):
        globber = fileprefix + '*'
        allfiles = glob.glob(globber)
        n_files = len(allfiles)
        cache = []
        while True:
            idx2use = np.random.permutation(np.arange(n_files)) if \
                shuf else np.arange(n_files)
            for i in idx2use:
                data1f = h5py.File(fileprefix + str(i + 1), 'r')
                data1 = data1f['data'][()]
                label = data1f['label'][()]
                datalen = len(data1)
                if shuf:
                    reorder = np.random.permutation(np.arange(datalen))
                    data1 = data1[reorder]
                    label = label[reorder]
                minibatch_size = mb_size or datalen
                idx = 0
                if len(cache) != 0:
                    idx = minibatch_size - len(cache)
                    yield ([np.vstack((cache[0], data1[:idx])),
                            np.vstack((cache[1], label[:idx]))])
                while idx + minibatch_size <= datalen:
                    idx += minibatch_size
                    yield ([data1[(idx - minibatch_size):idx],
                            label[(idx - minibatch_size):idx]])
                if idx < datalen:
                    cache = [data1[idx:], label[idx:]]

    def probedata(self, dataprefix):
        allfiles = glob.iglob(dataprefix + "*")
        cnt = 0
        samplecnt = 0
        for x in allfiles:
            if x.split(dataprefix)[1].isdigit():
                cnt += 1
                data = h5py.File(x, 'r')
                samplecnt += len(data['label'])
        return (cnt, samplecnt)
