from __future__ import print_function

import argparse
import os
import pwd

from Bio import SeqIO
import h5py
import numpy as np
import pandas as pd

# Python 2/3 compatibility
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    # Python 3's zip is a lazy iterator by default
    pass

from os import makedirs
from os.path import exists, dirname, join, basename


def outputHDF5(data, label, filename, labelname, dataname):
    print('data shape: ', data.shape)
    comp_kwargs = {'compression': 'gzip', 'compression_opts': True}
    # label = [[x.astype(np.float32)] for x in label]
    with h5py.File(filename, 'w') as f:
        f.create_dataset(dataname, data=data, **comp_kwargs)
        f.create_dataset(labelname, data=label, **comp_kwargs)


def seq2feature(data, mapper, label, out_filename, worddim, labelname,
                dataname):
    out = []
    for seq in data:
        mat = embed(seq, mapper, worddim)
        result = mat.transpose()
        result1 = [[a] for a in result]
        out.append(result1)
    outputHDF5(np.asarray(out), label, out_filename, labelname, dataname)


def feature2feature(data, mapper, label, out_filename, worddim, labelname,
                    dataname):
    out = np.asarray(data)[:, np.nan, np.nan, :]
    outputHDF5(out, label, out_filename, labelname, dataname)


def embed(seq, mapper, worddim):
    mat = np.asarray([mapper[element] if element in mapper else np.random.rand(
        worddim) * 2 - 1 for element in seq])
    return mat


def seq2feature_siamese(data1, data2, mapper, label, out_filename, worddim,
                        labelname, dataname):
    out = []
    datalen = len(data1)
    for dataidx in range(datalen):
        mat = np.asarray([embed(data1[dataidx], mapper, worddim),
                          embed(data2[dataidx], mapper, worddim)])
        result = mat.transpose((2, 0, 1))
        out.append(result)
    outputHDF5(np.asarray(out), label, out_filename, labelname, dataname)


def _read_fasta(filename):
    return pd.DataFrame([[record.id, str(record.seq)] for record in
                             SeqIO.parse(filename, 'fasta')])

def convert(in_filename, label_filename, outfile, mapper, worddim,
            batchsize, labelname, dataname, isseq):
    try:
        seqs = pd.read_table(in_filename, header=None, sep='\s+')
    except pd.errors.ParserError:
        seqs = _read_fasta(in_filename)

    if len(seqs.columns) == 1:
        # This is actually a fasta file, so convert to a table
        # for future use
        seqs = _read_fasta(in_filename)


    target = pd.read_table(label_filename, header=None, sep='\s+')

    n_samples = len(target.index)
    n_batches = int(np.ceil(float(n_samples) / batchsize))

    # Create batch column for selection
    batch_iter = range(1, n_batches+1)
    batch_array = np.concatenate(
        [np.ones(batchsize, dtype=int) * i for i in batch_iter])

    print('batch_array.shape - before slicing', batch_array.shape)

    # Only take as many samples as we need
    batch_array = batch_array[:n_samples]

    print('batch_array.shape - after slicing', batch_array.shape)
    print('target.shape', target.shape)
    print('seqs.shape', seqs.shape)
    target['batch'] = batch_array
    seqs['batch'] = batch_array

    for batch_num, batch_seqs in seqs.groupby('batch'):
        batch_target = target.query('batch == @batch_num')

        batch_outfile = outfile + '.batch' + str(batch_num)

        seq_data = batch_seqs[1].map(lambda x: np.asarray(list(x)))
        target_data = batch_target[[0, 1]].values

        if isseq:
            seq2feature(seq_data, mapper, target_data, batch_outfile, worddim,
                        labelname, dataname)
        else:
            feature2feature(seq_data, mapper, target_data, batch_outfile, worddim,
                            labelname, dataname)
    return batch_num


def convert_siamese(infile1, infile2, labelfile, outfile, mapper, worddim,
                    batchsize, labelname, dataname):
    with open(infile1) as seqfile1, open(infile2) as seqfile2, open(
            labelfile) as labelfile:
        cnt = 0
        seqdata1 = []
        seqdata2 = []
        label = []
        batchnum = 0
        for x1, x2, y in zip(seqfile1, seqfile2, labelfile):
            seqdata1.append(list(x1.strip().split()[1]))
            seqdata2.append(list(x2.strip().split()[1]))
            # label.append(float(y.strip()))
            label.append(list(map(float, y.strip().split())))
            cnt = (cnt + 1) % batchsize
            if cnt == 0:
                batchnum = batchnum + 1
                seqdata1 = np.asarray(seqdata1)
                seqdata2 = np.asarray(seqdata2)
                label = np.asarray(label)
                t_outfile = outfile + '.batch' + str(batchnum)
                seq2feature_siamese(seqdata1, seqdata2, mapper, label,
                                    t_outfile, worddim, labelname, dataname)
                seqdata1 = []
                seqdata2 = []
                label = []

        if cnt > 0:
            batchnum = batchnum + 1
            seqdata1 = np.asarray(seqdata1)
            seqdata2 = np.asarray(seqdata2)
            label = np.asarray(label)
            t_outfile = outfile + '.batch' + str(batchnum)
            seq2feature_siamese(seqdata1, seqdata2, mapper, label, t_outfile,
                                worddim, labelname, dataname)

    return batchnum


def manifest(out_filename, batchnum, prefix):
    locfile = join(dirname(out_filename),
                   basename(out_filename).split('.')[0] + '.txt')
    with open(locfile, 'w') as f:
        for i in range(batchnum):
            f.write('.'.join(
                ['/'.join([prefix] + out_filename.split('/')[-2:]),
                 'batch' + str(i + 1)]) + '\n')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert sequence and target for Caffe")
    user = pwd.getpwuid(os.getuid())[0]

    # Positional (unnamed) arguments:
    parser.add_argument("infile", type=str,
                        help="Sequence in FASTA/TSV format (with .fa/.fasta or .tsv extension)")
    parser.add_argument("labelfile", type=str,
                        help="Label of the sequence. One number per line")
    parser.add_argument("outfile", type=str,
                        help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")

    # Optional arguments:
    parser.add_argument("-m", "--mapperfile", dest="mapperfile", default="",
                        help="A TSV file mapping each nucleotide to a vector. "
                             "The first column should be the nucleotide, and "
                             "the rest denote the vectors. "
                             "(Default mapping: A:[1,0,0,0],C:[0,1,0,0],"
                             "G:[0,0,1,0],T:[0,0,0,1])")
    parser.add_argument("-i", "--infile2", dest="infile2", default="",
                        help="The paired input file for siamese network")
    parser.add_argument("-b", "--batch", dest="batch", type=int, default=5000,
                        help="Batch size for data storage (Defalt:5000)")
    parser.add_argument("-p", "--prefix", dest="maniprefix", default='/data',
                        help="The model_dir (Default: /data . This only "
                             "works for mri-wrapper)")
    parser.add_argument("-l", "--labelname", dest="labelname", default='label',
                        help="The group name for labels in the HDF5 file")
    parser.add_argument("-d", "--dataname", dest="dataname", default='data',
                        help="The group name for data in the HDF5 file")
    parser.add_argument("-s", "--isseq", dest="isseq", default='Y',
                        help="The group name for data in the HDF5 file")

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    outdir = dirname(args.outfile)
    if not exists(outdir):
        makedirs(outdir)

    print("args:", args)

    if args.mapperfile == "":
        args.mapper = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0],
                       'T': [0, 0, 0, 1], 'N': [0, 0, 0, 0]}
    else:
        args.mapper = {}
        with open(args.mapperfile, 'r') as f:
            for x in f:
                line = x.strip().split()
                word = line[0]
                vec = [float(item) for item in line[1:]]
                args.mapper[word] = vec
    if args.infile2 == '':
        print(args.isseq == 'Y')
        batchnum = convert(args.infile, args.labelfile, args.outfile,
                           args.mapper, len(args.mapper['A']), args.batch,
                           args.labelname, args.dataname, args.isseq == 'Y')
    else:
        batchnum = convert_siamese(args.infile, args.infile2, args.labelfile,
                                   args.outfile, args.mapper,
                                   len(args.mapper['A']), args.batch,
                                   args.labelname, args.dataname)
    manifest(args.outfile, batchnum, args.maniprefix)
