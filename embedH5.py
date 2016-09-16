import argparse,pwd,os,numpy as np,h5py
from os.path import splitext,exists,dirname,join,basename
from os import makedirs
from itertools import izip

def outputHDF5(data,label,filename,labelname,dataname):
    print 'data shape: ',data.shape
    comp_kwargs = {'compression': 'gzip', 'compression_opts': 1}
    #label = [[x.astype(np.float32)] for x in label]
    with h5py.File(filename, 'w') as f:
    	f.create_dataset(dataname, data=data, **comp_kwargs)
    	f.create_dataset(labelname, data=label, **comp_kwargs)

def seq2feature(data,mapper,label,out_filename,worddim,labelname,dataname):
    out = []
    for seq in data:
        mat = embed(seq,mapper,worddim)
        result = mat.transpose()
        result1 = [ [a] for a in result]
        out.append(result1)
    outputHDF5(np.asarray(out),label,out_filename,labelname,dataname)

def feature2feature(data,mapper,label,out_filename,worddim,labelname,dataname):
    out = np.asarray(data)[:,None,None,:]
    outputHDF5(out,label,out_filename,labelname,dataname)


def embed(seq,mapper,worddim):
    mat = np.asarray([mapper[element] if element in mapper else np.random.rand(worddim)*2-1 for element in seq])
    return mat

def seq2feature_siamese(data1,data2,mapper,label,out_filename,worddim,labelname,dataname):
    out = []
    datalen = len(data1)
    for dataidx in range(datalen):
        mat = np.asarray([embed(data1[dataidx],mapper,worddim),embed(data2[dataidx],mapper,worddim)])
        result = mat.transpose((2,0,1))
        out.append(result)
    outputHDF5(np.asarray(out),label,out_filename,labelname,dataname)

def convert(infile,labelfile,outfile,mapper,worddim,batchsize,labelname,dataname,isseq):
    with open(infile) as seqfile, open(labelfile) as labelfile:
        cnt = 0
        seqdata = []
        label = []
        batchnum = 0
        for x,y in izip(seqfile,labelfile):
            if isseq:
                seqdata.append(list(x.strip().split()[1]))
            else:
                seqdata.append(map(float,x.strip().split()))
            #label.append(float(y.strip()))
            label.append(map(float,y.strip().split()))
            cnt = (cnt+1)% batchsize
            if cnt == 0:
                batchnum = batchnum + 1
                seqdata = np.asarray(seqdata)
                label = np.asarray(label)
                t_outfile = outfile + '.batch' + str(batchnum)
                if isseq:
                    seq2feature(seqdata,mapper,label,t_outfile,worddim,labelname,dataname)
                else:
                    feature2feature(seqdata,mapper,label,t_outfile,worddim,labelname,dataname)
                seqdata = []
                label = []
        if cnt >0:
            batchnum = batchnum + 1
            seqdata = np.asarray(seqdata)
            label = np.asarray(label)
            t_outfile = outfile + '.batch' + str(batchnum)
            if isseq:
                seq2feature(seqdata,mapper,label,t_outfile,worddim,labelname,dataname)
            else:
                feature2feature(seqdata,mapper,label,t_outfile,worddim,labelname,dataname)
    return batchnum


def convert_siamese(infile1,infile2,labelfile,outfile,mapper,worddim,batchsize,labelname,dataname):
    with open(infile1) as seqfile1, open(infile2) as seqfile2,open(labelfile) as labelfile:
        cnt = 0
        seqdata1 = []
        seqdata2 = []
        label = []
        batchnum = 0
        for x1,x2,y in izip(seqfile1,seqfile2,labelfile):
            seqdata1.append(list(x1.strip().split()[1]))
            seqdata2.append(list(x2.strip().split()[1]))
            #label.append(float(y.strip()))
            label.append(map(float,y.strip().split()))
            cnt = (cnt+1)% batchsize
            if cnt == 0:
                batchnum = batchnum + 1
                seqdata1 = np.asarray(seqdata1)
                seqdata2 = np.asarray(seqdata2)
                label = np.asarray(label)
                t_outfile = outfile + '.batch' + str(batchnum)
                seq2feature_siamese(seqdata1,seqdata2,mapper,label,t_outfile,worddim,labelname,dataname)
                seqdata1 = []
                seqdata2 = []
                label = []

        if cnt > 0:
            batchnum = batchnum + 1
            seqdata1 = np.asarray(seqdata1)
            seqdata2 = np.asarray(seqdata2)
            label = np.asarray(label)
            t_outfile = outfile + '.batch' + str(batchnum)
            seq2feature_siamese(seqdata1,seqdata2,mapper,label,t_outfile,worddim,labelname,dataname)

    return batchnum

def manifest(out_filename,batchnum,prefix):
    locfile = join(dirname(out_filename),basename(out_filename).split('.')[0] + '.txt')
    with open(locfile,'w') as f:
        for i in range(batchnum):
            f.write('.'.join(['/'.join([prefix]+out_filename.split('/')[-2:]),'batch'+str(i+1)])+'\n')


def parse_args():
    parser = argparse.ArgumentParser(description="Convert sequence and target for Caffe")
    user = pwd.getpwuid(os.getuid())[0]

    # Positional (unnamed) arguments:
    parser.add_argument("infile",  type=str, help="Sequence in FASTA/TSV format (with .fa/.fasta or .tsv extension)")
    parser.add_argument("labelfile",  type=str,help="Label of the sequence. One number per line")
    parser.add_argument("outfile",  type=str, help="Output file (example: $MODEL_TOPDIR$/data/train.h5). ")

    # Optional arguments:
    parser.add_argument("-m", "--mapperfile", dest="mapperfile", default="", help="A TSV file mapping each nucleotide to a vector. The first column should be the nucleotide, and the rest denote the vectors. (Default mapping: A:[1,0,0,0],C:[0,1,0,0],G:[0,0,1,0],T:[0,0,0,1])")
    parser.add_argument("-i", "--infile2", dest="infile2", default="", help="The paired input file for siamese network")
    parser.add_argument("-b", "--batch", dest="batch", type=int,default=5000, help="Batch size for data storage (Defalt:5000)")
    parser.add_argument("-p", "--prefix", dest="maniprefix",default='/data', help="The model_dir (Default: /data . This only works for mri-wrapper)")
    parser.add_argument("-l", "--labelname", dest="labelname",default='label', help="The group name for labels in the HDF5 file")
    parser.add_argument("-d", "--dataname", dest="dataname",default='data', help="The group name for data in the HDF5 file")
    parser.add_argument("-s", "--isseq", dest="isseq",default='Y', help="The group name for data in the HDF5 file")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()

    outdir = dirname(args.outfile)
    if not exists(outdir):
        makedirs(outdir)

    if args.mapperfile == "":
        args.mapper = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1],'N':[0,0,0,0]}
    else:
        args.mapper = {}
        with open(args.mapperfile,'r') as f:
            for x in f:
                line = x.strip().split()
                word = line[0]
                vec = [float(item) for item in line[1:]]
                args.mapper[word] = vec
    if args.infile2 == '':
        print args.isseq =='Y'
        batchnum = convert(args.infile,args.labelfile,args.outfile,args.mapper,len(args.mapper['A']),args.batch,args.labelname,args.dataname,args.isseq=='Y')
    else:
        batchnum = convert_siamese(args.infile,args.infile2,args.labelfile,args.outfile,args.mapper,len(args.mapper['A']),args.batch,args.labelname,args.dataname)
    manifest(args.outfile,batchnum,args.maniprefix)
