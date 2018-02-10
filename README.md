A [Keras](https://keras.io/)-based deep learning platform to perform hyper-parameter tuning, training and prediction on genomics data.

Table of contents
=================

<!--ts-->
* [Data preparation](#data-preparation)
* [Model preparation](#model-preparation)
* [Running the model](#running-the-model)
* [Run on toy data](#quick-run-on-the-toy-data)
<!--te-->

## Notice of major refactorization

The latest version has gone through major refactorization that changes the interface substantially, now using [Hyperband](https://github.com/zygmuntz/hyperband) to optimize the hyperparameter space. To use the old version, please download release 0.1 from [here](https://github.com/gifford-lab/Keras-genomics/releases) or checkout the README at [here](https://github.com/gifford-lab/Keras-genomics/tree/v0.1).


## Data preparation
User needs to prepare [sequence file](https://github.com/gifford-lab/Keras-genomics/blob/master/example/train.fa) in [FASTA](https://en.wikipedia.org/wiki/FASTA_format) format and [target file](https://github.com/gifford-lab/Keras-genomics/blob/master/example/train.target) for training,validation and test set. Refer to the [toy data](https://github.com/gifford-lab/Keras-genomics/blob/master/example/) we provided for more examples.

Then run the following to embed each set into HDF5 format.
```
paste - - -d' ' < FASTA_FILE > tmp.tsv
python $REPO_HOME/embedH5.py tmp.tsv TARGET_FILE DATA_TOPDIR/FILE_NAME  -b BATCHSIZE
```
+ `FASTA_FILE`: sequence in FASTA format 
+ `TARGET_FILE`: targets (labels or real values) corresponding to the sequences (in the same order)
+ `DATA_TOPDIR`: the *absolute path* of the output directory 
+ `FILE_NAME`: 'train.h5.batch','valid.h5.batch',or 'test.h5.batch' for training, validation and test set.
+ `BATCHSIZE`: optional and the default is 5000. Save every this number of samples to a separate file `DATA_CODE.h5.batchX` where X is the corresponding batch index.

## Model preparation
Change the `model` function in the [template](https://github.com/gifford-lab/Keras-genomics/blob/master/example/model.py) provided to implement your favorite network. Refer to [here](https://github.com/zygmuntz/hyperband/blob/master/defs/keras_mlp.py) for examples of how to specifying hyper-parameters to tune.

## Running the model

```
python main.py -d DATA_TOPDIR -m MODEL_FILE_NAME ORDER
```

+ `DATA_TOPDIR`: same as above.
+ `MODEL_FILE`: the model file prepared.
+ `ORDER`: actions to take. *Multiple ones can be used and they will be executed in order*. 

	+ `-y`: hyper-parameter tuning. Output will saved under "$DATA_TOPDIR/$MODELNAME", where `MODELNAME` is the base name of the model file (minus the ".py" at the end if there is one).
	
		Optional:
		+	`-hi`: the max number of iterations for each hyper-parameter combinations (default:20)
		+	`-dm`: 'memory' for loading all the data into memory and 'generator' for using a Python generator to load the data in batch (default: 'memory').
	+ `-t`: train on the training set. Output will be saved in the same folder as `-y`. After early stopping or reaching the maximum number of epoches specified, the model from the epoch with the smallest validation loss (best model) and the model from the last epoch (last model) will be saved.
	
		Optional:
		+	`-te`: the number of epochs to train for (default 20)
		+	`-bs`: the size of minibatch (default 100).
		+   `-dm`: same as above.
		+   `-pa`: number of epochs with no improvement in validation loss after which training will be stopped (default 10).
		+   `-w`: the file to save the weight of the best model at (default: $DATA_TOPDIR/$MODELNAME/${MODELNAME}_bestmodel_weights.h5).
		+   `-l`: the file to save the weight of the last model at (default: $DATA_TOPDIR/$MODELNAME/${MODELNAME}_lastmodel_weights.h5).

	+ `-e`: evaluate the model on the test set. Output will be saved in the same folder as `-y`.
	+ `-p data_to_predict`: predict on new data.`data_to_predict` should be the prefix of the embedded file to predict on up to the batch number. For example, assume we are to predict on some sequence data prepared at `/my_folder/mydata.batchX`, where X is 1,2,3,etc., then `data_to_predict` should be `/my_folder/mydata.batch`.
	
		Optional:	
		+	`-o`: the output directory (default `/my_folder/pred.mymodel.mydata.batch`). Predictions for every batch will be saved to a separate subdirectory and split into different [pickle](https://wiki.python.org/moin/UsingPickle) files, one for each output neuron.
		
	+ `-r runcode -re weightfile`: resume training from a weight file
		+	`runcode`: the codename for this new run. The new model files will be the original ones plus `.runcode`. 
		+	`weightfile`: the weight file to resume training from.


## Quick run on the toy data
We prepare some toy data and toy model [here](https://github.com/gifford-lab/Keras-genomics/blob/master/example/). 

To perform a quick run, first run the following command to convert the data to desired format and save under "expt1" in the current folder.

```
cd $REPO_HOME
for dtype in 'train' 'valid' 'test'
do
	paste - - -d' ' < example/$dtype.fa > tmp.tsv
	python embedH5.py tmp.tsv example/$dtype.target expt1/$dtype.h5
done
```

Then perform hyper-parameter tuning, training and testing by:

```
python main.py -d expt1 -m example/model.py -y -t -e
```
All the intermediate output will be under "expt1". If everything works fine, you should get a test AUC around 0.97
