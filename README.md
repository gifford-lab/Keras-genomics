A [Keras](https://keras.io/)-based deep learning platform to perform hyper-parameter tuning, training and prediction on genomics data.

## Dependencies
+ [Docker](https://www.docker.com/)
+ NVIDIA 346.46 driver
	+ If you want to run on Amazon EC2, we recommend using [EC2-launcher-pro](https://github.com/gifford-lab/ec2-launcher-pro) which lauches docker jobs on instance (ami-763a311e) with matched NVIDIA driver and GPU computing enviroment set up

## Quick run on the toy data
We prepare some toy data and toy model [here](https://github.com/gifford-lab/Keras-genomics/blob/master/example/). 

To perform a quick run, first convert the data to desired format and save under `$REPO_HOME/expt1`, where `$REPO_HOME` is the directory of the repository:

```
cd $REPO_HOME
for dtype in 'train' 'valid' 'test'
do
	paste - - -d' ' < example/$dtype.fa > tmp.tsv
	python embedH5.py tmp.tsv example/$dtype.target expt1/trial2.$dtype.h5
done
```

Then perform hyper-parameter tuning, training and testing by:

```
docker pull haoyangz/keras-genomics
docker run --rm --device /dev/nvidiactl --device /dev/nvidia-uvm --device /dev/nvidia0 \
    -v $(pwd)/example:/modeldir -v $(pwd)/expt1:/datadir haoyangz/keras-genomics \
	    python main.py -d /datadir -c trial2 -m /modeldir/model.py -s 101 -y -t -e
```
If everything works fine, you should get a test AUC around 0.86

## Data preparation
User needs to prepare [sequence file](https://github.com/gifford-lab/Keras-genomics/blob/master/example/train.fa) in [FASTA](https://en.wikipedia.org/wiki/FASTA_format) format and [target file](https://github.com/gifford-lab/Keras-genomics/blob/master/example/train.target) for training,validation and test set. Refer to the [toy data](https://github.com/gifford-lab/Keras-genomics/blob/master/example/) we provided for more examples.

Then run the following to embed each set into HDF5 format.
```
paste - - -d' ' < FASTA_FILE > tmp.tsv
python $REPO_HOME/embedH5.py tmp.tsv TARGET_FILE DATA_TOPDIR/DATA_CODE.SET_NAME.h5  -b BATCHSIZE
```
+ `FASTA_FILE`: sequence in FASTA format 
+ `TARGET_FILE`: targets (labels or real values) corresponding to the sequences (in the same order)
+ `DATA_TOPDIR`: the *absolute path* of the output directory 
+ `DATA_CODE`: a customized prefix to put at the begining of all the output HDF5 files
+ `SET_NAME`: 'train','valid',or 'test' for corresponding dataset. The main code below will search for training, validation and test data by this naming convention.
+ `BATCHSIZE`: optional and the default is 5000. Save every this number of samples to a separate file `DATA_CODE.h5.batchX` where X is the corresponding batch index.

## Model preparation
Change the `model` function in the [template](https://github.com/gifford-lab/Keras-genomics/blob/master/example/model.py) provided to implement your favorite network. Refer to [here](https://github.com/maxpumperla/hyperas) for instructions and examples of specifying hyper-parameters to tune.

## Perform hyper-parameter tuning, training and testing
We use Docker to free users from spending hours configuring the environment. But as the trade-off, it takes a long time to compile the model every time, although it won't affect the actual training time much. So below we provide instructions for running with and without Docker. 

#### Run with Docker (off-the-shelf)
```
docker pull haoyangz/keras-genomics
docker run --rm --device /dev/nvidiactl --device /dev/nvidia-uvm MOREDEVICE \
	-v MODEL_TOPDIR:/modeldir -v DATA_TOPDIR:/datadir haoyangz/keras-genomics \
	python main.py -d /datadir -c DATA_CODE -m /modeldir/MODEL_FILE_NAME -s SEQ_SIZE ORDER
```

+ `MODEL_TOPDIR`: the *absolute path* of the model file directory
+ `MODEL_FILE_NAME`: the filename of the model file
+ `DATA_TOPDIR`: same as above
+ `DATA_CODE`: same as above
+ `SEQ_SIZE`: the length of the genomic sequences
+ `ORDER`: 
	
	actions to take. *Multiple ones can be used and they will be executed in order*. 
	+ `-y [-hi 9]`: hyper-parameter tuning. Output will saved under "$DATA_TOPDIR/$MODEL_FILE_NAME".
		+	`-hi`: the number of hyper-parameter combinations to try (default:9)
	+ `-t [-te 20 -bs 100]`: train on the training set. Output will be saved in the same folder as `-y`.
	
		+	`-te`: the number of epochs to train for (default 20)
		+	`-bs`: the size of minibatch (default 100).
		+	The model for epoch with the smallest validation loss (best model) and the model for the last epoch (last model) will be saved.

	+ `-e`: evaluate the model on the test set. Output will be saved in the same folder as `-y`.
	+ `-p data_to_predict [-o output_folder]`: predict on new data.

		+	`data_to_predict`: should be the prefix of the embedded file up to the batch number. For example, assume we are to predict on some sequence data prepared at `/my_folder/mydata.batchX`, where X is 1,2,3,etc., then `data_to_predict` should be `/my_folder/mydata.batch`.
	
		+	`-o`: the output directory (default `/my_folder/pred.mymodel.mydata.batch`). Predictions for every batch will be saved to a separate subdirectory and split into different [pickle](https://wiki.python.org/moin/UsingPickle) files, one for each output neuron.
		
	+ `-r runcode -re weightfile`: resume training from a weight file
		+	`runcode`: the codename for this new run. The new model files will be the original ones plus `.runcode`. 
		+	`weightfile`: the weight file to resume training from.

+ `MOREDEVICE`: For each of the GPU device available on your machine, append one "--device /dev/nvidiaNUM" where NUM is the device index. For hsf1/hsf2 in  Gifford Lab, since there are three GPUs, it should be :
	
	```
		--device /dev/nvidia0 --device /dev/nvidia1 --device /dev/nvidia2
	```

#### Run without Docker (need to manually install packages)
Please refer to [here](https://github.com/gifford-lab/Keras-genomics/blob/master/Dockerfile)  and [here](https://hub.docker.com/r/haoyangz/keras-docker/~/dockerfile/) to configure your enviroment.
```
python $REPO_HOME/main.py -d DATA_TOPDIR -c DATA_CODE -m MODEL_TOPDIR/MODEL_FILE_NAME -s SEQ_SIZE ORDER
```

