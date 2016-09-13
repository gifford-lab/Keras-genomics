```
docker pull haoyangz/keras-genomics
docker run --rm --device /dev/nvidiactl --device /dev/nvidia-uvm MOREDEVICE -v MODEL_TOPDIR:/modeldir -v DATA_TOPDIR:/datadir haoyangz/keras-genomics python main.py -d /datadir -c DATA_CODE -m /modeldir/MODEL_FILE_NAME -s SEQ_SIZE ORDER
```

+ MODEL_TOPDIR: the *absolute path* of the model file directory
+ MODEL_FILE_NAME: the filename of the model file
+ DATA_TOPDIR: the *absolute path* of the directory containing all the data
+ DATA_CODE: the prefix of all the data files.
+ SEQ_SIZE: the length of the genomic sequences
+ ORDER:
	+ `-y`: hyper-parameter tuning. Use `-hi` to change the number of hyper-parameter combinations to try (default:9)
	+ `-t`: train. Use `-te` to change the number of epochs to train for.
	+ `-e`: evaluate the model on the test set.
	+ `-p`: predict on new data. Specifiy the data file with `-i`
