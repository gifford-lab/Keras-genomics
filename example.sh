#!/bin/bash

for dtype in 'train' 'valid' 'test'; do
    echo $dtype
	paste - - -d' ' < example/$dtype\.fa > tmp.tsv
	python embedH5.py tmp.tsv example/$dtype\.target expt1/$dtype\.h5
done