#!/bin/bash

# Assign inputs
inputs[1]=$1  #translations
inputs[2]=$2  #references

# Assign locations
EVAL_DIR=`dirname $0`
TEMP_DIR=$EVAL_DIR/../temp
temp_file=$TEMP_DIR/temp.txt
temp_file2=$TEMP_DIR/temp2.txt

for i in $(seq 1 2)
    do
        # De-segment
        outputs[$i]="$(cat ${inputs[$i]} | \
        sed -r 's/ \@(\S*?)\@ /\1/g' | \
        sed -r 's/\@\@ //g' | \
        sed 's/&lt;s&gt;//' | \
        # De-truecase/ tokenize
        $EVAL_DIR/detruecase.perl | \
        $EVAL_DIR/detokenizer.perl -q)"
    done

echo "${outputs[1]}" > $temp_file
echo "${outputs[2]}" > $temp_file2

# Calculate BLEU
$EVAL_DIR/multi-bleu-detok.perl <(echo "${outputs[2]}") < <(echo "${outputs[1]}")




