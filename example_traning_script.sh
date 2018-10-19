
#!/bin/sh

# This script trains the model on a pre-processed (incl. BPE) parallel corpus.

# Check if all required arguments are supplied
if [ $# -ne 1  ]; then
    echo 'Expected one argument. Exiting.'
    echo 'Usage: bash train_tr.sh run_id'
    exit 1
fi

# Source and target languages
src=en
tgt=de

# Directories
home_dir=/home/name
main_dir=$home_dir/exp_dir
data_dir=$main_dir/data_dir
train_dir=$data_dir/train_data
devtest_dir=$data_dir/devtest_data
model_dir=$main_dir/model_dir

venv=$home_dir/tensorflow_venv/bin/activate
nematode_home=$home_dir/nematode

run_id=$1

# Activate python virtual environment
. $venv
# Create run-specific directory
exp_dir=$model_dir/trained_models
st_dir=$exp_dir/$src-$tgt
run_dir=$st_dir/$run_id

if [ ! -d "$exp_dir" ]; then
    mkdir $exp_dir
fi

if [ ! -d "$st_dir" ]; then
    mkdir $st_dir
fi

if [ ! -d "$run_dir" ]; then
    mkdir $run_dir
    echo "Creating $run_dir ... "
else
    echo "$run_dir already exists, either loading or overwriting its contents ... "
fi

echo "Commencing run $run_id ... "
echo "Trained model is saved to $run_dir . "

model_name=nematode_model

python $nematode_home/codebase/nmt.py \
    --save_to $run_dir/$model_name.npz \
    --model_name $model_name \
    --source_dataset $train_dir/train.de-en.bpe.$src \
    --target_dataset $train_dir/train.de-en.bpe.$tgt \
    --valid_source_dataset $devtest_dir/dev.de-en.bpe.$src \
    --valid_target_dataset $devtest_dir/dev.de-en.bpe.$tgt \
    --dictionaries $data_dir/joint_corpus.bpe.ende.json $data_dir/joint_corpus.bpe.ende.json \
    --model_type transformer \
    --embedding_size 512 \
    --num_encoder_layers 6 \
    --num_decoder_layers 6 \
    --ffn_hidden_size 2048 \
    --hidden_size 512 \
    --num_heads 8 \
    --max_len -1 \
    --translation_max_len 200 \
    --token_batch_size 4096 \
    --sentence_batch_size 64 \
    --maxibatch_size 20 \
    --beam_size 4 \
    --disp_freq 100 \
    --valid_freq 4000 \
    --greedy_freq 10000 \
    --beam_freq 10000 \
    --save_freq 10000 \
    --summary_freq 10000 \
    --log_file $run_dir/log.txt \
    --bleu_script $nematus_home/eval/bleu_script.sh \
    --gradient_delay 0

