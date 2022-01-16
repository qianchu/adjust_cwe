

cuda=$1
model=$2
layers=$3
ft_model=$4
dim=$5

# evaluation dataset
for f in "usim/usim_en.txt" "context_simlex/evaluation_kit_final/data/data_en.tsv.out" "scws/scws.txt" "WiC_dataset/train/train.data.txt" "WiC_dataset/dev/dev.data.txt" "WiC_dataset/test/test.data.txt";
do
    echo "processing ${f}"
    python3 extract_features.py --input_file eval_data/${f}  --layers ${layers} --model ${model}  --gpu ${cuda} --batch_size 100;
    python hdf5_to_json.py eval_data/${f}__${model}.ly_${layers}__.hdf5
done

for f in "train_data/en_200k_shuffled.whitespace.txt";
do
    echo "processing ${f}"
    python3 extract_features.py --input_file ${f}  --layers ${layers} --model ${model}  --gpu ${cuda} --batch_size 100;
    python hdf5_to_json.py ${f}__${model}.ly_${layers}__.hdf5
    python fasttext_embed.py --model ${ft_model} --text ${f} --dim ${dim}
done