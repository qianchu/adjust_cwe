

cuda=$1
model=$2
layers=$3

# evaluation dataset
for f in "usim/usim_en.txt" "scws/scws.txt" "WiC_dataset/train/train.data.txt" "WiC_dataset/dev/dev.data.txt" "WiC_dataset/test/test.data.txt";
do
    echo "processing ${f}"
    python3 extract_features.py --input_file eval_data/${f}  --layers ${layers} --model ${models}  --gpu ${cuda} --batch_size 100;
    python hdf5_to_json.py eval_data/${f}.${model}.ly_${layers}
done

for f in "train_data/en_200k_shuffled.whitespace.txt";
do
    echo "processing ${f}"
    python3 extract_features.py --input_file ${f}  --layers ${layers} --model ${models}  --gpu ${cuda} --batch_size 100;
    python hdf5_to_json.py ${f}.${model}.ly_${layers}
done