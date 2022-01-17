cuda=$1
train_data=$2
tgt_data=$3
src_data=$4


python3 main.py \
--gpu $cuda \
--data $train_data \
--tgt_data $tgt_data \
--src_data $src_data \
--norm normalize,center \
--type \
--save \
--eval \

