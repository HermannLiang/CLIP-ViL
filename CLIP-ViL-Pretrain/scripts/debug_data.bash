# The name of this experiment.
name=$2

# Save logs and models under snap/vqa; make backup.
output=$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

export PYTHONPATH=$PYTHONPATH:/local/harold/ubert/clip_vlp/CLIP

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    unbuffer python -m torch.distributed.launch --master_port=$3 --nproc_per_node=$4 src/tasks/vqa.py \
    --debug_data \
    --use_clip_features \
    --pin_memory False \
    --numWorkers 4 \
    --tqdm --output $output \
    --use_clip \
    --optim bert --lr 1e-5 \
    --llayers 12 --xlayers 0 --rlayers 0 \
    --visualbert_style \
    --vqa_style_transform \
    --clip_model_name RN50 \
    --fp16 \
    --add_zero_padding \
    ${@:5}  | tee $output/log.log