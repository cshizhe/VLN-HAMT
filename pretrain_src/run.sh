ps -ef | grep cmt3cat | grep -v grep | cut -c 9-15 | xargs kill


cd /sequoia/data1/shichen/codes/VLN-Transformer/pretrain_thor_src


NODE_RANK=0
NUM_GPUS=4
CUDA_VISIBLE_DEVICES='0,1,2,7' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    pretrain_src/main_r2r.py --world_size ${NUM_GPUS} \
    --model_config pretrain_src/config/r2r_model_config.json \
    --config pretrain_src/config/pretrain_r2r.json \
    --output_dir datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks


NODE_RANK=0
NUM_GPUS=4
CUDA_VISIBLE_DEVICES='0,1,2,7' python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK \
    pretrain_src/main_r2r.py --world_size ${NUM_GPUS} \
    --model_config pretrain_src/config/rxr_xlm_model_config.json \
    --config pretrain_src/config/pretrain_rxr.json \
    --output_dir datasets/R2R/exprs/pretrain/cmt-vitbase-6tasks

NODE_RANK=0
NUM_GPUS=4
CUDA_VISIBLE_DEVICES='0,1,2,7' python -m torch.distributed.launch 
--nproc_per_node=${NUM_GPUS} --node_rank $NODE_RANK main_r2r_image.py \
    --model_config pretrain_src/config/r2r_model_config.json \
    --config pretrain_src/config/pretrain-6tasks.json \
    --output_dir datasets/R2R/exprs_pretrain_lxmert/thor-r2r-6tasks-vitbase.e2e