
# features=vitbase
features=vitbase_r2rfte2e
ft_dim=768

ngpus=1
seed=0

outdir=../datasets/REVERIE/trained_models/

flag="--root_dir ../datasets
      --output_dir ${outdir}

      --dataset reverie
      --multi_endpoints

      --ob_type pano
      
      --world_size ${ngpus}
      --seed ${seed}
      
      --num_l_layers 9
      --num_x_layers 4
      
      --hist_enc_pano
      --hist_pano_num_layers 2
      --no_lang_ca
      
      --features ${features}
      --feedback sample

      --max_action_len 15
      --max_instr_len 60

      --image_feat_size ${ft_dim}
      --obj_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 200000
      --log_every 1000
      --batch_size 8
      --optim adamW

      --ml_weight 0.2    

      --feat_dropout 0.4
      --dropout 0.5"

# inference
CUDA_VISIBLE_DEVICES='0' python reverie/main_navref.py $flag \
      --resume_file ../datasets/REVERIE/trained_models/best_val_unseen \
      --test --submit

# train
CUDA_VISIBLE_DEVICES='0' python reverie/main_navref.py $flag \
      --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt \
      --eval_first