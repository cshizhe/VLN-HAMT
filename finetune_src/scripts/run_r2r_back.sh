features=vitbase_r2rfte2e
ft_dim=768
feedback=sample
ngpus=1
seed=0


flag="--dataset r2r_back
      --seed ${seed}

      --ngpus ${ngpus}

      --fix_lang_embedding
      --fix_hist_embedding

      --hist_enc_pano
      --hist_pano_num_layers 2
      
      --features ${features}
      --feedback ${feedback}

      --maxAction 30
      --batchSize 4
      --image_feat_size ${ft_dim}

      --lr 1e-5
      --iters 300000
      --log_every 1000
      --optim adamW

      --mlWeight 0.2
      --maxInput 60
      --angle_feat_size 4
      --featdropout 0.4
      --dropout 0.5"

CUDA_VISIBLE_DEVICES='6' python r2r/main.py $flag \
      --output_dir ../datasets/R2R/exprs_r2rback/ \
      --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt \
      --eval_first 
