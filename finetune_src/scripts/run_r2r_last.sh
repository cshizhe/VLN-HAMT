features=vitbase_r2rfte2e
ft_dim=768
feedback=sample
ngpus=1


flag="--dataset r2r_last
      --ngpus ${ngpus}
      
      --hist_enc_pano
      --hist_pano_num_layers 2

      --fix_lang_embedding
      --fix_hist_embedding

      --features ${features}
      --feedback ${feedback}

      --maxAction 15
      --batchSize 8
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

CUDA_VISIBLE_DEVICES='3' python r2r/main.py $flag \
      --output_dir ../datasets/R2R/exprs_r2rlast/ \
      --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt \
      --eval_first 
