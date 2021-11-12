features=vitbase_r2rfte2e
ft_dim=768
feedback=sample
ngpus=1

flag="--dataset r4r

      --seed 0
      --ngpus ${ngpus}

      --no_lang_ca
      --ob_type pano
      --hist_enc_pano
      --hist_pano_num_layers 2

      --features ${features}
      --feedback ${feedback}

      --maxInput 100
      --maxAction 30
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --lr 1e-5
      --iters 300000
      --log_every 1000
      --optim adamW

      --mlWeight 0.2
      
      --batchSize 4
      --featdropout 0.4
      --dropout 0.5"

# train
CUDA_VISIBLE_DEVICES='5' python r2r/main.py $flag \
      --output_dir ../datasets/R2R/exprs_r4r/ \
      --bert_ckpt_file ../datasets/R2R/trained_models/vitbase-6tasks-pretrain-e2e/model_step_22000.pt \
      --eval_first