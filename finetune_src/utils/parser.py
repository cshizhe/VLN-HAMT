import argparse
import os
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='/sequoia/data1/shichen/datasets')
    parser.add_argument(
        '--dataset', type=str, default='r2r', 
        choices=['r2r', 'r4r', 'r2r_back', 'r2r_last', 'rxr', 'touchdown']
    )
    parser.add_argument('--output_dir', type=str, default='default', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")
    
    # General
    parser.add_argument('--iters', type=int, default=300000, help='training iterations')
    parser.add_argument('--log_every', type=int, default=2000)
    parser.add_argument('--eval_first', action='store_true', default=False)
        
    parser.add_argument('--vlnbert', type=str, choices=['cmt', 'causal.cmt', 'mmt', 'cmt3cat', 'cmt3stack'])
    parser.add_argument('--ob_type', type=str, choices=['cand', 'pano'])
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--eval_teacher_forcing', action='store_true', default=False)

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=80)
    parser.add_argument('--max_action_len', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')
    
    # Load the model from
    parser.add_argument("--resume_file", default=None, help='path of the trained model')
    parser.add_argument("--resume_optimizer", action="store_true", default=False)

    # Augmented Paths from
    parser.add_argument("--aug", default=None)
    parser.add_argument('--bert_ckpt_file', default=None, help='init vlnbert')

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)
    parser.add_argument("--teacher_weight", type=float, default=1.)

    parser.add_argument("--features", type=str, default='places365')
    parser.add_argument('--fix_lang_embedding', action='store_true', default=False)
    parser.add_argument('--fix_hist_embedding', action='store_true', default=False)
    parser.add_argument('--fix_obs_embedding', action='store_true', default=False)

    parser.add_argument('--num_l_layers', type=int, default=9)
    parser.add_argument('--num_h_layers', type=int, default=0)
    parser.add_argument('--num_x_layers', type=int, default=4)
    parser.add_argument('--hist_enc_pano', action='store_true', default=False)
    parser.add_argument('--hist_pano_num_layers', type=int, default=2)
    # mmt
    parser.add_argument('--prefix_causal_attn', action='store_true', default=False)
    # cmt
    parser.add_argument('--no_lang_ca', action='store_true', default=False)
    parser.add_argument('--act_pred_token', default='ob_txt', choices=['ob', 'ob_txt', 'ob_hist', 'ob_txt_hist'])

    # Dropout Param
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--feat_dropout', type=float, default=0.3)

    # Submision configuration
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--no_cand_backtrack', action='store_true', default=False)

    # Training Configurations
    parser.add_argument(
        '--optim', type=str, default='rms',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )    # rms, adam
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='sample',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument(
        '--teacher', type=str, default='final',
        help="How to get supervision. one of ``next`` and ``final`` "
    )
    parser.add_argument('--epsilon', type=float, default=0.1, help='')

    # Model hyper params:
    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--image_feat_size', type=int, default=2048)
    parser.add_argument('--views', type=int, default=36)

    # A2C
    parser.add_argument("--gamma", default=0.9, type=float, help='reward discount factor')
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total", 
        type=str, help='batch or total'
    )

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    # Setup input paths
    if args.dataset == 'touchdown': 
        ft_file_map = {
            'resnet18': 'resnet18_view_fts.hdf5',
            'vit_clip': 'vit_clip_view_fts.hdf5'
        }
        args.img_ft_file = os.path.join(ROOTDIR, 'Touchdown', 'features', ft_file_map[args.features])
        args.anno_dir = os.path.join(ROOTDIR, 'Touchdown', 'annotations', 'touchdown', 'data')
        args.graph_dir = os.path.join(ROOTDIR, 'Touchdown', 'annotations', 'touchdown', 'graph')

        #args.output_dir = os.path.join(ROOTDIR, 'Touchdown', 'exprs_%s'%args.dataset, 'seqvlnbert', args.name)
        
    else:   # Use Matterport3D environment
        ft_file_map = {
            'imagenet': 'pth_resnet152_imagenet.hdf5',
            'imagenet_caffe': 'caffe_resnet152_imagenet.hdf5',
            'places365': 'caffe_resnet152_places365.hdf5',
            'vitbase': 'pth_vit_base_patch16_224_imagenet.hdf5',
            'vitbase_r2rfte2e': 'pth_vit_base_patch16_224_imagenet_r2r.e2e.ft.22k.hdf5',
            'vitbase_r2rfte2e.aug': 'pth_vit_base_patch16_224_imagenet_r2r.e2e.dataaug.20k.hdf5',
            'vitbase_in21k': 'pth_vit_base_patch16_224_in21k_imagenet.hdf5',
            'vitbase_clip': 'pth_vit_base_patch32_224_clip.hdf5',
            'vitbase_r2r.e2e.singlestage': 'pth_vit_base_patch16_224_imagenet_r2r.e2e.pt.25k.hdf5',
            'vitbase_r2r.e2e.noobs': 'pth_vit_base_patch16_224_imagenet_r2r.e2e.noobs.20k.hdf5',
            'vitbase_r2r.e2e.nosprel': 'pth_vit_base_patch16_224_imagenet_r2r.e2e.nosprel.19k.hdf5',
        }
        args.img_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', ft_file_map[args.features])
    
        args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
        args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

        if args.dataset == 'rxr':
            args.anno_dir = os.path.join(ROOTDIR, 'RxR', 'annotations')
            #args.output_dir = os.path.join(ROOTDIR, 'RxR', 'exprs_%s'%args.dataset, 'seqvlnbert', args.name)
        else:
            args.anno_dir = os.path.join(ROOTDIR, 'R2R', 'annotations')
            #args.output_dir = os.path.join(ROOTDIR, 'R2R', 'exprs_%s'%args.dataset, 'seqvlnbert', args.name)

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    # remove unnecessary args
    if args.vlnbert == 'cmt':
        del args.prefix_causal_attn
    elif args.vlnbert == 'mmt':
        del args.no_lang_ca
        del args.act_pred_token
    elif args.vlnbert == 'causal.cmt':
        del args.prefix_causal_attn
        del args.no_lang_ca
        del args.act_pred_token

    return args

