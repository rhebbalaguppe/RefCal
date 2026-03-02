import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training for Knowledge Distillation')

    parser.add_argument('--current_time', default='', type=str)
    # Datasets
    parser.add_argument('--dataset', default='cifar10', type=str)
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--seed', default=42, type=int,help='seed to use')

    parser.add_argument('--T', default=20, type=float, help='temperature to use for KD')
    parser.add_argument('--T_t', default=20, type=float, help='teacher temperature to use for KD')
    parser.add_argument('--T_s', default=20, type=float, help='student temperature to use for KD')
    parser.add_argument('--temp', default=1, type=float, help='temperature to smooth a model')
    #parser.add_argument('--ls_alpha', default=0.1, type=float, help='LS smoothing factor')
    parser.add_argument('--Lambda', default=0.9, type=float, help='distilling weight to use for KD')
    

    # Optimization options
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    
    parser.add_argument('--exp_name', default="", type=str,
                        help='Experiment name')

    parser.add_argument('--teacher', default="", type=str, help='Experiment name')
    parser.add_argument('--teacher_path', default="", type=str, help='Experiment name')

    parser.add_argument('--train-batch-size', default=128, type=int, metavar='N',
                        help='train batchsize')
    parser.add_argument('--test-batch-size', default=100, type=int, metavar='N',
                        help='test batchsize')

    parser.add_argument('--lr', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')

    parser.add_argument('--alpha', default=0.1, type=float, help='alpha to train Label Smoothing with')
    parser.add_argument('--beta', default=1.0, type=float, help='beta to train MDCA with')
    parser.add_argument('--gamma', default=1, type=float, help='gamma to train Focal Loss with')

    parser.add_argument('--scheduler', default="multistep", type=str, help='scheduler to use for training')
    parser.add_argument('--schedule-steps', type=int, nargs='+', default=[],
                            help='Decrease learning rate at these epochs.')
    parser.add_argument('--lr-decay-factor', type=float, default=0.1, help='LR is multiplied by this on schedule.')
    parser.add_argument('--warmup', default=0, type=int, 
        help='warmup to use for training with scheduler. Should be less than 0.1 of total training time'
    )

    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)')
    # Checkpoints
    parser.add_argument('--checkpoint', default='checkpoint', type=str, help='path to save checkpoint (default: checkpoint)')

    parser.add_argument('--loss', default='cross_entropy', type=str)
    parser.add_argument('--model', default='resnet20', type=str)
    parser.add_argument('--optimizer', default='sgd', type=str)

    parser.add_argument('--prefix', default='', type=str, metavar='PRNAME')
    parser.add_argument('--regularizer', default='l2', type=str, metavar='RNAME')
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--chain_length', default=None, type=int)
    parser.add_argument('--custom_temp', default=None, type=float)
    parser.add_argument('--teacher_temp', default=1.0, type=float)
    
    parser.add_argument('--use_parallel', action='store_true')

    parser.add_argument('--pretrained_model', default=None, type=str)
    parser.add_argument('--pretrained_teacher_model', default=None, type=str)
    parser.add_argument('--pretrained_student_model', default=None, type=str)
    parser.add_argument('--model_path', default=None, type=str, help='Path Bert model to be loaded.')
    parser.add_argument('--model_is_TS_scaled', action="store_true", help='Model is TS scaled')

    # Adafocal
    num_bins = 15
    adafocal_lambda = 1.0
    adafocal_gamma_initial = 1.0
    adafocal_gamma_max = 20.0
    adafocal_gamma_min = -2.0
    adafocal_switch_pt = 0.2
    update_gamma_every = -1

    parser.add_argument("--num-bins", type=int, default=num_bins, dest="num_bins", help="Number of calibration bins")
    parser.add_argument("--adafocal-lambda", type=float, default=adafocal_lambda, dest="adafocal_lambda", help="lambda for adafocal.")
    parser.add_argument("--adafocal-gamma-initial", type=float, default=adafocal_gamma_initial, dest="adafocal_gamma_initial", help="Initial gamma for each bin.")
    parser.add_argument("--adafocal-gamma-max", type=float, default=adafocal_gamma_max, dest="adafocal_gamma_max", help="Maximum cutoff value for gamma.")
    parser.add_argument("--adafocal-gamma-min", type=float, default=adafocal_gamma_min, dest="adafocal_gamma_min", help="Minimum cutoff value for gamma.")
    parser.add_argument("--adafocal-switch-pt", type=float, default=adafocal_switch_pt, dest="adafocal_switch_pt", help="Gamma at which to switch to inverse-focal loss.")
    parser.add_argument("--update-gamma-every", type=int, default=update_gamma_every, dest="update_gamma_every", help="Update gamma every nth batch. If -1, update after epoch end.")
   
    # MIXUP
    parser.add_argument('--mixup', action='store_true')
    parser.add_argument('--mixup_alpha', default=None, type=float)

    # Teacher is temperature scaled
    parser.add_argument('--teacher_is_TS_scaled', action='store_true')

    # MbLS
    parser.add_argument('--margin', default=10.0, type=float)

    # ACLS
    #parser.add_argument('--margin', default=10.0, type=float)
    parser.add_argument('--lambda_1', default=0.1, type=float)
    parser.add_argument('--lambda_2', default=0.01, type=float)
    

    return parser.parse_args()
