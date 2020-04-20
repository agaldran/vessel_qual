import argparse
from datetime import datetime
import json
import operator
from utils.lookahead import Lookahead
from models.get_reg_model import get_arch
from utils.get_loaders import get_reg_loaders
from utils.evaluation import ewma
from utils.reproducibility import set_seeds
from utils.model_saving_loading import write_model
from tqdm import trange
import numpy as np
import torch

import os.path as osp
import os
import sys


def str2bool(v):
    # as seen here: https://stackoverflow.com/a/43357954/3208255
    if isinstance(v, bool):
       return v
    if v.lower() in ('true','yes'):
        return True
    elif v.lower() in ('false','no'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')

def compare_op(metric):
    '''
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    '''
    if metric == 'auc':
        return operator.gt, 0.5
    elif metric == 'kappa':
        return operator.gt, 0
    elif metric == 'kappa_auc_avg':
        return operator.gt, 0.25
    elif metric == 'loss':
        return operator.lt, np.inf
    elif metric == 'bal_acc':
        return operator.gt, 0
    elif metric == 'err':
        return operator.lt, 1000
    else:
        raise NotImplementedError

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def reduce_lr(optimizer, epoch, factor=0.1, verbose=True):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = old_lr * factor
        param_group['lr'] = new_lr
        if verbose:
            print('Epoch {:5d}: reducing learning rate'
                  ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

parser = argparse.ArgumentParser()
parser.add_argument('--csv_train', type=str, default='DRIVE/train.csv', help='path to training data csv')
parser.add_argument('--model_name', type=str, default='resnet50', help='selected architecture')
parser.add_argument('--pretrained', type=str2bool, nargs='?', const=True, default=False, help='from pretrained weights')
parser.add_argument('--loss_fn', type=str, default='mae', help='loss function (mse/mae)')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--optimizer', type=str, default='adam', help='sgd/adam')
parser.add_argument('--n_epochs', type=int, default=1000, help='total max epochs (1000)')
parser.add_argument('--patience', type=int, default=100, help='epochs until early stopping (50)')
parser.add_argument('--decay_f', type=float, default=0.1, help='decay factor after 3/4 of patience epochs (0=no decay)')
parser.add_argument('--metric', type=str, default='err', help='which metric to monitor (err/loss)')
parser.add_argument('--save_model', type=str2bool, nargs='?', const=True, default=False, help='avoid saving anything')
parser.add_argument('--save_path', type=str, default='date_time', help='path to save model (defaults to date/time')

args = parser.parse_args()

def run_one_epoch_reg(loader, model, criterion, optimizer=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train = optimizer is not None
    model.train() if train else model.eval()
    preds_all, labels_all = [], []
    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0
        for i_batch, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device, non_blocking=True), labels.float().to(device, non_blocking=True)
            logits = model(inputs)
            preds = torch.sigmoid(logits)
            # pp = preds.squeeze().detach().cpu().numpy()
            # ll = labels.cpu().numpy()
            # print(list(zip(pp,ll)))
            # import time
            # time.sleep(0.5)
            loss = criterion(preds.squeeze(), labels)

            if train:  # only in training mode
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            ll = loss.item()
            del loss
            preds_all.extend(list(preds.squeeze().cpu().detach().numpy()))# , axis=1
            labels_all.extend(list(labels.squeeze().cpu().numpy())) # , axis=1

            # Compute running loss
            running_loss += ll * inputs.size(0)
            n_elems += inputs.size(0)
            run_loss = running_loss / n_elems
            if train: t.set_postfix(tr_loss="{:.4f}".format(float(run_loss)))
            else: t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()
    return np.array(preds_all, dtype=float), np.array(labels_all, dtype=float), run_loss

def train_reg(model, optimizer, train_criterion, val_criterion, train_loader, val_loader,
          n_epochs, metric, patience, decay_f, exp_path):
    counter_since_checkpoint = 0
    tr_losses, tr_errs, vl_losses, vl_errs = [], [], [], [] # 5*[10], 5*[10], 5*[10], 5*[10]
    stats = {}
    is_better, best_monitoring_metric = compare_op(metric)
    best_err = 0
    for epoch in range(n_epochs):
        print('\n EPOCH: {:d}/{:d}'.format(epoch+1, n_epochs))
        tr_preds, tr_labels, tr_loss = run_one_epoch_reg(train_loader, model, train_criterion, optimizer)
        # validate one epoch, note no optimizer is passed
        with torch.no_grad():
            vl_preds, vl_labels, vl_loss = run_one_epoch_reg(val_loader, model, val_criterion)
        tr_err = train_criterion(torch.from_numpy(tr_preds), torch.from_numpy(tr_labels)).item()
        print('\n')
        vl_err = train_criterion(torch.from_numpy(vl_preds), torch.from_numpy(vl_labels)).item()
        print('Train/Val. Loss: {:.4f}/{:.4f} -- ERR: {:.4f}/{:.4f}  -- LR={:.6f}'.format(
                tr_loss, vl_loss, tr_err, vl_err, get_lr(optimizer)).rstrip('0'))
        # print(vl_preds.shape)
        print('preds/labels = {:.3f}/{:.3f} -- {:.3f}/{:.3f} -- {:.3f}/{:.3f}, -- '
              '{:.3f}/{:.3f} '.format(vl_preds[0], vl_labels[0], vl_preds[1], vl_labels[1 ],
                                      vl_preds[2], vl_labels[2], vl_preds[3], vl_labels[3]))
        import time
        time.sleep(2)

        # store performance for this epoch
        tr_losses.append(tr_loss)
        tr_errs.append(tr_err)
        vl_losses.append(vl_loss)
        vl_errs.append(vl_err) # .detach().numpy()


        #  smooth val values with a moving average before comparing
        vl_err = ewma(vl_errs, window=5)[-1]
        vl_loss = ewma(vl_losses, window=5)[-1]

        # check if performance was better than anyone before and checkpoint if so
        if metric =='loss': monitoring_metric = vl_loss
        elif metric == 'err': monitoring_metric = vl_err
        else: sys.exit('Not a suitable metric for this task')

        if is_better(monitoring_metric, best_monitoring_metric):
             print('Best (smoothed) val {} attained. {:.4f} --> {:.4f}'.format(
                 metric, best_monitoring_metric, monitoring_metric))
             best_err = vl_err
             if exp_path != None:
                 print(15*'-',' Checkpointing ', 15*'-')
                 write_model(exp_path, model, optimizer, stats)

             best_monitoring_metric = monitoring_metric
             stats['tr_losses'], stats['vl_losses'] = tr_losses, vl_losses
             stats['tr_errs'], stats['vl_errs'] = tr_errs, vl_errs
             counter_since_checkpoint = 0  # reset patience
        else:
            counter_since_checkpoint += 1

        if decay_f != 0 and counter_since_checkpoint == 3*patience//4:
            reduce_lr(optimizer, epoch, factor=decay_f, verbose=False)
            print(8 * '-', ' Reducing LR now ', 8 * '-')

        # early stopping if no improvement happened for `patience` epochs
        if counter_since_checkpoint == patience:
            print('\n Early stopping the training, trained for {:d} epochs'.format(epoch))
            del model
            torch.cuda.empty_cache()
            return best_err

    del model
    torch.cuda.empty_cache()
    return best_err

if __name__ == '__main__':
    '''
    Example:
    python train.py --load_checkpoint resnext50_eyepacs_gls
    '''
    data_path = 'data'
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # reproducibility
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # gather parser parameters
    args = parser.parse_args()
    model_name = args.model_name
    lr, bs, loss_fn, optimizer_choice = args.lr, args.batch_size, args.loss_fn, args.optimizer
    csv_train = args.csv_train
    csv_train = osp.join(data_path, csv_train)
    csv_val = csv_train.replace('train', 'val')
    n_epochs, patience, decay_f, metric = args.n_epochs, args.patience, args.decay_f, args.metric
    save_model = str2bool(args.save_model)

    if save_model:
        save_path = args.save_path
        if save_path == 'date_time':
            save_path = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        experiment_path = osp.join('experiments', save_path)
        args.experiment_path = experiment_path  # store experiment path
        os.makedirs(experiment_path, exist_ok=True)
        config_file_path = osp.join(experiment_path,'config.cfg')
        with open(config_file_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    else: experiment_path=None

    print('* Instantiating model {}'.format(model_name))
    model = get_arch(model_name, in_channels=1, n_classes=1)


    # class MyModel(nn.Module):
    #     def __init__(self):
    #         super(MyModel, self).__init__()
    #         self.conv1 = nn.Conv2d(3, 3, 3, 1, 1)
    #         self.bn1 = nn.BatchNorm2d(3)
    #
    #     def forward(self, x):
    #         x = self.bn1(self.conv1(x))
    #         return x
    # model = MyModel()
    # print(model)
    #
    # for name, module in model.named_modules():
    #     if isinstance(module, nn.BatchNorm2d):
    #         # Get current bn layer
    #         bn = getattr(model, name)
    #         # Create new gn layer
    #         gn = nn.GroupNorm(1, bn.num_features)
    #         # gn = torch.nn.InstanceNorm2d(bn.num_features)
    #         # Assign gn
    #         print('Swapping {} with {}'.format(bn, gn))
    #         setattr(model, name, gn)



    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    model = model.to(device)

    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    train_loader, val_loader = get_reg_loaders(csv_path_train=csv_train, csv_path_val=csv_val, batch_size=bs,
                                               p_manual=0.5, p_nothing=0.20, max_deg_patches=100,
                                               max_patch_size=(64, 64), sim_method='mutual_info')

    if optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_choice == 'look_ahead':
        base_opt = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = Lookahead(base_opt, k=5, alpha=0.5)  # Initialize Lookahead
    elif optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        sys.exit('not a valid optimizer choice')

    print('* Instantiating base loss function {}'.format(loss_fn))
    if loss_fn == 'mse':
        train_crit, val_crit = torch.nn.MSELoss(reduction='mean'), torch.nn.MSELoss(reduction='mean')
    elif loss_fn == 'mae':
        train_crit, val_crit = torch.nn.L1Loss(reduction='mean'), torch.nn.L1Loss(reduction='mean')


    print('* Starting to train\n','-' * 10)
    err = train_reg(model, optimizer, train_crit, val_crit, train_loader, val_loader,
              n_epochs, metric, patience, decay_f, experiment_path)
    print("ERR: %f" % err)


    if save_model:
        file = open(osp.join(experiment_path, 'val_metrics.txt'), 'w')
        file.write(str(err))
        file.close()
