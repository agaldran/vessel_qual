import sys, os, argparse
import os.path as osp
import operator
from tqdm import trange
import numpy as np
import torch
from models.res_unet_adrian import UNet as unet
from utils.get_loaders import get_seg_loaders, get_seg_test_dataset
from utils.evaluation import evaluate, ewma
from utils.reproducibility import set_seeds
from skimage.transform import resize
import warnings
from tqdm import tqdm
from skimage.io import imsave
from skimage.util import img_as_ubyte

# argument parsing
parser = argparse.ArgumentParser()
# as seen here: https://stackoverflow.com/a/15460288/3208255
# parser.add_argument('--layers',  nargs='+', type=int, help='unet configuration (depth/filters)')
# annoyingly, this does not get on well with guild.ai, so we need to reverse to this one:
parser.add_argument('--layers', type=str, default='8/16', help='unet configuration (filters x layer)')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='batch Size')
parser.add_argument('--n_epochs', type=int, default=60, help='total max epochs (1000)')
parser.add_argument('--patience', type=int, default=20, help='epochs until early stopping (50)')
parser.add_argument('--decay_f', type=float, default=0.1, help='decay factor after 3/4 of patience epochs (0=no decay)')
parser.add_argument('--csv_train', type=str, default='train.csv', help='path to training data csv')
parser.add_argument('--data_path', type=str, default='data/DRIVE/', help='where the training data is')
parser.add_argument('--data_out', type=str, default='data/DRIVE/', help='where the new examples will be outputed')
parser.add_argument('--n_epochs_gen', type=int, default=20, help='epochs when we generate examples')

def compare_op(metric):
    '''
    This should return an operator that given a, b returns True if a is better than b
    Also, let us return which is an appropriately terrible initial value for such metric
    '''
    if metric == 'auc':
        return operator.gt, 0
    elif metric == 'dice':
        return operator.gt, 0
    elif metric == 'loss':
        return operator.lt, np.inf
    else:
        raise NotImplementedError

def reduce_lr(optimizer, epoch, factor=0.1, verbose=True):
    for i, param_group in enumerate(optimizer.param_groups):
        old_lr = float(param_group['lr'])
        new_lr = old_lr * factor
        param_group['lr'] = new_lr
        if verbose:
            print('Epoch {:5d}: reducing learning rate'
                  ' of group {} to {:.4e}.'.format(epoch, i, new_lr))

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def create_pred(model, tens, mask, coords_crop, original_sz):
    act = torch.sigmoid
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        logits = model(tens.unsqueeze(dim=0).to(device)).squeeze(dim=0)
    pred = act(logits)
    pred = pred.detach().cpu().numpy()[-1]  # this takes last channel in multi-class, ok for 2-class
    # Orders: 0: NN, 1: Bilinear(default), 2: Biquadratic, 3: Bicubic, 4: Biquartic, 5: Biquintic
    pred = resize(pred, output_shape=original_sz, order=1)
    full_pred = np.zeros_like(mask, dtype=float)
    full_pred[coords_crop[0]:coords_crop[2], coords_crop[1]:coords_crop[3]] = pred
    full_pred[~mask.astype(bool)] = 0
    return full_pred

def save_pred(full_pred, save_results_path, im_name):
    os.makedirs(save_results_path, exist_ok=True)
    im_name = im_name.rsplit('/', 1)[-1]
    save_name = osp.join(save_results_path, im_name[:-4] + '.gif')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        imsave(save_name, img_as_ubyte(full_pred))

def build_examples(loader, model, epoch, data_out):
    csv_test = loader.dataset.csv_path
    test_dataset = get_seg_test_dataset(csv_path=csv_test, tg_size=(512, 512))
    model.eval()
    save_results_path = osp.join(data_out, 'predicted_epoch_{}'.format(epoch))
    for i in tqdm(range(len(test_dataset))):
        im_tens, mask, coords_crop, original_sz, im_name = test_dataset[i]
        full_pred = create_pred(model, im_tens, mask, coords_crop, original_sz)
        save_pred(full_pred, save_results_path, im_name.replace('training','manual1'))

def run_one_epoch(loader, model, criterion, optimizer=None):
    device='cuda' if next(model.parameters()).is_cuda else 'cpu'
    train = optimizer is not None  # if we are in training mode there will be an optimizer and train=True here

    if train:
        model.train()
    else:
        model.eval()
    logits_all, labels_all = [], []
    with trange(len(loader)) as t:
        n_elems, running_loss = 0, 0
        for i_batch, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(device), labels.to(device)
            logits = model(inputs)
            if model.n_classes == 1:
                loss = criterion(logits, labels.unsqueeze(dim=1).float())  # BCEWithLogitsLoss()/DiceLoss()
            else:
                loss = criterion(logits, labels)  # CrossEntropyLoss()

            if train:  # only in training mode
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logits_all.extend(logits)
            labels_all.extend(labels)

            # Compute running loss
            running_loss += loss.item() * inputs.size(0)
            n_elems += inputs.size(0)
            run_loss = running_loss / n_elems
            if train: t.set_postfix(tr_loss="{:.4f}".format(float(run_loss)))
            else: t.set_postfix(vl_loss="{:.4f}".format(float(run_loss)))
            t.update()
    return logits_all, labels_all, run_loss


def train(model, optimizer, criterion, train_loader, val_loader, n_epochs, patience, decay_f, data_out, n_epochs_gen):
    counter_since_checkpoint = 0
    tr_losses, tr_aucs, tr_dices, vl_losses, vl_aucs, vl_dices = [], [], [], [], [], []
    is_better, best_monitoring_metric = compare_op('auc')

    for epoch in range(n_epochs):
        print('\n EPOCH: {:d}/{:d}'.format(epoch+1, n_epochs))
        # train one epoch
        train_logits, train_labels, train_loss = run_one_epoch(train_loader, model, criterion, optimizer)
        train_auc, train_dice = evaluate(train_logits, train_labels, model.n_classes)

        # validate one epoch, note no optimizer is passed
        with torch.no_grad():
            val_logits, val_labels, val_loss = run_one_epoch(val_loader, model, criterion)
            val_auc, val_dice = evaluate(val_logits, val_labels, model.n_classes)
        print('Train/Val Loss: {:.4f}/{:.4f}  -- Train/Val AUC: {:.4f}/{:.4f}  -- Train/Val DICE: {:.4f}/{:.4f} -- LR={:.6f}'.format(
                train_loss, val_loss, train_auc, val_auc, train_dice, val_dice, get_lr(optimizer)).rstrip('0'))

        # store performance for this epoch
        tr_aucs.append(train_auc)
        vl_aucs.append(val_auc)
        #  smooth val values with a moving average before comparing
        val_auc = ewma(vl_aucs)[-1]

        # check if performance was better than anyone before and checkpoint if so
        monitoring_metric = val_auc

        if is_better(monitoring_metric, best_monitoring_metric):
            print('Best (smoothed) val {} attained. {:.4f} --> {:.4f}'.format(
                'auc', best_monitoring_metric, monitoring_metric))
            best_monitoring_metric = monitoring_metric
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
            return
        # create examples at this point
        if epoch != 0 and (epoch+1) % n_epochs_gen == 0:
            print('\n Generating examples for model trained for {:d} epochs'.format(epoch+1))
            build_examples(train_loader, model, epoch+1, data_out)
            build_examples(val_loader, model, epoch+1, data_out)
    del model
    torch.cuda.empty_cache()
    return
if __name__ == '__main__':
    '''
    Example:
    python
    '''

    # reproducibility
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    seed_value = 0
    set_seeds(seed_value, use_cuda)

    # gather parser parameters
    args = parser.parse_args()
    layers = args.layers.split('/')
    layers = list(map(int, layers))
    lr, bs = args.lr, args.batch_size
    n_epochs, patience, decay_f = args.n_epochs, args.patience, args.decay_f
    data_path = args.data_path
    csv_train = args.csv_train
    data_out = args.data_out
    n_epochs_gen = args.n_epochs_gen
    csv_train = osp.join(data_path, csv_train)
    csv_val = csv_train.replace('train', 'val')

    print('* Creating Dataloaders, batch size = {:d}'.format(bs))
    train_loader, val_loader = get_seg_loaders(csv_path_train=csv_train, csv_path_val=csv_val, batch_size=bs)

    print('* Instantiating a Unet model with config = '+str(layers))
    model = unet(in_c=3, n_classes=1, layers=layers).to(device)
    print("Total params: {0:,}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    print('* Instantiating loss function', str(criterion))
    print('* Starting to train\n','-' * 10)
    train(model, optimizer, criterion, train_loader, val_loader, n_epochs, patience, decay_f, data_out, n_epochs_gen)


