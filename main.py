import shutil
import time
from datetime import datetime

import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboard_logger import configure, log_value
import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from dataset import CatDogDataset
from utils import split

from torch.utils.data import DataLoader

best_acc1 = 0


def main():
    global best_acc1
    log_dir = './output'
    log_dir = os.path.join(log_dir, datetime.today().strftime('%Y-%m-%d-%H:%M:%S'))
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    # init log
    configure(log_dir)

    device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')
    epochs = 10
    train_dir = './input/train'
    val_dir = './input/val'
    test_dir = './input/test1'

    # 这个代码只用执行一次 执行一次后注释掉
    split.split_train_test(train_dir, val_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(128),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = CatDogDataset(dir=train_dir, is_test=False, transform=train_transform)
    val_dataset = CatDogDataset(dir=val_dir, is_test=False, transform=val_transform)
    # test_dataset = CatDogDataset(dir=test_dir, is_test=True, transforms=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 500),
        nn.Linear(500, 2)
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, amsgrad=True)

    for epoch in range(epochs):
        train(train_loader, model, criterion, optimizer, epoch, device)
        val_acc = validate(val_loader, model, criterion, epoch, device)

        # remember best prec@1 and save checkpoint
        is_best = val_acc > best_acc1
        best_acc1 = max(val_acc, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
        }, is_best, log_dir)

        print('epoch: {}, val_acc: {}'.format(epoch, val_acc))


def train(train_loader, model, criterion, optimizer, epoch, device, print_freq=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1,))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if print_freq:
            if i % print_freq == 0:
                progress.display(i)

    # log
    log_value('train/acc1', top1.avg, epoch)
    log_value('train/loss', losses.avg, epoch)
    log_value('train/data_time', data_time.avg, epoch)
    log_value('train/batch_time', batch_time.avg, epoch)


def validate(val_loader, model, criterion, epoch, device, print_freq=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, losses, top1],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if print_freq:
                if i % print_freq == 0:
                    progress.display(i)

    # log
    log_value('val/acc', top1.avg, epoch)
    log_value('val/loss', losses.avg, epoch)
    log_value('val/data_time', data_time.avg, epoch)
    log_value('val/batch_time', batch_time.avg, epoch)

    return top1.avg


def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth.tar'):
    file_path = os.path.join(save_dir, filename)
    torch.save(state, file_path)
    if is_best:
        best_file_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(file_path, best_file_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    """
    python train_cifar.py --arch resnet20 --dataset cifar10 --dataset_dir /root/Workspace/Dataset --log_dir "./runs/cifar10/resnet20"
    """
    main()
