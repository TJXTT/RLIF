import argparse
import os
import shutil
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.cuda.amp import autocast as autocast
from datasets.dvs128_gesture import DVS128Gesture

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--data-path', default='', type=str, help='data path')
parser.add_argument('--arch', '-a', metavar='ARCH', default='rlif_model',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 100)')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-load', default='', type=str, metavar='PATH',
                    help='path to training mask (default: none)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=2000, type=int, metavar='N',
                    help='number of total epochs to run')

best_prec1 = 0

change_arr = [300, 600, 1200, 1500]
tp1 = [];
tp5 = [];
ep = [];
lRate = [];
device_num = 2


tp1_tr = [];
tp5_tr = [];
losses_tr = [];
losses_eval = [];


def main():
    global args, best_prec1, batch_size, device_num

    args = parser.parse_args()
    batch_size = args.batch_size
    model = CNNModel()
    
    print(model)
    torch.manual_seed(3)
    torch.cuda.manual_seed_all(44)

    if device_num < 2:
        device = 0
        torch.cuda.set_device(device)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()


    criterion = torch.nn.MSELoss(size_average=False).cuda()
    criterion_en = torch.nn.CrossEntropyLoss()

    learning_rate = args.lr

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = 5e-5
    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.557, 0.549, 0.5534])
    transform_train = transforms.Compose([
        transforms.RandomCrop(128, padding=4),
        # transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    
    data_path = args.data_path
    train_data = DVS128Gesture(data_path, train=True, data_type='frame', frames_number=30, split_by='number') #  duration=50)#
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    val_data = DVS128Gesture(data_path, train=False, data_type='frame', frames_number=30, split_by='number')
    val_loader = torch.utils.data.DataLoader(val_data,  # val_data for testing
                                             batch_size=int(args.batch_size), shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=False)


    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)
        ep.append(epoch)

        # train for one epoch
        start_time = time.time()
        train(train_loader, model, criterion, optimizer, epoch, time_steps=30)
        end_time = time.time()
        print('Train cost: {} sec/epoch'.format(end_time-start_time))
        # evaluate on validation set
        start_time = time.time()
        prec1 = validate(val_loader, model, criterion, time_steps=30)
        end_time = time.time()
        print('Eval cost: {} sec/epoch'.format(end_time-start_time))
        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, is_best)

    for k in range(0, args.epochs - args.start_epoch):
        print('Epoch: [{0}/{1}]\t'
              'LR:{2}\t'
              'Prec@1 {top1:.3f} \t'
              'Prec@5 {top5:.3f} \t'
              'En_Loss_Eval {losses_en_eval: .4f} \t'
              'Prec@1_tr {top1_tr:.3f} \t'
              'Prec@5_tr {top5_tr:.3f} \t'
              'Loss_train {losses: .4f}'.format(
            ep[k], args.epochs, lRate[k], top1=tp1[k], top5=tp5[k], losses_en_eval=losses_eval[k], top1_tr=tp1_tr[k],
            top5_tr=tp5_tr[k], losses=losses_tr[k]))


def train(train_loader, model, criterion, optimizer, epoch, time_steps):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    top1_tr = AverageMeter()
    top5_tr = AverageMeter()
    logits = AverageMeter()
    # switch to train mode
    model.train()
    for module in model.modules():
        module.train()
    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target = target.cuda()
        labels = Variable(target.cuda())
        if device_num < 2:
            input_var = Variable(input.type(torch.cuda.FloatTensor))
        else:
            input_var = torch.autograd.Variable(input.type(torch.cuda.FloatTensor))

        optimizer.zero_grad()  # Clear gradients w.r.t. parameters

        output = model(input_var, steps=time_steps, training=True)
        targetN = output.data.clone().zero_().cuda()
        targetN.scatter_(1, target.unsqueeze(1), 1)
        targetN = Variable(targetN.type(torch.cuda.FloatTensor))
        loss = criterion(output, targetN)

        loss.backward(retain_graph=False)
        optimizer.step()
 
        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))

        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        logits.update_logits(output, input.size(0))
        prec1_tr, prec5_tr = accuracy(output.data, target, topk=(1, 5))
        top1_tr.update(prec1_tr.item(), input.size(0))
        top5_tr.update(prec5_tr.item(), input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()
        if i % 2 == 0:
            print('iter: {}, loss: {}'.format(i, loss.item()))

    print('Epoch: [{0}] Prec@1 {top1_tr.avg:.3f} Prec@5 {top5_tr.avg:.3f} Loss {losses.avg:.4f}'
          .format(epoch, top1_tr=top1_tr, top5_tr=top5_tr, losses=losses))


    losses_tr.append(losses.avg)
    tp1_tr.append(top1_tr.avg)
    tp5_tr.append(top5_tr.avg)


def validate(val_loader, model, criterion, time_steps):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    logits = AverageMeter()


    # switch to evaluate mode
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        labels = Variable(target.cuda())
        target = target.cuda()
        if device_num < 2:
            input_var = Variable(input.type(torch.cuda.FloatTensor))
        else:
            input_var = torch.autograd.Variable(input.type(torch.cuda.FloatTensor))
        with torch.no_grad():
            output = model(input=input_var, steps=time_steps, training=False)
        targetN = output.data.clone().zero_().cuda()
        targetN.scatter_(1, target.unsqueeze(1), 1)
        targetN = Variable(targetN.type(torch.cuda.FloatTensor))
        loss = criterion(output, targetN)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        logits.update_logits(output, input.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print('Test: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {losses.avg:.4f}'
          .format(top1=top1, top5=top5, losses=losses))
    tp1.append(top1.avg)
    tp5.append(top5.avg)
    losses_eval.append(losses.avg)
    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint_dvs128.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_dvs128.pth.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum = self.sum + val * n
        self.count = self.count + n
        self.avg = self.sum / self.count

    def update_logits(self, val, n=1):
        self.val = val
        self.sum = self.sum + torch.sum(val, 0)
        self.count = self.count + n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    for param_group in optimizer.param_groups:

        if epoch in change_arr:
            param_group['lr'] = param_group['lr']*0.5
    lRate.append(param_group['lr'])


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].float().sum()
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class RLIF(nn.Module):


    def __init__(self, C, H, W, decay=0.9, dp_rate=0.25, use_dropout=False):
        super(RLIF, self).__init__()
        self.C = C
        self.H = H
        self.W = W
        self.use_dp = use_dropout

        self.mem = torch.zeros((self.C, self.H, self.W)).cuda()

        self.dp = nn.Dropout(dp_rate)


    def reset_parameters(self):
        self.mem = torch.zeros((self.C, self.H, self.W)).cuda()

    def forward(self, input, steps, training=True):
        I = input
        if steps == 0:
            self.reset_parameters()
        self.mem, out, mem = self.Relaxion_LIF_Neuron(self.mem, I, 1.0, 0.3, training=training)
        if training and self.use_dp:
            out = self.dp(out)

        return out, mem


    def Relaxion_LIF_Neuron(self, membrane_potential, I, threshold, l, training=True):
        # check exceed membrane potential and reset

        I = nn.functional.layer_norm(I, I.size()[-2:])

        mp_output = membrane_potential + I

        keep = (mp_output - threshold).gt(0).type(torch.cuda.FloatTensor)
        mem =  (mp_output) * keep
        out = mem
        if training is False:
            out = out.gt(0).type(torch.cuda.FloatTensor)
            
        else:
            out = torch.div(out, (out).detach()+1e-10)

        acc_mp = l*(mp_output - out*threshold)

        return acc_mp, out, mem


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()

        self.rlif_0 = RLIF(C=64, H=64, W=64, decay=0.9)

        self.rlif_1 = RLIF(C=64, H=32, W=32, decay=0.9)
        self.rlif_block1 = nn.ModuleList([])
        self.rlif_block2 = nn.ModuleList([])
        self.cnn_block1 = nn.ModuleList([])
        self.cnn_block2 = nn.ModuleList([])
        res_block_ids = [1,3]
        for i in range(6):
            self.rlif_block1.append(RLIF(C=64, H=16, W=16, decay=0.9, use_dropout=True))
            if i == 0:
                seq = nn.Sequential(
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),) 
            else:
                seq = nn.Sequential(
                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),)
                if i - 1 == 1:            
                    seq = nn.Sequential(
                            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.LayerNorm([16, 16], elementwise_affine=False))
            self.cnn_block1.append(seq)
        for i in range(8):
            use_dropout = True
            if i == 7:
                use_dropout = False
            self.rlif_block2.append(RLIF(C=128, H=8, W=8, decay=0.9, use_dropout=use_dropout))
            if i == 0:
                seq = nn.Sequential(
                            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
                            nn.LayerNorm([8, 8], elementwise_affine=False))
            else:
                seq = nn.Sequential(
                            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),)
                if i - 1 == 1:
                    seq = nn.Sequential(
                            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
                            nn.LayerNorm([8, 8], elementwise_affine=False))
            self.cnn_block2.append(seq)


        self.cnn00 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False) # encoder
        # self.bn00 = nn.BatchNorm2d(64)

        self.cnn1 =  nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False) # resdual connect to shortcut11

        self.shortcut11 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                        nn.LayerNorm([16, 16], elementwise_affine=False)) # cnn1 to block1-2's input
    
        self.shortcut12 = nn.Sequential() # block1-1's output to block1-4's input

        self.shortcut13 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                        nn.LayerNorm([8, 8], elementwise_affine=False)) # block1-3's output to block2-0's input

        self.shortcut1 = nn.ModuleList([self.shortcut11, self.shortcut12, self.shortcut13])

        self.shortcut21 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                        nn.LayerNorm([8, 8], elementwise_affine=False)) # block1-5's output to block-2-2's input

        self.shortcut22 = nn.Sequential() # block2-1's output to block2-4's input

        self.shortcut23 = nn.Sequential() # block2-3's output to block2-6's input

        self.shortcut24 = nn.Sequential() # block2-5's output to block3-0's input

        self.shortcut2 = nn.ModuleList([self.shortcut21, self.shortcut22, self.shortcut23, self.shortcut24])


        self.avgpool = nn.AvgPool2d(kernel_size=2)
        self.fc = nn.Linear(128*4*4, 11, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
                variance1 = math.sqrt(2.0 / n)
                m.weight.data.normal_(0, variance1)


            elif isinstance(m, nn.Linear):
                size = m.weight.size()
                fan_in = size[1]  # number of columns
                variance2 = math.sqrt(2.0 / fan_in)
                m.weight.data.normal_(0.0, variance2)


    def forward(self, input, steps=100, training=True):

        with torch.no_grad():
            membrane_f1 = Variable(torch.zeros(input.size(0), 11).cuda())

            outc_input = Variable(torch.zeros(input.size(0), 64, 64, 64).cuda())
            outc_1 = Variable(torch.zeros(input.size(0), 64, 32, 32).cuda())

            outc_block1 = [Variable(torch.zeros(input.size(0), 64, 16, 16).cuda()) for i in range(6)]
            outc_block2 = [Variable(torch.zeros(input.size(0), 128, 8, 8).cuda()) for i in range(8)]

            memc_input = Variable(torch.zeros(input.size(0), 64, 64, 64).cuda())
            memc_1 = Variable(torch.zeros(input.size(0), 64, 32, 32).cuda())

            memc_block1 = [Variable(torch.zeros(input.size(0), 64, 16, 16).cuda()) for i in range(6)]
            memc_block2 = [Variable(torch.zeros(input.size(0), 128, 8, 8).cuda()) for i in range(8)]
            outc_0 = Variable(torch.zeros(input.size(0), 128*4*4).cuda())


        input = input.permute(1, 0, 2, 3, 4)

        steps = input.size(0)

        for i in range(input.size()[0]):
            I_0 = self.cnn00(input[i])
            out_00, mem = self.rlif_0(I_0, steps=i, training=training)
            I_1 = self.cnn1(out_00)
            # I_1 = self.bn1(I_1)
            out_1, mem_1_1 = self.rlif_1(I_1, steps=i, training=training)
            out = out_1
            res_inp = out_1
            res_bock1_ids = [1,3]
            b1_cnt = 0
            for j in range(6):
                # print(j)
                Inp = self.cnn_block1[j](out)
                if j-1 in res_bock1_ids:
                    shortcut = self.shortcut1[b1_cnt](res_inp)
                    Inp = Inp + shortcut
                    if j-1 == res_bock1_ids[-1]:
                        res_inp = out
                    else:
                        res_inp = mem_cur
                    b1_cnt = b1_cnt + 1
                out, mem_cur = self.rlif_block1[j](Inp, steps=i, training=training)
                outc_block1[j] += out

            res_block2_ids = [1,3,5]
            b2_cnt = 0
            for j in range(8):
                Inp = self.cnn_block2[j](out)
                if j == 0:
                    Inp = Inp + self.shortcut1[-1](res_inp)
                    res_inp = out
                else:
                    if j-1 in res_block2_ids:
                        shortcut = self.shortcut2[b2_cnt](res_inp)
                        Inp = Inp + shortcut
                        res_inp = mem_cur
                        b2_cnt = b2_cnt + 1
                out, mem_cur = self.rlif_block2[j](Inp, steps=i, training=training)
                outc_block2[j] += out

            out = mem_cur + res_inp
            out = nn.functional.layer_norm(out, out.size()[-2:])

            out = self.avgpool(out)
            out = out.view(out.size(0),-1)

            fc1_out = self.fc(out)

            membrane_f1 = membrane_f1 + fc1_out


        return membrane_f1/steps

if __name__ == '__main__':
    main()