###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils import data
import torchvision.transforms as transform
from torch.nn.parallel.scatter_gather import gather
import paddle
import encoding.utils as utils
from encoding.nn import SegmentationLosses, SyncBatchNorm, BatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model

from .option import Options

import collections
from reprod_log import ReprodLogger
torch.set_printoptions(precision=8)
torch_ver = torch.__version__[:3]
if torch_ver == '0.3':
    from torch.autograd import Variable



class Trainer():
    def __init__(self, args):
        self.args = args
        # data transforms
        input_transform = transform.Compose([
            transform.ToTensor(),
            transform.Normalize([.485, .456, .406], [.229, .224, .225])])
        # dataset
        data_kwargs = {'transform': input_transform, 'base_size': args.base_size,
                       'crop_size': args.crop_size}
        trainset = get_segmentation_dataset(args.dataset, split=args.train_split, mode='train',
                                           **data_kwargs)
        testset = get_segmentation_dataset(args.dataset, split='val', mode ='val',
                                           **data_kwargs)
        # dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True} \
            if args.cuda else {}
        self.trainloader = data.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=True, shuffle=True, **kwargs)
        self.valloader = data.DataLoader(testset, batch_size=args.batch_size,
                                         drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class
        # model
        model = get_segmentation_model(args.model, dataset = args.dataset,
                                       backbone = args.backbone, dilated = args.dilated,
                                       lateral = args.lateral, jpu = args.jpu, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer = BatchNorm,
                                       base_size = args.base_size, crop_size = args.crop_size)

        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'lr': args.lr},]
        if hasattr(model, 'jpu'):
            params_list.append({'params': model.jpu.parameters(), 'lr': args.lr*10})
        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'lr': args.lr*10})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'lr': args.lr*10})
        # optimizer = torch.optim.SGD(params_list, lr=args.lr,
        #     momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
        # criterions
        self.criterion = SegmentationLosses(se_loss=args.se_loss, aux=args.aux,
                                            nclass=self.nclass, 
                                            se_weight=args.se_weight,
                                            aux_weight=args.aux_weight)
        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            print("using cuda")
            self.model = DataParallelModel(self.model).cuda()
            self.criterion = DataParallelCriterion(self.criterion).cuda()
        # resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        # clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        # lr scheduler
        self.scheduler = utils.LR_Scheduler(args.lr_scheduler, args.lr,
                                            args.epochs, len(self.trainloader))

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.trainloader)
        for i, (image, target) in enumerate(tbar):
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            if torch_ver == "0.3":
                image = Variable(image)
                target = Variable(target)

            outputs = self.model(image)
            loss = self.criterion(*outputs, target)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            utils.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, self.args, is_best, filename='checkpoint_{}.pth.tar'.format(epoch))


    def validation(self, epoch):
        # Fast test during the training
        def eval_batch(model, image, target):
            outputs = model(image)
            outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]
            target = target.cuda()
            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
            return correct, labeled, inter, union

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.valloader, desc='\r')
        for i, (image, target) in enumerate(tbar):
            if torch_ver == "0.3":
                image = Variable(image, volatile=True)
                correct, labeled, inter, union = eval_batch(self.model, image, target)
            else:
                with torch.no_grad():
                    correct, labeled, inter, union = eval_batch(self.model, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            tbar.set_description(
                'pixAcc: %.3f, mIoU: %.3f' % (pixAcc, mIoU))

        new_pred = (pixAcc + mIoU)/2
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_pred': new_pred,
        }, self.args, is_best)

    def save_model(self):
        self.model.train()
        torch.save(self.model.state_dict(), 'model_pytorch.pt')

    def show_pkl(self):
        path_pytorch = "./model_pytorch.pt"
        torch_dict = torch.load(path_pytorch, map_location=torch.device('cpu'))
        for key in torch_dict:
            print(key)

    def show_pd(self):
        path_paddle = "./model_paddle.pdparams"
        paddle_dict = paddle.load(path_paddle)
        for key in paddle_dict:
            print(key)

    def pytorch2paddle(self):
        input_fp = "./model_pytorch.pt"
        output_fp = "../FastFCN_paddle/model_paddle.pdparams"
        torch_dict = torch.load(input_fp, map_location=torch.device('cpu'))
        paddle_dict = torch_dict
        fc_names = ["module.pretrained.fc.weight", "module.head.encmodule.fc.0.weight", "module.head.encmodule.selayer.weight"]
        for key in paddle_dict:
            weight = paddle_dict[key].cpu().detach().numpy()
            flag = [i in key for i in fc_names]
            if any(flag):
                print("weight {} need to be trans".format(key))
                weight = weight.transpose()
            paddle_dict[key] = weight
        paddle.save(paddle_dict, output_fp)

        paddle_dict = collections.OrderedDict(
            [(k.replace('running_mean', '_mean'), v) if 'running_mean' in k else (k, v) for k, v in
             paddle_dict.items()])
        paddle_dict = collections.OrderedDict(
            [(k.replace('running_var', '_variance'), v) if 'running_var' in k else (k, v) for k, v in
             paddle_dict.items()])
        paddle_dict = collections.OrderedDict(
            [(k.replace('encmodule', 'wangzhenglai'), v) if 'encmodule' in k else (k, v) for k, v in
             paddle_dict.items()])
        paddle_dict = collections.OrderedDict(
            [(k.replace('module.', ''), v) if 'module.' in k else (k, v) for k, v in
             paddle_dict.items()])
        paddle_dict = collections.OrderedDict(
            [(k.replace('wangzhenglai', 'encmodule'), v) if 'wangzhenglai' in k else (k, v) for k, v in
             paddle_dict.items()])

        paddle.save(paddle_dict, output_fp)

    def forward_pytorch(self):
        np.random.seed(0)
        torch.manual_seed(0)
        reprod_logger = ReprodLogger()
        self.model.load_state_dict(torch.load("./model_pytorch.pt"))
        self.model.cuda()
        self.model.eval()
        # read or gen fake data
        fake_data = np.load("../fake_data.npy")
        fake_data = torch.from_numpy(fake_data).cuda()
        # forward
        out = self.model(fake_data)
        print(out)
        reprod_logger.add("out[0]", out[0].cpu().detach().numpy())
        reprod_logger.add("out[1]", out[2].cpu().detach().numpy())
        # reprod_logger.add("out[2]", out[2].cpu().detach().numpy())
        reprod_logger.save("../diff/forward_pytorch.npy")

    def metric_pytorch(self):
        np.random.seed(0)
        torch.manual_seed(0)
        reprod_logger = ReprodLogger()
        self.model.cuda()
        self.model.load_state_dict(torch.load("./model_pytorch.pt"))
        self.model.eval()
        def eval_batch(model, image, target):
            outputs = model(image)
            # outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]
            target = target.cuda()

            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
            return correct, labeled, inter, union
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.valloader, desc='\r')
        for i, (image, target) in enumerate(tbar):

            with torch.no_grad():
                correct, labeled, inter, union = eval_batch(self.model, image, target)
            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            break
        print("mIoU:%.5f"%mIoU)
        reprod_logger.add("mIoU", np.array(mIoU))
        reprod_logger.save("../diff/metric_pytorch.npy")


    def loss_pytorch(self):
        np.random.seed(0)
        torch.manual_seed(0)
        reprod_logger = ReprodLogger()
        self.model.cuda()
        self.model.load_state_dict(torch.load("./model_pytorch.pt"))
        self.model.eval()
        # read or gen fake data
        fake_data = np.load("../fake_data.npy")
        fake_data = torch.from_numpy(fake_data).cuda()
        # forward
        fake_label = np.load("../fake_label.npy")
        fake_label = torch.from_numpy(fake_label).cuda()
        outputs = self.model(fake_data)
        loss = self.criterion(*outputs, fake_label.long())
        print("loss:", loss)
        reprod_logger.add("loss", loss.cpu().detach().numpy())
        reprod_logger.save("../diff/loss_pytorch.npy")
    #
    def bp_align_pytorch(self):
        np.random.seed(0)
        torch.manual_seed(0)
        reprod_logger = ReprodLogger()

        self.model.cuda()
        self.model.load_state_dict(torch.load("./model_pytorch.pt"))
        self.model.train()
        # read or gen fake data
        fake_data = np.load("../fake_data.npy")
        fake_data = torch.from_numpy(fake_data).cuda()
        # forward
        fake_label = np.load("../fake_label.npy")
        fake_label = torch.from_numpy(fake_label).cuda()
        loss_list = []
        for idx in range(3):
            fake_data = torch.tensor(fake_data)
            fake_label = torch.tensor(fake_label)
            output = self.model(fake_data)
            loss = self.criterion(*output, fake_label.long())
            loss.backward()
            print("loss:", loss)
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_list.append([loss.cpu().detach().numpy()])
        print(loss_list)
        reprod_logger.add("loss_list", np.array(loss_list))
        reprod_logger.save("../diff/bp_align_pytorch.npy")


if __name__ == "__main__":
    args = Options().parse()
    torch.manual_seed(args.seed)
    trainer = Trainer(args)

    #####################
    # trainer.save_model()
    # trainer.pytorch2paddle()
    # trainer.show_pkl()
    # trainer.show_pd()
    # trainer.forward_pytorch()
    # trainer.loss_pytorch()
    # trainer.metric_pytorch()
    trainer.bp_align_pytorch()
    ####################

    #
    # print('Starting Epoch:', trainer.args.start_epoch)
    # print('Total Epoches:', trainer.args.epochs)
    # for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
    #     trainer.training(epoch)
    #     if not trainer.args.no_val:
    #         trainer.validation(epoch)
