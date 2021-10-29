###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import os
import numpy as np
from tqdm import tqdm

import paddle
import paddle.vision.transforms as transform
# from paddle.nn.parallel.scatter_gather import gather

import encoding.utils as utils
from encoding.nn import SegmentationLosses
from paddle.nn import SyncBatchNorm, BatchNorm
from encoding.parallel import DataParallelModel, DataParallelCriterion
from encoding.datasets import get_segmentation_dataset
from encoding.models import get_segmentation_model

from .option import Options
from reprod_log import ReprodLogger
import collections

paddle_ver = paddle.__version__[:3]

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
        kwargs = {'num_workers': args.workers} \
            if args.cuda else {}
        self.trainloader = paddle.io.DataLoader(trainset, batch_size=args.batch_size,
                                           drop_last=True, shuffle=True, **kwargs)
        self.valloader = paddle.io.DataLoader(testset, batch_size=args.batch_size,
                                         drop_last=False, shuffle=False, **kwargs)
        self.nclass = trainset.num_class
        # model
        model = get_segmentation_model(args.model, dataset = args.dataset,
                                       backbone = args.backbone, dilated = args.dilated,
                                       lateral = args.lateral, jpu = args.jpu, aux = args.aux,
                                       se_loss = args.se_loss, norm_layer = BatchNorm,
                                       base_size = args.base_size, crop_size = args.crop_size)
        print(model)

        # optimizer using different LR
        params_list = [{'params': model.pretrained.parameters(), 'learning_rate': args.lr},]
        if hasattr(model, 'jpu'):
            params_list.append({'params': model.jpu.parameters(), 'learning_rate': args.lr*10})
        if hasattr(model, 'head'):
            params_list.append({'params': model.head.parameters(), 'learning_rate': args.lr*10})
        if hasattr(model, 'auxlayer'):
            params_list.append({'params': model.auxlayer.parameters(), 'learning_rate': args.lr*10})
        # optimizer = paddle.optimizer.Momentum(learning_rate=args.lr, parameters=params_list, momentum=args.momentum,
        #                                       weight_decay=args.weight_decay)
        optimizer = paddle.optimizer.Momentum(learning_rate=args.lr, parameters=model.parameters(), momentum=args.momentum,
                                                                                     weight_decay=args.weight_decay)

        # criterions
        self.criterion = SegmentationLosses(se_loss=args.se_loss, aux=args.aux,
                                            nclass=self.nclass, 
                                            se_weight=args.se_weight,
                                            aux_weight=args.aux_weight)
        self.model, self.optimizer = model, optimizer
        # using cuda
        if args.cuda:
            paddle.set_device("gpu")
        else:
            paddle.set_device("cpu")
            # self.model = DataParallelModel(self.model)
            # self.criterion = DataParallelCriterion(self.criterion)
        # resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = paddle.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            self.model.load_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_dict(checkpoint['optimizer'])
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
            self.optimizer.clear_grad()

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
            # outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]
            target = target
            correct, labeled = utils.batch_pix_accuracy(pred.data, target)
            inter, union = utils.batch_intersection_union(pred.data, target, self.nclass)
            return correct, labeled, inter, union

        is_best = False
        self.model.eval()
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.valloader, desc='\r')
        for i, (image, target) in enumerate(tbar):
            if paddle_ver == "0.3":
                image = paddle.to_tensor(image)
                correct, labeled, inter, union = eval_batch(self.model, image, target)
            else:
                with paddle.no_grad():
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
        paddle.save(self.model.state_dict(), 'model_paddle.pdparams')

    def show_pkl(self):
        path_paddle = "./model_paddle.pdparams"
        paddle_dict = paddle.load(path_paddle)
        for key in paddle_dict:
            print(key)

    def forward_paddle(self):
        paddle.set_device("gpu")
        np.random.seed(0)
        paddle.seed(0)
        reprod_logger = ReprodLogger()
        self.model.load_dict(paddle.load("./model_paddle.pdparams"))
        self.model.eval()
        # read or gen fake data
        fake_data = np.load("../fake_data.npy")
        fake_data = paddle.to_tensor(fake_data)
        # forward
        out = self.model(fake_data)
        print(out)
        reprod_logger.add("out[0]", out[0].cpu().detach().numpy())
        reprod_logger.add("out[1]", out[2].cpu().detach().numpy())
        # reprod_logger.add("out[2]", out[2].cpu().detach().numpy())
        reprod_logger.save("../diff/forward_paddle.npy")

    def metric_paddle(self):
        paddle.set_device("gpu")
        np.random.seed(0)
        paddle.seed(0)
        reprod_logger = ReprodLogger()
        self.model.load_dict(paddle.load("./model_paddle.pdparams"))
        self.model.eval()
        def eval_batch(model, image, target):
            outputs = model(image)
            # outputs = gather(outputs, 0, dim=0)
            pred = outputs[0]
            target = target

            correct, labeled = utils.batch_pix_accuracy(pred, target)
            inter, union = utils.batch_intersection_union(pred, target, self.nclass)
            return correct, labeled, inter, union
        total_inter, total_union, total_correct, total_label = 0, 0, 0, 0
        tbar = tqdm(self.valloader, desc='\r')
        for i, (image, target) in enumerate(tbar):

            with paddle.no_grad():
                correct, labeled, inter, union = eval_batch(self.model, image, target)

            total_correct += correct
            total_label += labeled
            total_inter += inter
            total_union += union
            pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
            IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
            mIoU = IoU.mean()
            break
        print("mIoU:", mIoU)
        reprod_logger.add("mIoU", np.array(mIoU))
        reprod_logger.save("../diff/metric_paddle.npy")


    def loss_paddle(self):
        paddle.set_device("gpu")
        np.random.seed(0)
        paddle.seed(0)
        reprod_logger = ReprodLogger()
        self.model.load_dict(paddle.load("./model_paddle.pdparams"))
        self.model.eval()
        # read or gen fake data
        fake_data = np.load("../fake_data.npy")
        fake_data = paddle.to_tensor(fake_data)
        fake_label = np.load("../fake_label.npy")
        fake_label = paddle.to_tensor(fake_label)

        outputs = self.model(fake_data)
        loss = self.criterion(*outputs, fake_label)
        print("loss:", loss)
        reprod_logger.add("loss", loss.cpu().detach().numpy())
        reprod_logger.save("../diff/loss_paddle.npy")

    def bp_align_paddle(self):
        paddle.set_device("cpu")
        np.random.seed(0)
        paddle.seed(0)
        reprod_logger = ReprodLogger()
        self.model.load_dict(paddle.load("./model_paddle.pdparams"))
        self.model.train()
        # read or gen fake data
        fake_data = np.load("../fake_data.npy")
        fake_label = np.load("../fake_label.npy")
        loss_list = []
        for idx in range(3):
            fake_data = paddle.to_tensor(fake_data)
            fake_label = paddle.to_tensor(fake_label)

            output = self.model(fake_data)
            loss = self.criterion(*output, fake_label)
            print("loss:", loss)
            loss.backward()
            self.optimizer.step()
            self.optimizer.clear_grad()

            loss_list.append(loss.cpu().detach().numpy())
        print(loss_list)
        print(np.array(loss_list).shape)
        reprod_logger.add("loss_list", np.array(loss_list))
        reprod_logger.save("../diff/bp_align_paddle.npy")


if __name__ == "__main__":
    args = Options().parse()
    paddle.seed(args.seed)
    trainer = Trainer(args)
    # print('Starting Epoch:', trainer.args.start_epoch)
    # print('Total Epoches:', trainer.args.epochs)
    # for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
    #     trainer.training(epoch)
    #     if not trainer.args.no_val:
    #         trainer.validation(epoch)
    #####################
    # trainer.save_model()
    # trainer.show_pkl()
    # trainer.forward_paddle()
    # trainer.loss_paddle()
    # trainer.metric_paddle()
    trainer.bp_align_paddle()
    ####################