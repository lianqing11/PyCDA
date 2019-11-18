'''
define the convolutinal gaussian blur
define the softmax loss

'''
import time
from tqdm import tqdm
import os
import json
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from models import ModelBuilder, SegmentationModule
from lib.nn import user_scattered_collate, patch_replication_callback
from torch.autograd import Variable
import segtransforms
import torch.backends.cudnn as cudnn
import os.path as osp
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from utils.utils import create_logger, AverageMeter, robust_binary_crossentropy, bugged_cls_bal_bce, log_cls_bal
from utils.utils import save_checkpoint as save_best_checkpoint
from utils import transforms_seg
from torchvision import transforms
from dataset.gta5_dataset import GTA5DataSet
from dataset.cityscapes_dataset import cityscapesDataSet, fake_cityscapesDataSet
from PIL import Image
IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Adabn Network")
    parser.add_argument('--config', type=str, default='cfgs/adabn_exp001.yaml')
    return parser.parse_args()


args = get_arguments()

def mkdirs(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)



def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 2]
    x2 = dets[:, 1]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w*h
        over = inter / (areas[i] + areas[order[1:]] - inter + 1e-45)
        inds = np.where(over <= thresh)[0]
        order = order[inds + 1]
    return keep



def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))




def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def adjust_learning_rate(optimizer, cur_iter, learning_rate, args):
    scale_running_lr = ((1. - float(cur_iter) / args.num_steps) ** args.lr_pow)
    running_lr = learning_rate * scale_running_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = running_lr


def create_optimizer(nets, args):
    (net_encoder, net_decoder, net_discriminator, net_reconst) = nets
    optimizer_encoder = None
    optimizer_decoder = None
    optimizer_disc = None
    optimizer_reconst = None

    optimizer_encoder = torch.optim.SGD(
        group_weight(net_encoder),
        lr=args.lr_encoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    if args.arch_decoder:
        optimizer_decoder = torch.optim.SGD(
        group_weight(net_decoder),
        lr=args.lr_decoder,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    if args.arch_disc:
        optimizer_disc = torch.optim.SGD(
        group_weight(net_discriminator),
        lr=args.lr_disc,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    if args.arch_reconst:
        optimizer_reconst = torch.optim.SGD(
        group_weight(net_reconst),
        lr=args.lr_reconst,
        momentum=args.beta1,
        weight_decay=args.weight_decay)
    return (optimizer_encoder, optimizer_decoder, optimizer_disc, optimizer_reconst)

def save_checkpoint(save_model, which_model, i_iter, args, is_best=True):
    suffix = '{}_i_iter'.format(which_model)
    dict_model = save_model.state_dict()
    print(args.snapshot_dir + suffix)
    save_best_checkpoint(dict_model, is_best, os.path.join(args.snapshot_dir, suffix))

def main():
    """Create the model and start the training."""
    with open(args.config) as f:
        config = yaml.load(f)
    for k, v in config['common'].items():
        setattr(args, k, v)
    mkdirs(osp.join("logs/"+args.exp_name))

    logger = create_logger('global_logger', "logs/" + args.exp_name + '/log.txt')
    logger.info('{}'.format(args))
##############################

    for key, val in vars(args).items():
        logger.info("{:16} {}".format(key, val))
    logger.info("random_scale {}".format(args.random_scale))
    logger.info("is_training {}".format(args.is_training))

    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    h, w = map(int, args.input_size_target.split(','))
    input_size_target = (h, w)
    print(type(input_size_target[1]))
    cudnn.enabled = True
    args.snapshot_dir = args.snapshot_dir + args.exp_name
    tb_logger = SummaryWriter("logs/"+args.exp_name)
##############################

#validation data
    h, w = map(int, args.input_size_test.split(','))
    input_size_test = (h,w)
    h, w = map(int, args.com_size.split(','))
    com_size = (h, w)
    h, w = map(int, args.input_size_crop.split(','))
    input_size_crop = h,w
    h,w = map(int, args.input_size_target_crop.split(','))
    input_size_target_crop = h,w


    test_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
                         transforms.Resize((input_size_test[1], input_size_test[0])),
                         transforms.ToTensor(),
                         test_normalize])

    valloader = data.DataLoader(cityscapesDataSet(
                                       args.data_dir_target,
                                       args.data_list_target_val,
                                       crop_size=input_size_test,
                                       set='train',
                                       transform=test_transform),num_workers=args.num_workers,
                                 batch_size=1, shuffle=False, pin_memory=True)
    with open('./dataset/cityscapes_list/info.json', 'r') as fp:
        info = json.load(fp)
    mapping = np.array(info['label2train'], dtype=np.int)
    label_path_list_val = args.label_path_list_val
    label_path_list_test = args.label_path_list_test
    label_path_list_test = './dataset/cityscapes_list/label.txt'
    gt_imgs_val = open(label_path_list_val, 'r').read().splitlines()
    gt_imgs_val = [osp.join(args.data_dir_target_val, x) for x in gt_imgs_val]
    testloader = data.DataLoader(cityscapesDataSet(
                                    args.data_dir_target,
                                    args.data_list_target_test,
                                    crop_size=input_size_test,
                                    set='val',
                                    transform=test_transform),
                            num_workers=args.num_workers,
                            batch_size=1,
                            shuffle=False, pin_memory=True)

    gt_imgs_test = open(label_path_list_test ,'r').read().splitlines()
    gt_imgs_test = [osp.join(args.data_dir_target_test, x) for x in gt_imgs_test]

    name_classes = np.array(info['label'], dtype=np.str)
    interp_val = nn.Upsample(size=(com_size[1], com_size[0]),mode='bilinear', align_corners=True)

    ####
    #build model
    ####
    builder = ModelBuilder()
    net_encoder = builder.build_encoder(
        arch=args.arch_encoder,
        fc_dim=args.fc_dim,
        weights=args.weights_encoder)
    net_decoder = builder.build_decoder(
        arch=args.arch_decoder,
        fc_dim=args.fc_dim,
        num_class=args.num_classes,
        weights=args.weights_decoder,
        use_aux=True)



    model = SegmentationModule(
        net_encoder, net_decoder, args.use_aux)

    if args.num_gpus > 1:
        model = torch.nn.DataParallel(model)
        patch_replication_callback(model)
    model.cuda()

    nets = (net_encoder, net_decoder, None, None)
    optimizers = create_optimizer(nets, args)
    cudnn.enabled=True
    cudnn.benchmark=True
    model.train()



    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]


    source_normalize = transforms_seg.Normalize(mean=mean,
                                                std=std)

    mean_mapping = [0.485, 0.456, 0.406]
    mean_mapping = [item * 255 for item in mean_mapping]

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)
    source_transform = transforms_seg.Compose([
                             transforms_seg.Resize([input_size[1], input_size[0]]),
                             segtransforms.RandScale((args.scale_min, args.scale_max)),
                             #segtransforms.RandRotate((args.rotate_min, args.rotate_max), padding=mean_mapping, ignore_label=args.ignore_label),
                             #segtransforms.RandomGaussianBlur(),
                             segtransforms.RandomHorizontalFlip(),
                             segtransforms.Crop([input_size_crop[1], input_size_crop[0]], crop_type='rand', padding=mean_mapping, ignore_label=args.ignore_label),
                             transforms_seg.ToTensor(),
                             source_normalize])
    target_normalize = transforms_seg.Normalize(mean=mean,
                                            std=std)
    target_transform = transforms_seg.Compose([
                             transforms_seg.Resize([input_size_target[1], input_size_target[0]]),
                             segtransforms.RandScale((args.scale_min, args.scale_max)),
                             #segtransforms.RandRotate((args.rotate_min, args.rotate_max), padding=mean_mapping, ignore_label=args.ignore_label),
                             #segtransforms.RandomGaussianBlur(),
                             segtransforms.RandomHorizontalFlip(),
                             segtransforms.Crop([input_size_target_crop[1], input_size_target_crop[0]],crop_type='rand', padding=mean_mapping, ignore_label=args.ignore_label),
                             transforms_seg.ToTensor(),
                             target_normalize])
    trainloader = data.DataLoader(
        GTA5DataSet(args.data_dir, args.data_list, max_iters=args.num_steps * args.iter_size * args.batch_size,
                    crop_size=input_size, transform = source_transform),
        batch_size=args.batch_size, shuffle=True, num_workers=1, pin_memory=True)

    trainloader_iter = enumerate(trainloader)

    targetloader = data.DataLoader(fake_cityscapesDataSet(args.data_dir_target, args.data_list_target,
                                                     max_iters=args.num_steps * args.iter_size * args.batch_size,
                                                     crop_size=input_size_target,
                                                     set=args.set,
                                                     transform=target_transform),
                                   batch_size=args.batch_size, shuffle=True, num_workers=1,
                                   pin_memory=True)


    targetloader_iter = enumerate(targetloader)
    # implement model.optim_parameters(args) to handle different models' lr setting


    criterion_seg = torch.nn.CrossEntropyLoss(ignore_index=255,reduce=False)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), align_corners=True, mode='bilinear')

    # labels for adversarial training
    source_label = 0
    target_label = 1


    optimizer_encoder, optimizer_decoder, optimizer_disc, optimizer_reconst = optimizers
    batch_time = AverageMeter(10)
    loss_seg_value1 = AverageMeter(10)
    is_best_test = True
    best_mIoUs = 0
    loss_seg_value2 = AverageMeter(10)
    loss_balance_value = AverageMeter(10)
    loss_pseudo_value = AverageMeter(10)
    bounding_num = AverageMeter(10)
    pseudo_num = AverageMeter(10)

    for i_iter in range(args.num_steps):
        # train G

        # don't accumulate grads in D

        end = time.time()
        _, batch = trainloader_iter.__next__()
        images, labels, _ = batch
        images = Variable(images).cuda(async=True)
        labels = Variable(labels).cuda(async=True)
        seg, aux_seg, loss_seg2, loss_seg1 = model(images, labels)


        loss_seg2 = torch.mean(loss_seg2)
        loss_seg1 = torch.mean(loss_seg1)
        loss = loss_seg2+args.lambda_seg*loss_seg1
        #logger.info(loss_seg1.data.cpu().numpy())
        loss_seg_value2.update(loss_seg2.data.cpu().numpy())
        # train with target
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()
        loss.backward()
        optimizer_encoder.step()
        optimizer_decoder.step()

        del seg, loss_seg2

        _, batch = targetloader_iter.__next__()
        with torch.no_grad():
            images, labels, _ = batch
            images = Variable(images).cuda(async=True)
            result = model(images, None)
            del result



        batch_time.update(time.time() - end)

        remain_iter = args.num_steps - i_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))




        adjust_learning_rate(optimizer_encoder, i_iter, args.lr_encoder, args)
        adjust_learning_rate(optimizer_decoder, i_iter, args.lr_decoder, args)
        if i_iter % args.print_freq == 0:
            lr_encoder = optimizer_encoder.param_groups[0]['lr']
            lr_decoder = optimizer_decoder.param_groups[0]['lr']
            logger.info('exp = {}'.format(args.snapshot_dir))
            logger.info('Iter = [{0}/{1}]\t'
                        'Time = {batch_time.avg:.3f}\t'
                        'loss_seg1 = {loss_seg1.avg:4f}\t'
                        'loss_seg2 = {loss_seg2.avg:.4f}\t'
                        'lr_encoder = {lr_encoder:.8f} lr_decoder = {lr_decoder:.8f}'.format(
                         i_iter, args.num_steps, batch_time=batch_time,
                         loss_seg1=loss_seg_value1, loss_seg2=loss_seg_value2,
                         lr_encoder=lr_encoder,
                         lr_decoder=lr_decoder))


            logger.info("remain_time: {}".format(remain_time))
            if not tb_logger is None:
                tb_logger.add_scalar('loss_seg_value1', loss_seg_value1.avg, i_iter)
                tb_logger.add_scalar('loss_seg_value2', loss_seg_value2.avg, i_iter)
                tb_logger.add_scalar('lr', lr_encoder, i_iter)
            #####
            #save image result

            if i_iter % args.save_pred_every == 0 and i_iter != 0:
                logger.info('taking snapshot ...')
                model.eval()

                val_time = time.time()
                hist = np.zeros((19,19))
                for index, batch in tqdm(enumerate(valloader)):
                    with torch.no_grad():
                        image, name = batch
                        output2, _ = model(Variable(image).cuda(), None)
                        pred = interp_val(output2)
                        del output2
                        pred = pred.cpu().data[0].numpy()
                        pred = pred.transpose(1, 2, 0)
                        pred = np.asarray(np.argmax(pred, axis=2), dtype=np.uint8)
                        label = np.array(Image.open(gt_imgs_val[index]))
                        #label = np.array(label.resize(com_size, Image.
                        label = label_mapping(label, mapping)
                        #logger.info(label.shape)
                        hist += fast_hist(label.flatten(), pred.flatten(), 19)
                mIoUs = per_class_iu(hist)
                for ind_class in range(args.num_classes):
                    logger.info('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
                    tb_logger.add_scalar(name_classes[ind_class] + '_mIoU', mIoUs[ind_class], i_iter)

                mIoUs = round(np.nanmean(mIoUs) *100, 2)
                if mIoUs >= best_mIoUs:
                    is_best_test = True
                    best_mIoUs = mIoUs
                else:
                    is_best_test = False

                logger.info("current mIoU {}".format(mIoUs))
                logger.info("best mIoU {}".format(best_mIoUs))
                tb_logger.add_scalar('val mIoU', mIoUs, i_iter)
                tb_logger.add_scalar('val mIoU', mIoUs, i_iter)
                net_encoder, net_decoder, net_disc, net_reconst = nets
                save_checkpoint(net_encoder, 'encoder', i_iter, args, is_best_test)
                save_checkpoint(net_decoder, 'decoder', i_iter, args, is_best_test)
            model.train()
if __name__ == '__main__':
    main()
