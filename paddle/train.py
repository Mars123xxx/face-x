import paddle
import paddle.nn as nn
import paddle.optimizer as optim
from config import configurations
from dataload import NormalDataset, BalancingClassDataset
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from backbone.resnet_pp import ResNet, download, load_weight
from loss.focal import FocalLoss
from utils import separate_irse_bn_paras, separate_resnet_bn_paras, schedule_lr, AverageMeter, accuracy, \
    get_time
from head.metrics import Softmax, ArcFace, CosFace, SphereFace, Am_softmax
import os
from tqdm import tqdm
import logging
from paddle.static import InputSpec
from paddleslim.dygraph.quant import QAT
import warnings

warnings.filterwarnings('ignore')
import numpy as np

FORMAT = '%(asctime)s-%(levelname)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)
if __name__ == '__main__':
    # ======= hyper parameters & data loaders =======#
    cfg = configurations[1]

    SEED = cfg['SEED']  # random seed for reproduce results
    paddle.seed(SEED)
    DATA_ROOT = cfg['DATA_ROOT']  # the parent root where your train/val/test data are stored
    MODEL_ROOT = cfg['MODEL_ROOT']  # the root to buffer your checkpoints
    LOG_ROOT = cfg['LOG_ROOT']  # the root to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg[
        'BACKBONE_NAME']  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    HEAD_NAME = cfg['HEAD_NAME']  # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    LOSS_NAME = cfg['LOSS_NAME']  # support: ['Focal', 'Softmax']

    INPUT_SIZE = cfg['INPUT_SIZE']
    RGB_MEAN = cfg['RGB_MEAN']  # for normalize inputs
    RGB_STD = cfg['RGB_STD']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE']  # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST']  # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR']  # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES']  # epoch stages to decay learning rate
    DEVICE = None
    GPU_ID = cfg['GPU_ID']  # specify your GPU ids
    PIN_MEMORY = cfg['PIN_MEMORY']
    NUM_WORKERS = cfg['NUM_WORKERS']
    USE_PRETRAINED = cfg['USE_PRETRAINED']
    save_quant_model = cfg['SAVE_QUANT_MODEL']
    SAVE_CHECKPOINT = cfg['SAVE_CHECKPOINT']
    logger.info("=" * 60)
    logger.info("Overall Configurations:")
    for cfg_, value in cfg.items():
        logger.info(cfg_ + ":" + str(value))
    logger.info("=" * 60)
    logger.info('loading data form file:{}'.format(DATA_ROOT))
    train_dataset = NormalDataset(os.path.join(DATA_ROOT), INPUT_SIZE, RGB_MEAN, RGB_STD)
    num_classes = train_dataset.num_classes
    train_loader = paddle.io.DataLoader(train_dataset,
                                        places=[paddle.CPUPlace()],
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        drop_last=DROP_LAST,
                                        num_workers=NUM_WORKERS)

    NUM_CLASS = train_dataset.num_classes
    logger.info("Number of Training Classes: {}".format(NUM_CLASS))
    # lfw, cfp_ff, cfp_fp, agedb, calfw, cplfw, vgg2_fp, lfw_issame, cfp_ff_issame, cfp_fp_issame, agedb_issame, calfw_issame, cplfw_issame, vgg2_fp_issame = get_val_data(
    #     DATA_ROOT)

    # ======= model & loss & optimizer =======#
    BACKBONE_DICT = {
        'ppResNet_50': ResNet(input_size=INPUT_SIZE, depth=50),
        'ResNet_50': ResNet_50(INPUT_SIZE),
        'ResNet_101': ResNet_101(INPUT_SIZE),
        'ResNet_152': ResNet_152(INPUT_SIZE),
        'IR_50': IR_50(INPUT_SIZE),
        'IR_101': IR_101(INPUT_SIZE),
        'IR_152': IR_152(INPUT_SIZE),
        'IR_SE_50': IR_SE_50(INPUT_SIZE),
        'IR_SE_101': IR_SE_101(INPUT_SIZE),
        'IR_SE_152': IR_SE_152(INPUT_SIZE)}
    BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
    if BACKBONE_NAME == 'ppResNet_50' and USE_PRETRAINED:
        pretrained_weight = download()
        load_weight(model=BACKBONE, weight_path=pretrained_weight)

    logger.info("=" * 60)
    # logger.info(BACKBONE)
    logger.info("{} Backbone Generated".format(BACKBONE_NAME))
    logger.info("=" * 60)

    HEAD_DICT = {'ArcFace': ArcFace(embedding_size=EMBEDDING_SIZE, class_dim=NUM_CLASS),
                 'CosFace': CosFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS),
                 'SphereFace': SphereFace(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS),
                 'Am_softmax': Am_softmax(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS),
                 'Softmax': Softmax(in_features=EMBEDDING_SIZE, out_features=NUM_CLASS)}
    HEAD = HEAD_DICT[HEAD_NAME]
    logger.info("=" * 60)
    # logger.info(HEAD)
    logger.info("{} Head Generated".format(HEAD_NAME))
    logger.info("=" * 60)

    LOSS_DICT = {'Focal': FocalLoss(),
                 'Softmax': nn.CrossEntropyLoss()}
    LOSS = LOSS_DICT[LOSS_NAME]
    logger.info("=" * 60)
    logger.info(LOSS)
    logger.info("{} Loss Generated".format(LOSS_NAME))
    logger.info("=" * 60)

    if BACKBONE_NAME.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(
            BACKBONE)  # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_irse_bn_paras(HEAD)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(
            BACKBONE)  # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_resnet_bn_paras(HEAD)

    DISP_FREQ = len(train_loader)  # frequency to display training loss & acc
    # NUM_EPOCH_WARM_UP = NUM_EPOCH // 25  # use the first 1/25 epochs to warm up
    NUM_EPOCH_WARM_UP = NUM_EPOCH // 3
    print('NUM_EPOCH',NUM_EPOCH)
    print('___________________')
    print('NUM_EPOCH_WARM_UP',NUM_EPOCH_WARM_UP)
    print('___________________')
    # print('NUM_EPOCH',NUM_EPOCH)
    print('train_loader',len(train_loader))
    print('___________________')
    NUM_BATCH_WARM_UP = len(train_loader) * NUM_EPOCH_WARM_UP  # use the first 1/25 epochs to warm up
    scheduler = paddle.optimizer.lr.LinearWarmup(
        learning_rate=LR, warmup_steps=NUM_BATCH_WARM_UP, start_lr=LR / 2, end_lr=LR, verbose=True)
    OPTIMIZER_decay = optim.Momentum(parameters=backbone_paras_wo_bn + head_paras_wo_bn,
                                     learning_rate=scheduler, weight_decay=WEIGHT_DECAY,
                                     momentum=MOMENTUM)
    OPTIMIZER = optim.Momentum(parameters=backbone_paras_only_bn,
                               learning_rate=scheduler, momentum=MOMENTUM)
    logger.info("=" * 60)
    logger.info(OPTIMIZER)
    logger.info("Optimizer Generated")
    logger.info("=" * 60)

    # optionally resume from a checkpoint
    if BACKBONE_RESUME_ROOT and HEAD_RESUME_ROOT:
        logger.info("=" * 60)
        if os.path.isfile(BACKBONE_RESUME_ROOT) and os.path.isfile(HEAD_RESUME_ROOT):
            logger.info("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
            load_weight(model=BACKBONE, weight_path=BACKBONE_RESUME_ROOT)
            logger.info("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
            load_weight(model=HEAD, weight_path=HEAD_RESUME_ROOT)
        else:
            logger.info(
                "No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(
                    BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT))
        logger.info("=" * 60)

    # ======= train & validation & save checkpoint =======#
    batch = 0  # batch index
    BACKBONE.train()  # set to training mode
    HEAD.train()
    if not save_quant_model:
        for epoch in tqdm(range(NUM_EPOCH), ncols=80):  # start training process
            if epoch == STAGES[
                0]:  # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plaueau observed
                schedule_lr(OPTIMIZER)
            if epoch == STAGES[1]:
                schedule_lr(OPTIMIZER)
            if epoch == STAGES[2]:
                schedule_lr(OPTIMIZER)

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            for iters, data in enumerate(train_loader()):
                inputs, labels = data[0], data[1]
                features = BACKBONE(inputs)
                outputs = HEAD(features, labels)
                loss = LOSS(outputs, labels)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
                losses.update(loss.numpy().item(), inputs.shape[0])
                top1.update(prec1.item(), inputs.shape[0])
                top5.update(prec5.item(), inputs.shape[0])

                OPTIMIZER.clear_grad()
                OPTIMIZER_decay.clear_grad()
                loss.backward()
                OPTIMIZER.step()
                OPTIMIZER_decay.step()
            print('\n')
            logger.info(' Epoch {}/{}'
                        'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch + 1, NUM_EPOCH, loss=losses, top1=top1, top5=top5))
            batch += 1  # batch index
            print('=' * 60)
            if SAVE_CHECKPOINT:
                # save checkpoints per epoch
                if not os.path.exists(os.path.join(os.getcwd(), MODEL_ROOT)):
                    os.makedirs(os.path.join(os.getcwd(), MODEL_ROOT))
                backbone_state = BACKBONE.state_dict()
                paddle.save(backbone_state, os.path.join(os.path.join(os.getcwd(), MODEL_ROOT),
                                                        "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pdparams".format(
                                                            BACKBONE_NAME, epoch + 1, batch, get_time())))
                head_state = HEAD.state_dict()
                paddle.save(head_state, os.path.join(os.path.join(os.getcwd(), MODEL_ROOT),
                                                    "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pdparams".format(HEAD_NAME,
                                                                                                                    epoch + 1,
                                                                                                                    batch,
                                                                                                                    get_time()
                                                                                                                    )))
                paddle.save(OPTIMIZER_decay.state_dict(), os.path.join(os.path.join(os.getcwd(), MODEL_ROOT),
                                                                "OPTIMIZER_decay{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pdopt".format(
                                                                    BACKBONE_NAME, epoch + 1, batch, get_time())))
                paddle.save(OPTIMIZER.state_dict(), os.path.join(os.path.join(os.getcwd(), MODEL_ROOT),
                                                            "OPTIMIZER{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pdopt".format(
                                                                HEAD_NAME, epoch + 1, batch, get_time())))
        # save  inference model
        if not os.path.exists(os.path.join(os.getcwd(), MODEL_ROOT)):
            os.makedirs(os.path.join(os.getcwd(), MODEL_ROOT))
        save_backbone = paddle.jit.to_static(
            BACKBONE,
            input_spec=[InputSpec(shape=[None, 3, 112, 112], dtype='float32')]
        )
        paddle.jit.save(save_backbone, os.path.join(os.path.join(os.getcwd(), MODEL_ROOT),
                                                    "Backbone_epoch{}".format(epoch)))
    else:
        quant_config = {
            'weight_preprocess_type': 'PACT',
            'weight_quantize_type': 'channel_wise_abs_max',
            'activation_quantize_type': 'moving_average_abs_max',
            'weight_bits': 8,
            'activation_bits': 8,
            'dtype': 'int8',
            'window_size': 10000,
            'moving_rate': 0.9,
            'quantizable_layer_type': ['Conv2D', 'Linear'],
        }
        quanter = QAT(config=quant_config)
        quanter.quantize(BACKBONE)
        for epoch in tqdm(range(NUM_EPOCH), ncols=80):  # start training process
            if epoch == STAGES[
                0]:  # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plaueau observed
                schedule_lr(OPTIMIZER)
            if epoch == STAGES[1]:
                schedule_lr(OPTIMIZER)
            if epoch == STAGES[2]:
                schedule_lr(OPTIMIZER)

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            for iters, data in enumerate(train_loader()):
                inputs, labels = data[0], data[1]
                features = BACKBONE(inputs)
                outputs = HEAD(features, labels)
                loss = LOSS(outputs, labels)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs, labels, topk=(1, 5))
                losses.update(loss.numpy().item(), inputs.shape[0])
                top1.update(prec1.item(), inputs.shape[0])
                top5.update(prec5.item(), inputs.shape[0])

                OPTIMIZER.clear_grad()
                OPTIMIZER_decay.clear_grad()
                loss.backward()
                OPTIMIZER.step()
                OPTIMIZER_decay.step()
            print('\n')
            logger.info(' Epoch {}/{}'
                        'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch + 1, NUM_EPOCH, loss=losses, top1=top1, top5=top5))
            batch += 1  # batch index
            print('=' * 60)
        # save  inference model
        save_path = os.path.join(os.getcwd(), MODEL_ROOT)
        if not os.path.exists(save_path):
            os.makedirs(os.path.join(os.getcwd(), MODEL_ROOT))
        quanter.save_quantized_model(
            model = BACKBONE,
            path = os.path.join(save_path,'Backbone_int8_epoch{}'.format(epoch)),
            input_spec=[InputSpec(shape=[None, 3, 112, 112], dtype='float32')]
        )


