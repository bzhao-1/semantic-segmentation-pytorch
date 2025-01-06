# # System libs
# import os
# import argparse
# from distutils.version import LooseVersion
# # Numerical libs
# import numpy as np
# import torch
# import torch.nn as nn
# from scipy.io import loadmat
# import csv
# # Our libs
# from mit_semseg.dataset import TestDataset
# from mit_semseg.models import ModelBuilder, SegmentationModule
# from mit_semseg.utils import colorEncode, find_recursive, setup_logger
# from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
# from mit_semseg.lib.utils import as_numpy
# from PIL import Image
# from tqdm import tqdm
# from mit_semseg.config import cfg

# colors = loadmat('data/color150.mat')['colors']
# classMapping = {
#     1: 4,  # wall -> Wall
#     2: 3,  # building -> Building
#     3: 11,  # sky -> Sky
#     4: 25,  # floor -> ground
#     5: 9,  # tree -> Vegetation
#     6: 10,  # ceiling -> Terrain
#     7: 1,  # road -> Roads
#     8: 22,  # bed -> Other
#     9: 3,  # window -> building
#     10: 10,  # grass -> Terrain
#     11: 22,  # cabinet -> other
#     12: 2,  # sidewalk -> sidewalk
#     13: 12,  # person -> Pedestrian
#     14: 10,  # earth -> Terrain
#     15: 3,  # door -> building
#     16: 22,  # table -> other
#     17: 22,  # mountain -> other
#     18: 9,  # plant -> Vegetation
#     19: 22,  # curtain -> other
#     20: 21,  # chair -> dynamic
#     21: 14,  # car -> Car
#     22: 23,  # water -> Water
#     23: 22,  # painting -> other
#     24: 22,  # sofa -> other
#     25: 22,  # shelf -> other
#     26: 3,  # house -> building
#     27: 23,  # sea -> water
#     28: 22,  # mirror -> other
#     29: 22,  # rug -> other
#     30: 10,  # field -> Terrain
#     31: 20,  # armchair -> Static
#     32: 20,  # seat -> Static
#     33: 5,  # fence -> Fence
#     34: 20,  # desk -> Static
#     35: 3,  # rock -> Building
#     36: 20,  # wardrobe -> Static
#     37: 20,  # lamp -> Static
#     38: 23,  # bathtub -> Water
#     39: 5,  # railing -> Fence
#     40: 20,  # cushion -> Static
#     41: 20,  # base -> Static
#     42: 20,  # box -> Static
#     43: 5,  # column -> Fence
#     44: 10,  # signboard -> Terrain
#     45: 20,  # chest of drawers -> Static
#     46: 20,  # counter -> Static
#     47: 10,  # sand -> Terrain
#     48: 23,  # sink -> Water
#     49: 3,  # skyscraper -> Building
#     50: 23,  # fireplace -> Water
#     51: 23,  # refrigerator -> Water
#     52: 20,  # grandstand -> Static
#     53: 10,  # path -> Terrain
#     54: 20,  # stairs -> Static
#     55: 10,  # runway -> Terrain
#     56: 20,  # case -> Static
#     57: 20,  # pool table -> Static
#     58: 20,  # pillow -> Static
#     59: 20,  # screen door -> Static
#     60: 20,  # stairway -> Static
#     61: 23,  # river -> Water
#     62: 10,  # bridge -> Terrain
#     63: 20,  # bookcase -> Static
#     64: 20,  # blind -> Static
#     65: 20,  # coffee table -> Static
#     66: 20,  # toilet -> Static
#     67: 9,  # flower -> Vegetation
#     68: 20,  # book -> Static
#     69: 3,  # hill -> Building
#     70: 20,  # bench -> Static
#     71: 20,  # countertop -> Static
#     72: 20,  # stove -> Static
#     73: 9,  # palm -> Vegetation
#     74: 20,  # kitchen island -> Static
#     75: 20,  # computer -> Static
#     76: 20,  # swivel chair -> Static
#     77: 23,  # boat -> Water
#     78: 20,  # bar -> Static
#     79: 20,  # arcade machine -> Static
#     80: 20,  # hovel -> Static
#     81: 16,  # bus -> Static
#     82: 20,  # towel -> Static
#     83: 20,  # light -> Static
#     84: 15,  # truck -> Truck
#     85: 20,  # tower -> Static
#     86: 20,  # chandelier -> Static
#     87: 20,  # awning -> Static
#     88: 20,  # streetlight -> Static
#     89: 20,  # booth -> Static
#     90: 20,  # television -> Static
#     91: 20,  # airplane -> Static
#     92: 25,  # dirt track -> ground
#     93: 22,  # apparel -> other
#     94: 6,  # pole -> pole
#     95: 25,  # land -> ground
#     96: 22,  # bannister -> other
#     97: 22,  # escalator -> other
#     98: 22,  # ottoman -> other
#     99: 20,  # bottle -> Static
#     100: 22,  # buffet -> other
#     101: 20,  # poster -> Static
#     102: 22,  # stage -> other
#     103: 14,  # van -> car
#     104: 21,  # ship -> dynamic
#     105: 20,  # fountain -> static
#     106: 22,  # conveyer belt -> other
#     107: 22,  # canopy -> other
#     108: 22,  # washer -> other
#     109: 22,  # plaything -> other
#     110: 22,  # swimming pool -> Static
#     111: 22,  # stool -> Static
#     112: 21,  # barrel -> dynamic
#     113: 20,  # basket -> Static
#     114: 23,  # waterfall -> water
#     115: 20,  # tent -> Static
#     116: 21,  # bag -> dynamic
#     117: 19,  # minibike -> bike
#     118: 20,  # cradle -> Static
#     119: 20,  # oven -> other
#     120: 21,  # ball -> dynamic
#     121: 22,  # food -> other
#     122: 22,  # step -> other
#     123: 22,  # tank -> other
#     124: 22,  # trade name -> other
#     125: 22,  # microwave -> other
#     126: 22,  # pot -> Static
#     127: 21,  # animal -> dynamic
#     128: 19,  # bicycle -> bike
#     129: 23,  # lake -> water
#     130: 22,  # dishwasher -> other
#     131: 22,  # screen -> other
#     132: 22,  # blanket -> Static
#     133: 20,  # sculpture -> Static
#     134: 22,  # hood -> other
#     135: 22,  # sconce -> other
#     136: 22,  # vase -> other
#     137: 7,  # traffic light -> traffic light
#     138: 22,  # tray -> Static
#     139: 22,  # ashcan -> Static
#     140: 22,  # fan -> Static
#     141: 22,  # pier -> Static
#     142: 22,  # crt screen -> Static
#     143: 22,  # plate -> Static
#     144: 22,  # monitor -> Static
#     145: 22,  # bulletin board -> Static
#     146: 22,  # shower -> Static
#     147: 22,  # radiator -> Static
#     148: 22,  # glass -> Static
#     149: 22,  # clock -> Static
#     150: 20   # flag -> Static
# }

# # Create a 29-color array initialized with zeros
# new_colors = np.zeros((29, 3), dtype=np.uint8)

# # Populate new_colors by mapping from original_colors based on classMapping
# for original_class_id, new_class_id in classMapping.items():
#     # Map original class color to the new class position
#     new_colors[new_class_id] = colors[original_class_id-1]

# names = {}
# with open('data/object150_info.csv') as f:
#     reader = csv.reader(f)
#     next(reader)
#     for row in reader:
#         names[int(row[0])] = row[5].split(";")[0]


# def visualize_result(data, pred, cfg):
#     (img, info) = data

#     # print predictions in descending order
#     pred = np.int32(pred)
#     pixs = pred.size
#     uniques, counts = np.unique(pred, return_counts=True)
#     print("Predictions in [{}]:".format(info))
#     for idx in np.argsort(counts)[::-1]:
#         class_id = uniques[idx]
#         name = names.get(class_id, "Unknown")
#         ratio = counts[idx] / pixs * 100
#         if ratio > 0.1:
#             print("  {}: {:.2f}%".format(name, ratio))

#     # colorize prediction
#     pred_color = colorEncode(pred,  new_colors).astype(np.uint8)

#     # aggregate images and save
#     im_vis = np.concatenate((img, pred_color), axis=1)

#     img_name = info.split('/')[-1]
#     Image.fromarray(im_vis).save(
#         os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))


# def test(segmentation_module, loader, gpu):
#     segmentation_module.eval()

#     pbar = tqdm(total=len(loader))
#     for batch_data in loader:
#         # process data
#         batch_data = batch_data[0]
#         segSize = (batch_data['img_ori'].shape[0],
#                    batch_data['img_ori'].shape[1])
#         img_resized_list = batch_data['img_data']

#         with torch.no_grad():
#             scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
#             scores = async_copy_to(scores, gpu)

#             for img in img_resized_list:
#                 feed_dict = batch_data.copy()
#                 feed_dict['img_data'] = img
#                 del feed_dict['img_ori']
#                 del feed_dict['info']
#                 feed_dict = async_copy_to(feed_dict, gpu)

#                 # forward pass
#                 pred_tmp = segmentation_module(feed_dict, segSize=segSize)
#                 scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

#             _, pred = torch.max(scores, dim=1)
#             pred = as_numpy(pred.squeeze(0).cpu())

#         # visualization
#         visualize_result(
#             (batch_data['img_ori'], batch_data['info']),
#             pred,
#             cfg
#         )

#         pbar.update(1)


# def main(cfg, gpu):
#     torch.cuda.set_device(gpu)

#     # Network Builders
#     net_encoder = ModelBuilder.build_encoder(
#         arch=cfg.MODEL.arch_encoder,
#         fc_dim=cfg.MODEL.fc_dim,
#         weights=cfg.MODEL.weights_encoder)
#     net_decoder = ModelBuilder.build_decoder(
#         arch=cfg.MODEL.arch_decoder,
#         fc_dim=cfg.MODEL.fc_dim,
#         num_class=cfg.DATASET.num_class,
#         weights=cfg.MODEL.weights_decoder,
#         use_softmax=True)

#     crit = nn.NLLLoss(ignore_index=-1)

#     segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

#     # Dataset and Loader
#     dataset_test = TestDataset(
#         cfg.list_test,
#         cfg.DATASET)
#     loader_test = torch.utils.data.DataLoader(
#         dataset_test,
#         batch_size=cfg.TEST.batch_size,
#         shuffle=False,
#         collate_fn=user_scattered_collate,
#         num_workers=5,
#         drop_last=True)

#     segmentation_module.cuda()

#     # Main loop
#     test(segmentation_module, loader_test, gpu)

#     print('Inference done!')


# if __name__ == '__main__':
#     assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
#         'PyTorch>=0.4.0 is required'

#     parser = argparse.ArgumentParser(
#         description="PyTorch Semantic Segmentation Testing"
#     )
#     parser.add_argument(
#         "--imgs",
#         required=True,
#         type=str,
#         help="an image path, or a directory name"
#     )
#     parser.add_argument(
#         "--cfg",
#         default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
#         metavar="FILE",
#         help="path to config file",
#         type=str,
#     )
#     parser.add_argument(
#         "--gpu",
#         default=0,
#         type=int,
#         help="gpu id for evaluation"
#     )
#     parser.add_argument(
#         "opts",
#         help="Modify config options using the command-line",
#         default=None,
#         nargs=argparse.REMAINDER,
#     )
#     args = parser.parse_args()

#     cfg.merge_from_file(args.cfg)
#     cfg.merge_from_list(args.opts)
#     # cfg.freeze()

#     logger = setup_logger(distributed_rank=0)   # TODO
#     logger.info("Loaded configuration file {}".format(args.cfg))
#     logger.info("Running with config:\n{}".format(cfg))

#     cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
#     cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

#     # absolute paths of model weights
#     cfg.MODEL.weights_encoder = os.path.join(
#         cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
#     cfg.MODEL.weights_decoder = os.path.join(
#         cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

#     assert os.path.exists(cfg.MODEL.weights_encoder) and \
#         os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

#     # generate testing image list
#     if os.path.isdir(args.imgs):
#         imgs = find_recursive(args.imgs)
#     else:
#         imgs = [args.imgs]
#     assert len(imgs), "imgs should be a path to image (.jpg) or directory."
#     cfg.list_test = [{'fpath_img': x} for x in imgs]

#     # Make result directory with corresponding model name
#     cfg.TEST.result = os.path.join(
#         cfg.TEST.result,
#         'result_' + args.cfg.split('/')[-1][:-5] + '_' + os.path.basename(args.imgs))

#     if not os.path.isdir(cfg.TEST.result):
#         os.makedirs(cfg.TEST.result)

#     main(cfg, args.gpu)


# System libs
import os
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
from scipy.io import loadmat
import csv
# Our libs
from mit_semseg.dataset import TestDataset
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode, find_recursive, setup_logger
from mit_semseg.lib.nn import user_scattered_collate, async_copy_to
from mit_semseg.lib.utils import as_numpy
from PIL import Image
from tqdm import tqdm
from mit_semseg.config import cfg

colors = loadmat('data/color_29.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]


def visualize_result(data, pred, cfg):
    (img, info) = data

    # print predictions in descending order
    pred = np.int32(pred)
    pixs = pred.size
    uniques, counts = np.unique(pred, return_counts=True)
    print("Predictions in [{}]:".format(info))
    for idx in np.argsort(counts)[::-1]:
        name = names[uniques[idx] + 1]
        ratio = counts[idx] / pixs * 100
        if ratio > 0.1:
            print("  {}: {:.2f}%".format(name, ratio))

    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(np.uint8)

    # aggregate images and save
    im_vis = np.concatenate((img, pred_color), axis=1)

    img_name = info.split('/')[-1]
    Image.fromarray(im_vis).save(
        os.path.join(cfg.TEST.result, img_name.replace('.jpg', '.png')))


def test(segmentation_module, loader, gpu):
    segmentation_module.eval()

    pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        batch_data = batch_data[0]
        segSize = (batch_data['img_ori'].shape[0],
                   batch_data['img_ori'].shape[1])
        img_resized_list = batch_data['img_data']

        with torch.no_grad():
            scores = torch.zeros(1, cfg.DATASET.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, gpu)

            for img in img_resized_list:
                feed_dict = batch_data.copy()
                feed_dict['img_data'] = img
                del feed_dict['img_ori']
                del feed_dict['info']
                feed_dict = async_copy_to(feed_dict, gpu)

                # forward pass
                pred_tmp = segmentation_module(feed_dict, segSize=segSize)
                scores = scores + pred_tmp / len(cfg.DATASET.imgSizes)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        # visualization
        visualize_result(
            (batch_data['img_ori'], batch_data['info']),
            pred,
            cfg
        )

        pbar.update(1)


def main(cfg, gpu):
    torch.cuda.set_device(gpu)

    # Network Builders
    net_encoder = ModelBuilder.build_encoder(
        arch=cfg.MODEL.arch_encoder,
        fc_dim=cfg.MODEL.fc_dim,
        weights=cfg.MODEL.weights_encoder)
    net_decoder = ModelBuilder.build_decoder(
        arch=cfg.MODEL.arch_decoder,
        fc_dim=cfg.MODEL.fc_dim,
        num_class=cfg.DATASET.num_class,
        weights=cfg.MODEL.weights_decoder,
        use_softmax=True)

    crit = nn.NLLLoss(ignore_index=-1)

    segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)

    # Dataset and Loader
    dataset_test = TestDataset(
        cfg.list_test,
        cfg.DATASET)
    loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=cfg.TEST.batch_size,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    segmentation_module.cuda()

    # Main loop
    test(segmentation_module, loader_test, gpu)

    print('Inference done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    parser = argparse.ArgumentParser(
        description="PyTorch Semantic Segmentation Testing"
    )
    parser.add_argument(
        "--imgs",
        required=True,
        type=str,
        help="an image path, or a directory name"
    )
    parser.add_argument(
        "--cfg",
        default="config/ade20k-resnet50dilated-ppm_deepsup.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        default=0,
        type=int,
        help="gpu id for evaluation"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    logger = setup_logger(distributed_rank=0)   # TODO
    logger.info("Loaded configuration file {}".format(args.cfg))
    logger.info("Running with config:\n{}".format(cfg))

    cfg.MODEL.arch_encoder = cfg.MODEL.arch_encoder.lower()
    cfg.MODEL.arch_decoder = cfg.MODEL.arch_decoder.lower()

    # absolute paths of model weights
    cfg.MODEL.weights_encoder = os.path.join(
        cfg.DIR, 'encoder_' + cfg.TEST.checkpoint)
    cfg.MODEL.weights_decoder = os.path.join(
        cfg.DIR, 'decoder_' + cfg.TEST.checkpoint)

    assert os.path.exists(cfg.MODEL.weights_encoder) and \
        os.path.exists(cfg.MODEL.weights_decoder), "checkpoint does not exitst!"

    # generate testing image list
    if os.path.isdir(args.imgs):
        imgs = find_recursive(args.imgs)
    else:
        imgs = [args.imgs]
    assert len(imgs), "imgs should be a path to image (.jpg) or directory."
    cfg.list_test = [{'fpath_img': x} for x in imgs]

    if not os.path.isdir(cfg.TEST.result):
        os.makedirs(cfg.TEST.result)

    main(cfg, args.gpu)
