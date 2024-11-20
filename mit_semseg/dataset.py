# import os
# import json
# import torch
# from torchvision import transforms
# import numpy as np
# from PIL import Image


# def imresize(im, size, interp='bilinear'):
#     if interp == 'nearest':
#         resample = Image.NEAREST
#     elif interp == 'bilinear':
#         resample = Image.BILINEAR
#     elif interp == 'bicubic':
#         resample = Image.BICUBIC
#     else:
#         raise Exception('resample method undefined!')

#     return im.resize(size, resample)


# class BaseDataset(torch.utils.data.Dataset):
#     def __init__(self, odgt, opt, **kwargs):
#         # parse options
#         self.imgSizes = opt.imgSizes
#         self.imgMaxSize = opt.imgMaxSize
#         # max down sampling rate of network to avoid rounding during conv or pooling
#         self.padding_constant = opt.padding_constant

#         # parse the input list
#         self.parse_input_list(odgt, **kwargs)
#         self.classMapping = {
#             1: 4,  # wall -> Wall
#             2: 3,  # building -> Building
#             3: 11,  # sky -> Sky
#             4: 25,  # floor -> ground
#             5: 9,  # tree -> Vegetation
#             6: 10,  # ceiling -> Terrain
#             7: 1,  # road -> Roads
#             8: 22,  # bed -> Other
#             9: 3,  # window -> building
#             10: 10,  # grass -> Terrain
#             11: 22,  # cabinet -> other
#             12: 2,  # sidewalk -> sidewalk
#             13: 12,  # person -> Pedestrian
#             14: 10,  # earth -> Terrain
#             15: 3,  # door -> building
#             16: 22,  # table -> other
#             17: 22,  # mountain -> other
#             18: 9,  # plant -> Vegetation
#             19: 22,  # curtain -> other
#             20: 21,  # chair -> dynamic
#             21: 14,  # car -> Car
#             22: 23,  # water -> Water
#             23: 22,  # painting -> other
#             24: 22,  # sofa -> other
#             25: 22,  # shelf -> other
#             26: 3,  # house -> building
#             27: 23,  # sea -> water
#             28: 22,  # mirror -> other
#             29: 22,  # rug -> other
#             30: 10,  # field -> Terrain
#             31: 20,  # armchair -> Static
#             32: 20,  # seat -> Static
#             33: 5,  # fence -> Fence
#             34: 20,  # desk -> Static
#             35: 3,  # rock -> Building
#             36: 20,  # wardrobe -> Static
#             37: 20,  # lamp -> Static
#             38: 23,  # bathtub -> Water
#             39: 5,  # railing -> Fence
#             40: 20,  # cushion -> Static
#             41: 20,  # base -> Static
#             42: 20,  # box -> Static
#             43: 5,  # column -> Fence
#             44: 10,  # signboard -> Terrain
#             45: 20,  # chest of drawers -> Static
#             46: 20,  # counter -> Static
#             47: 10,  # sand -> Terrain
#             48: 23,  # sink -> Water
#             49: 3,  # skyscraper -> Building
#             50: 23,  # fireplace -> Water
#             51: 23,  # refrigerator -> Water
#             52: 20,  # grandstand -> Static
#             53: 10,  # path -> Terrain
#             54: 20,  # stairs -> Static
#             55: 10,  # runway -> Terrain
#             56: 20,  # case -> Static
#             57: 20,  # pool table -> Static
#             58: 20,  # pillow -> Static
#             59: 20,  # screen door -> Static
#             60: 20,  # stairway -> Static
#             61: 23,  # river -> Water
#             62: 10,  # bridge -> Terrain
#             63: 20,  # bookcase -> Static
#             64: 20,  # blind -> Static
#             65: 20,  # coffee table -> Static
#             66: 20,  # toilet -> Static
#             67: 9,  # flower -> Vegetation
#             68: 20,  # book -> Static
#             69: 3,  # hill -> Building
#             70: 20,  # bench -> Static
#             71: 20,  # countertop -> Static
#             72: 20,  # stove -> Static
#             73: 9,  # palm -> Vegetation
#             74: 20,  # kitchen island -> Static
#             75: 20,  # computer -> Static
#             76: 20,  # swivel chair -> Static
#             77: 23,  # boat -> Water
#             78: 20,  # bar -> Static
#             79: 20,  # arcade machine -> Static
#             80: 20,  # hovel -> Static
#             81: 16,  # bus -> Static
#             82: 20,  # towel -> Static
#             83: 20,  # light -> Static
#             84: 15,  # truck -> Truck
#             85: 20,  # tower -> Static
#             86: 20,  # chandelier -> Static
#             87: 20,  # awning -> Static
#             88: 20,  # streetlight -> Static
#             89: 20,  # booth -> Static
#             90: 20,  # television -> Static
#             91: 20,  # airplane -> Static
#             92: 25,  # dirt track -> ground
#             93: 22,  # apparel -> other
#             94: 6,  # pole -> pole
#             95: 25,  # land -> ground
#             96: 22,  # bannister -> other
#             97: 22,  # escalator -> other
#             98: 22,  # ottoman -> other
#             99: 20,  # bottle -> Static
#             100: 22,  # buffet -> other
#             101: 20,  # poster -> Static
#             102: 22,  # stage -> other
#             103: 14,  # van -> car
#             104: 21,  # ship -> dynamic
#             105: 20,  # fountain -> static
#             106: 22,  # conveyer belt -> other
#             107: 22,  # canopy -> other
#             108: 22,  # washer -> other
#             109: 22,  # plaything -> other
#             110: 22,  # swimming pool -> Static
#             111: 22,  # stool -> Static
#             112: 21,  # barrel -> dynamic
#             113: 20,  # basket -> Static
#             114: 23,  # waterfall -> water
#             115: 20,  # tent -> Static
#             116: 21,  # bag -> dynamic
#             117: 19,  # minibike -> bike
#             118: 20,  # cradle -> Static
#             119: 20,  # oven -> other
#             120: 21,  # ball -> dynamic
#             121: 22,  # food -> other
#             122: 22,  # step -> other
#             123: 22,  # tank -> other
#             124: 22,  # trade name -> other
#             125: 22,  # microwave -> other
#             126: 22,  # pot -> Static
#             127: 21,  # animal -> dynamic
#             128: 19,  # bicycle -> bike
#             129: 23,  # lake -> water
#             130: 22,  # dishwasher -> other
#             131: 22,  # screen -> other
#             132: 22,  # blanket -> Static
#             133: 20,  # sculpture -> Static
#             134: 22,  # hood -> other
#             135: 22,  # sconce -> other
#             136: 22,  # vase -> other
#             137: 7,  # traffic light -> traffic light
#             138: 22,  # tray -> Static
#             139: 22,  # ashcan -> Static
#             140: 22,  # fan -> Static
#             141: 22,  # pier -> Static
#             142: 22,  # crt screen -> Static
#             143: 22,  # plate -> Static
#             144: 22,  # monitor -> Static
#             145: 22,  # bulletin board -> Static
#             146: 22,  # shower -> Static
#             147: 22,  # radiator -> Static
#             148: 22,  # glass -> Static
#             149: 22,  # clock -> Static
#             150: 20   # flag -> Static
#         }
#         # mean and std
#         self.normalize = transforms.Normalize(
#             mean=[0.485, 0.456, 0.406],
#             std=[0.229, 0.224, 0.225])

#     def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
#         if isinstance(odgt, list):
#             self.list_sample = odgt
#         elif isinstance(odgt, str):
#             self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

#         if max_sample > 0:
#             self.list_sample = self.list_sample[0:max_sample]
#         if start_idx >= 0 and end_idx >= 0:     # divide file list
#             self.list_sample = self.list_sample[start_idx:end_idx]

#         self.num_sample = len(self.list_sample)
#         assert self.num_sample > 0
#         print('# samples: {}'.format(self.num_sample))

#     def img_transform(self, img):
#         # 0-255 to 0-1
#         img = np.float32(np.array(img)) / 255.
#         img = img.transpose((2, 0, 1))
#         img = self.normalize(torch.from_numpy(img.copy()))
#         return img

#     def segm_transform(self, segm):
#         # remap values in segm according to classMapping
#         segm = np.array(segm)
#         # If it's a 3D array, assume the red channel is in the first channel
#         if len(segm.shape) == 3:
#             segm = segm[:, :, 0]  # Read from red channel

#     # Now process the 2D array as expected
#         mapped_segm = np.zeros(segm.shape, dtype=np.uint64)
#         for k, v in self.classMapping.items():
#             mapped_segm[segm == k] = v
#         mapped_segm = torch.from_numpy(mapped_segm).long()
#         return mapped_segm

#     # Round x to the nearest multiple of p and x' >= x
#     def round2nearest_multiple(self, x, p):
#         return ((x - 1) // p + 1) * p


# class TrainDataset(BaseDataset):
#     def __init__(self, root_dataset, odgt, opt, batch_per_gpu=1, **kwargs):
#         super(TrainDataset, self).__init__(odgt, opt, **kwargs)
#         self.root_dataset = root_dataset
#         # down sampling rate of segm labe
#         self.segm_downsampling_rate = opt.segm_downsampling_rate
#         self.batch_per_gpu = batch_per_gpu

#         # classify images into two classes: 1. h > w and 2. h <= w
#         self.batch_record_list = [[], []]

#         # override dataset length when trainig with batch_per_gpu > 1
#         self.cur_idx = 0
#         self.if_shuffled = False

#     def _get_sub_batch(self):
#         while True:
#             # get a sample record
#             this_sample = self.list_sample[self.cur_idx]
#             if this_sample['height'] > this_sample['width']:
#                 self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
#             else:
#                 self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

#             # update current sample pointer
#             self.cur_idx += 1
#             if self.cur_idx >= self.num_sample:
#                 self.cur_idx = 0
#                 np.random.shuffle(self.list_sample)

#             if len(self.batch_record_list[0]) == self.batch_per_gpu:
#                 batch_records = self.batch_record_list[0]
#                 self.batch_record_list[0] = []
#                 break
#             elif len(self.batch_record_list[1]) == self.batch_per_gpu:
#                 batch_records = self.batch_record_list[1]
#                 self.batch_record_list[1] = []
#                 break
#         return batch_records

#     def __getitem__(self, index):
#         # NOTE: random shuffle for the first time. shuffle in __init__ is useless
#         if not self.if_shuffled:
#             np.random.seed(index)
#             np.random.shuffle(self.list_sample)
#             self.if_shuffled = True

#         # get sub-batch candidates
#         batch_records = self._get_sub_batch()

#         # resize all images' short edges to the chosen size
#         if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
#             this_short_size = np.random.choice(self.imgSizes)
#         else:
#             this_short_size = self.imgSizes

#         # calculate the BATCH's height and width
#         # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
#         batch_widths = np.zeros(self.batch_per_gpu, np.int32)
#         batch_heights = np.zeros(self.batch_per_gpu, np.int32)
#         for i in range(self.batch_per_gpu):
#             img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
#             this_scale = min(
#                 this_short_size / min(img_height, img_width), \
#                 self.imgMaxSize / max(img_height, img_width))
#             batch_widths[i] = img_width * this_scale
#             batch_heights[i] = img_height * this_scale

#         # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
#         batch_width = np.max(batch_widths)
#         batch_height = np.max(batch_heights)
#         batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
#         batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

#         assert self.padding_constant >= self.segm_downsampling_rate, \
#             'padding constant must be equal or large than segm downsamping rate'
#         batch_images = torch.zeros(
#             self.batch_per_gpu, 3, batch_height, batch_width)
#         batch_segms = torch.zeros(
#             self.batch_per_gpu,
#             batch_height // self.segm_downsampling_rate,
#             batch_width // self.segm_downsampling_rate).long()

#         for i in range(self.batch_per_gpu):
#             this_record = batch_records[i]

#             # load image and label
#             image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
#             segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

#             img = Image.open(image_path).convert('RGB')
#             segm = Image.open(segm_path).convert('L')
#             assert(img.size[0] == segm.size[0])
#             assert(img.size[1] == segm.size[1])

#             # random_flip
#             if np.random.choice([0, 1]):
#                 img = img.transpose(Image.FLIP_LEFT_RIGHT)
#                 segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

#             # note that each sample within a mini batch has different scale param
#             img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
#             segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

#             # further downsample seg label, need to avoid seg label misalignment
#             segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
#             segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
#             segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
#             segm_rounded.paste(segm, (0, 0))
#             segm = imresize(
#                 segm_rounded,
#                 (segm_rounded.size[0] // self.segm_downsampling_rate, \
#                  segm_rounded.size[1] // self.segm_downsampling_rate), \
#                 interp='nearest')

#             # image transform, to torch float tensor 3xHxW
#             img = self.img_transform(img)

#             # segm transform, to torch long tensor HxW
#             segm = self.segm_transform(segm)

#             # put into batch arrays
#             batch_images[i][:, :img.shape[1], :img.shape[2]] = img
#             batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

#         output = dict()
#         output['img_data'] = batch_images
#         output['seg_label'] = batch_segms
#         return output

#     def __len__(self):
#         return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
#         #return self.num_sampleclass


# class ValDataset(BaseDataset):
#     def __init__(self, root_dataset, odgt, opt, **kwargs):
#         super(ValDataset, self).__init__(odgt, opt, **kwargs)
#         self.root_dataset = root_dataset

#     def __getitem__(self, index):
#         this_record = self.list_sample[index]
#         # load image and label
#         image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
#         segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
#         img = Image.open(image_path).convert('RGB')
#         segm = Image.open(segm_path).convert('L')
#         assert(img.size[0] == segm.size[0])
#         assert(img.size[1] == segm.size[1])

#         ori_width, ori_height = img.size

#         img_resized_list = []
#         for this_short_size in self.imgSizes:
#             # calculate target height and width
#             scale = min(this_short_size / float(min(ori_height, ori_width)),
#                         self.imgMaxSize / float(max(ori_height, ori_width)))
#             target_height, target_width = int(ori_height * scale), int(ori_width * scale)

#             # to avoid rounding in network
#             target_width = self.round2nearest_multiple(target_width, self.padding_constant)
#             target_height = self.round2nearest_multiple(target_height, self.padding_constant)

#             # resize images
#             img_resized = imresize(img, (target_width, target_height), interp='bilinear')

#             # image transform, to torch float tensor 3xHxW
#             img_resized = self.img_transform(img_resized)
#             img_resized = torch.unsqueeze(img_resized, 0)
#             img_resized_list.append(img_resized)

#         # segm transform, to torch long tensor HxW
#         segm = self.segm_transform(segm)
#         batch_segms = torch.unsqueeze(segm, 0)

#         output = dict()
#         output['img_ori'] = np.array(img)
#         output['img_data'] = [x.contiguous() for x in img_resized_list]
#         output['seg_label'] = batch_segms.contiguous()
#         output['info'] = this_record['fpath_img']
#         return output

#     def __len__(self):
#         return self.num_sample


# class TestDataset(BaseDataset):
#     def __init__(self, odgt, opt, **kwargs):
#         super(TestDataset, self).__init__(odgt, opt, **kwargs)

#     def __getitem__(self, index):
#         this_record = self.list_sample[index]
#         # load image
#         image_path = this_record['fpath_img']
#         img = Image.open(image_path).convert('RGB')

#         ori_width, ori_height = img.size

#         img_resized_list = []
#         for this_short_size in self.imgSizes:
#             # calculate target height and width
#             scale = min(this_short_size / float(min(ori_height, ori_width)),
#                         self.imgMaxSize / float(max(ori_height, ori_width)))
#             target_height, target_width = int(ori_height * scale), int(ori_width * scale)

#             # to avoid rounding in network
#             target_width = self.round2nearest_multiple(target_width, self.padding_constant)
#             target_height = self.round2nearest_multiple(target_height, self.padding_constant)

#             # resize images
#             img_resized = imresize(img, (target_width, target_height), interp='bilinear')

#             # image transform, to torch float tensor 3xHxW
#             img_resized = self.img_transform(img_resized)
#             img_resized = torch.unsqueeze(img_resized, 0)
#             img_resized_list.append(img_resized)

#         output = dict()
#         output['img_ori'] = np.array(img)
#         output['img_data'] = [x.contiguous() for x in img_resized_list]
#         output['info'] = this_record['fpath_img']
#         return output

#     def __len__(self):
#         return self.num_sample
import os
import json
import torch
from torchvision import transforms
import numpy as np
from PIL import Image


def imresize(im, size, interp='bilinear'):
    if interp == 'nearest':
        resample = Image.NEAREST
    elif interp == 'bilinear':
        resample = Image.BILINEAR
    elif interp == 'bicubic':
        resample = Image.BICUBIC
    else:
        raise Exception('resample method undefined!')

    return im.resize(size, resample)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, odgt, opt, **kwargs):
        # parse options
        self.imgSizes = opt.imgSizes
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # parse the input list
        self.parse_input_list(odgt, **kwargs)

        # mean and std
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def parse_input_list(self, odgt, max_sample=-1, start_idx=-1, end_idx=-1):
        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def img_transform(self, img):
        # 0-255 to 0-1
        img = np.float32(np.array(img)) / 255.
        img = img.transpose((2, 0, 1))
        img = self.normalize(torch.from_numpy(img.copy()))
        return img

    def segm_transform(self, segm):
        # to tensor, -1 to 149
        segm = torch.from_numpy(np.array(segm)).long() - 1
        return segm

    # Round x to the nearest multiple of p and x' >= x
    def round2nearest_multiple(self, x, p):
        return ((x - 1) // p + 1) * p


class TrainDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, batch_per_gpu=1, **kwargs):
        super(TrainDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0
        self.if_shuffled = False

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        if not self.if_shuffled:
            np.random.seed(index)
            np.random.shuffle(self.list_sample)
            self.if_shuffled = True

        # get sub-batch candidates
        batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSizes, list) or isinstance(self.imgSizes, tuple):
            this_short_size = np.random.choice(self.imgSizes)
        else:
            this_short_size = self.imgSizes

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        batch_widths = np.zeros(self.batch_per_gpu, np.int32)
        batch_heights = np.zeros(self.batch_per_gpu, np.int32)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(
                this_short_size / min(img_height, img_width), \
                self.imgMaxSize / max(img_height, img_width))
            batch_widths[i] = img_width * this_scale
            batch_heights[i] = img_height * this_scale

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_width = np.max(batch_widths)
        batch_height = np.max(batch_heights)
        batch_width = int(self.round2nearest_multiple(batch_width, self.padding_constant))
        batch_height = int(self.round2nearest_multiple(batch_height, self.padding_constant))

        assert self.padding_constant >= self.segm_downsampling_rate, \
            'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(
            self.batch_per_gpu, 3, batch_height, batch_width)
        batch_segms = torch.zeros(
            self.batch_per_gpu,
            batch_height // self.segm_downsampling_rate,
            batch_width // self.segm_downsampling_rate).long()

        for i in range(self.batch_per_gpu):
            this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])

            img = Image.open(image_path).convert('RGB')
            segm = Image.fromarray(np.array(Image.open(segm_path))[:,:,0])
            assert(segm.mode == "L")
            assert(img.size[0] == segm.size[0])
            assert(img.size[1] == segm.size[1])

            # random_flip
            if np.random.choice([0, 1]):
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                segm = segm.transpose(Image.FLIP_LEFT_RIGHT)

            # note that each sample within a mini batch has different scale param
            img = imresize(img, (batch_widths[i], batch_heights[i]), interp='bilinear')
            segm = imresize(segm, (batch_widths[i], batch_heights[i]), interp='nearest')

            # further downsample seg label, need to avoid seg label misalignment
            segm_rounded_width = self.round2nearest_multiple(segm.size[0], self.segm_downsampling_rate)
            segm_rounded_height = self.round2nearest_multiple(segm.size[1], self.segm_downsampling_rate)
            segm_rounded = Image.new('L', (segm_rounded_width, segm_rounded_height), 0)
            segm_rounded.paste(segm, (0, 0))
            segm = imresize(
                segm_rounded,
                (segm_rounded.size[0] // self.segm_downsampling_rate, \
                 segm_rounded.size[1] // self.segm_downsampling_rate), \
                interp='nearest')

            # image transform, to torch float tensor 3xHxW
            img = self.img_transform(img)

            # segm transform, to torch long tensor HxW
            segm = self.segm_transform(segm)

            # put into batch arrays
            batch_images[i][:, :img.shape[1], :img.shape[2]] = img
            batch_segms[i][:segm.shape[0], :segm.shape[1]] = segm

        output = dict()
        output['img_data'] = batch_images
        output['seg_label'] = batch_segms
        return output

    def __len__(self):
        return int(1e10) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class ValDataset(BaseDataset):
    def __init__(self, root_dataset, odgt, opt, **kwargs):
        super(ValDataset, self).__init__(odgt, opt, **kwargs)
        self.root_dataset = root_dataset

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, this_record['fpath_segm'])
        img = Image.open(image_path).convert('RGB')
        segm = Image.fromarray(np.array(Image.open(segm_path))[:,:,0])
        assert(segm.mode == "L")
        assert(img.size[0] == segm.size[0])
        assert(img.size[1] == segm.size[1])

        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        # segm transform, to torch long tensor HxW
        segm = self.segm_transform(segm)
        batch_segms = torch.unsqueeze(segm, 0)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample


class TestDataset(BaseDataset):
    def __init__(self, odgt, opt, **kwargs):
        super(TestDataset, self).__init__(odgt, opt, **kwargs)

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image
        image_path = this_record['fpath_img']
        img = Image.open(image_path).convert('RGB')

        ori_width, ori_height = img.size

        img_resized_list = []
        for this_short_size in self.imgSizes:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                        self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_width = self.round2nearest_multiple(target_width, self.padding_constant)
            target_height = self.round2nearest_multiple(target_height, self.padding_constant)

            # resize images
            img_resized = imresize(img, (target_width, target_height), interp='bilinear')

            # image transform, to torch float tensor 3xHxW
            img_resized = self.img_transform(img_resized)
            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        output = dict()
        output['img_ori'] = np.array(img)
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample