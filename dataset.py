import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image


# class CatDogDataset(Dataset):
#     def __init__(self, file_list, dir, mode='train', transform=None):
#         self.file_list = file_list
#         self.dir = dir
#         self.mode = mode
#         self.transform = transform
#         if self.mode == 'train':
#             if 'dog' in self.file_list[0]:
#                 self.label = 1
#             else:
#                 self.label = 0
#
#     def __len__(self):
#         return len(self.file_list)
#
#     def __getitem__(self, idx):
#         img = Image.open(os.path.join(self.dir, self.file_list[idx]))
#         if self.transform:
#             img = self.transform(img)
#         if self.mode == 'train':
#             img = img.numpy()
#             return img.astype('float32'), self.label
#         else:
#             img = img.numpy()
#             return img.astype('float32'), self.file_list[idx]

class CatDogDataset(Dataset):
    def __init__(self, dir, is_test, transform):
        self.dir = dir
        self.transform = transform
        self.files = os.listdir(dir)
        self.is_test = is_test

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.files[idx]))
        if self.transform:
            img = self.transform(img)
        if self.is_test:
            img = img.numpy()
            return img.astype('float32')
        else: # train / val
            img = img.numpy()
            label = 1 if 'dog' in self.file[idx] else 0
            return img.astype('float32'), label

