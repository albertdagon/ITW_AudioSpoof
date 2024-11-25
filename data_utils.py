import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=True, target_only=False, enroll_spk=None, enroll=False, is_itw=False):
    d_meta = {}
    utt_list = []
    tag_list = []
    utt2spk = {}

    # with open(dir_meta, "r") as f:
    #     l_meta = f.readlines()

    # if is_train:
    #     for line in l_meta:
    #         _, key, _, _, label = line.strip().split(" ")
    #         file_list.append(key)
    #         d_meta[key] = 1 if label == "bonafide" else 0
    #     return d_meta, file_list

    # elif is_eval:
    #     for line in l_meta:
    #         if is_itw:
    #             key, _, _ = line.strip().split(" ")
    #         else:
    #             _, key, _, _, _ = line.strip().split(" ")
    #         # key = line.strip()
    #         file_list.append(key)
    #     return file_list
    # else:
    #     for line in l_meta:
    #         _, key, _, _, label = line.strip().split(" ")
    #         file_list.append(key)
    #         d_meta[key] = 1 if label == "bonafide" else 0
    #     return d_meta, file_list
    if not enroll and is_train:  # read train
        with open(dir_meta, "r") as f:
            l_meta = f.readlines()

        for line in l_meta:
            spk, key, _, tag, label = line.strip().split(" ")

            if key in utt2spk:
                print("Duplicated utt error", key)

            # utt2spk[key] = int(spk[-4:])
            utt2spk[key] = spk
            tag_list.append(tag)
            utt_list.append(key)
            d_meta[key] = 1 if label != "bonafide" else 0  # bona: 0 spoof: 1
    elif not enroll and not is_train:  # read dev and eval
        with open(dir_meta, "r") as f:
            l_meta = f.readlines()

        for line in l_meta:
            if is_itw:
                key, spk, label = line.strip().split(" ")
                tag = "-"
            else:
                spk, key, _, tag, label = line.strip().split(" ")
            # spk = int(spk[-4:])

            if not target_only or spk in enroll_spk:  # ensure target only speakers
                if key in utt2spk:
                    print("Duplicated utt error", key)

                utt2spk[key] = spk
                utt_list.append(key)
                tag_list.append(tag)
                d_meta[key] = 1 if label != "bonafide" else 0
    else:  # read in enroll data
        for dir in dir_meta:
            with open(dir, "r") as f:
                l_meta = f.readlines()

            for line in l_meta:
                tmp = line.strip().split(" ")

                spk = tmp[0]
                keys = tmp[1].split(",")

                for key in keys:
                    if key in utt2spk:
                        print("Duplicated utt error", key)

                    # utt2spk[key] = int(spk[-4:])
                    utt2spk[key] = spk
                    utt_list.append(key)
                    d_meta[key] = 0
                    tag_list.append("-")
        # print(utt2spk)

    return d_meta, utt_list, utt2spk, tag_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASV(Dataset):
    def __init__(self, list_IDs, labels, base_dir, utt2spk, tag_list, train=True):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.utt2spk = utt2spk
        self.tag_list = tag_list
        self.cut = 64600  # take ~4 sec audio (64600 samples)
        self.train = train

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        tag = self.tag_list[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        if self.train:
            X_pad = pad_random(X, self.cut)
        else:
            X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        spk = self.utt2spk[key]
        return x_inp, y, spk, key, tag


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key

class Dataset_in_the_wild_eval(Dataset):
    def __init__(self, list_IDs, base_dir):
        '''self.list_IDs	: list of strings (each string: utt key),
               '''
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir) + f"/release_in_the_wild/{key}")
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key