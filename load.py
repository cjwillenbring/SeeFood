import os
import sys
import shutil
import numpy as np


def get_folders(path):
    return os.listdir(path)


def split_folder(path):
    files = os.listdir(path)
    indices = np.random.permutation(len(files))
    num_files = len(files)
    train_end = int(num_files * 0.6)
    val_end = train_end + int(num_files * 0.2)
    train_files = [files[idx] for idx in indices[:train_end]]
    val_files = [files[idx] for idx in indices[train_end:val_end]]
    test_files = [files[idx] for idx in indices[val_end:]]
    return train_files, val_files, test_files


def transfer(src, dest, file_names):
    os.mkdir(dest)
    for name in file_names:
        shutil.copy(os.path.join(src, name), os.path.join(dest, name))



def transfer_folder(src, dest, folder_name):
    """

    :param folder_name: the name of the folder "i.e. apple_pie"
    :return: None
    """
    fq_src = os.path.join(src, folder_name)
    train, val, test = split_folder(fq_src)
    transfer(fq_src, os.path.join(dest, 'train', folder_name), train)
    transfer(fq_src, os.path.join(dest, 'val', folder_name), val)
    transfer(fq_src, os.path.join(dest, 'test', folder_name), test)


def shuffle_images(src, dest):
    os.mkdir(os.path.join(dest, 'train'))
    os.mkdir(os.path.join(dest, 'val'))
    os.mkdir(os.path.join(dest, 'test'))
    folders = get_folders(src)
    for folder in folders:
        transfer_folder(src, dest, folder)


if __name__ == '__main__':
    shuffle_images(sys.argv[1], sys.argv[2])
