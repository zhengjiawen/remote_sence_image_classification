import matplotlib.pyplot as plt
import os
import operator
import numpy as np
from clsName2id import getIdByName

train_root = 'D:/rssrai2019_scene_classification/train/'
val_root =  'D:/rssrai2019_scene_classification/val/'

def sortDictByKey(adict):
    return np.array(sorted(adict.items(), key=operator.itemgetter(0), reverse=False))

def data_hist(root, save_path='dataset_contribution.jpg'):
    total = 0
    idAndNumber = {}

    image_folders = list(map(lambda x: root + x, os.listdir(root)))
    for folder in image_folders:

        count = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        total+=count
        file_name = folder.split("/")[-1]
        idAndNumber[int(getIdByName(file_name))] = int(count)
    idAndNumber = sortDictByKey(idAndNumber)
    print(idAndNumber)
    print(total)

    plt.figure(figsize=(12,12))
    plt.title('dataset class')
    plt.bar(idAndNumber[:,0], idAndNumber[:,1])
    plt.xlabel('class')
    plt.ylabel('number')
    plt.savefig(save_path,format='jpg')
    plt.show()

if __name__ == '__main__':
    data_hist(train_root, 'train_data_hist.jpg')
    data_hist(val_root, 'val_data_hist.jpg')