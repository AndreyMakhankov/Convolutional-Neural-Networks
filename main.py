import gdown
import zipfile
import os
from PIL import Image
import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

if __name__ == '__main__':
    gdown.download('https://storage.yandexcloud.net/aiueducation/Content/base/l5/middle_fmr.zip', None, quiet=True)
    with zipfile.ZipFile('middle_fmr.zip', 'r') as zip_ref:
        zip_ref.extractall('content')
    imagePath = 'content'
    classList = sorted(os.listdir(imagePath))
    classCount = len(classList)
    fig, axs = plt.subplots(1, classCount, figsize=(25, 5))
    for i in range(classCount):
        car_path = f'{imagePath}/{classList[i]}/'
        img_path = car_path + random.choice(os.listdir(car_path))
        axs[i].set_title(classList[i])
        axs[i].imshow(Image.open(img_path))
        axs[i].axis('off')
    plt.show()