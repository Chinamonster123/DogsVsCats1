import random
import os
import shutil

def split_train_test(fileDir, tarDir):
    if not os.path.exists(tarDir):
        os.makedirs(tarDir)
    pathDir = os.listdir(fileDir)  # 取图片的原始路径
    filenumber = len(pathDir)
    tarnumber = len(tarDir)
    rate = 0.2  # 自定义抽取图片的比例，比方说100张抽10张，那就是0.1
    picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片
    print("=========开始移动图片============")
    for name in sample:
        shutil.move(fileDir + '/' + name, tarDir + '/' + name)
    print("=========移动图片完成============")


