# from operator import imod
import mmcv
import glob
import shutil


imgs = glob.glob("/root/yjzprivate/mmaction2-master/data/CEUS400/**/*0001.jpg",recursive=True)
print(len(imgs))
# imgs_shape = set([mmcv.imread(i).shape for i in imgs])
print(imgs[243])
# for index, img in enumerate(imgs):
    # shutil.copy(img, 'img_ceus/'+str(index)+'.jpg')