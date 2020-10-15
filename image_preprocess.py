import cv2
import numpy as np
from tqdm import tqdm 

# source = '/home/VS00564377/work/second_model/workspace/training_demo/images/train/'
# dest = '/home/VS00564377/work/second_model/workspace/training_demo/images/preprocessed_train/'
source = '/home/VS00564377/work/second_model/workspace/training_demo/images/test/'
dest = '/home/VS00564377/work/second_model/workspace/training_demo/images/preprocessed_test/'

import glob
types = ('*.jpg', '*.png','*.JPG','*.jpeg','*.PNG','*.Jpg','*.Png') # the tuple of file types

files_grabbed = []
for files in types:
    files_grabbed.extend(glob.glob(source+files))


def preprocessing(img_file):
    '''
    converting to greyscale, 
    remove background design 
    '''
    img = cv2.imread(img_file, 0)
    ret,thr = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    backtorgb = cv2.cvtColor(thr,cv2.COLOR_GRAY2RGB)
    cv2.imwrite(dest + img_file.split('/')[-1] , backtorgb)
    

if __name__ == "__main__": 
    for name in tqdm(files_grabbed):
        preprocessing(name)
        print("Done !!!")
