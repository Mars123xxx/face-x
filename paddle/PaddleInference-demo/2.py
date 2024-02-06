import os

from faceRecognition import FaceEval
import numpy as np
from PIL import Image
import cv2
test = FaceEval()
test.update_face_data()

# image_path = '3.bmp'
# img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
# names = test.recognition(img)
# print("name",names)
for item in os.listdir(r'D:\projects\face.evoLVe\paddle\data\imgs\3'):
    image_path = 'D:\\projects\\face.evoLVe\\paddle\\data\\imgs\\3\\' + item
    # image_path = 'jin1.jpg'
    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
    names = test.recognition(img)
    # cv2.imwrite('result.jpg',img)
    print(f'{item}结果是',names)
