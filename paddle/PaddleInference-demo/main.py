import random
import time

import cv2 as cv

# local modules
from utils.video import create_capture
from utils.common import clock, draw_str
from faceRecognition import FaceEval
import numpy as np
from PIL import Image
from utils.preprocess import resizeImg


def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(15, 15),
                                     flags=cv.CASCADE_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:, 2:] += rects[:, :2]
    return rects


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)


def main():
    import sys, getopt

    args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
    try:
        video_src = video_src[0]
    except:
        video_src = 0
    args = dict(args)
    # 检测人脸
    cascade_fn = args.get('--cascade', "./model_detect/haarcascade_frontalface_alt.xml")
    # 检测眼睛
    nested_fn = args.get('--nested-cascade', "./model_detect/haarcascade_eye.xml")

    cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
    nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))

    cam = create_capture(video_src,
                         fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('./model_detect/lena.jpg')))
    test = FaceEval()
    test.update_face_data()
    # 这里是识别的单张图片，如果需要添加其他图片就可以在对应文件夹里增加，然后修改一下路径就可以了
    img = cv.imread('pic/img.png')
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.equalizeHist(gray)
    print(gray)
    rects = detect(gray, cascade)
    print(rects)
    vis = img.copy()
    # 这里是人脸识别的入口文件，这个main函数是入口函数
    draw_rects(vis, rects, (0, 255, 0))
    for i, (x1, y1, x2, y2) in enumerate(rects):
        face_img = vis[y1:y2, x1:x2]
        if face_img.size == 0:
            print('no face detected')
        resized_face_img = cv.resize(face_img, (112, 112), interpolation=cv.INTER_AREA)
        t = random.randint(1000, 9999999999999)
        cv.imwrite(f'img_data/{t}.bmp', resized_face_img)
        try:
            names = test.recognition(resized_face_img)
            draw_str(vis, (x1 + 20, y1 - 20), names[0])
            cv.imwrite('result1.jpg', vis)
        except IndexError as e:
            print(e)
            break
    # while True:
    #     _ret, img = cam.read()
    #     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     gray = cv.equalizeHist(gray)
    #
    #     t = clock()
    #     rects = detect(gray, cascade)
    #     print(rects)  # 裁剪并保存人脸
    #     vis = img.copy()
    #     for i, (x1, y1, x2, y2) in enumerate(rects):
    #         # 根据坐标裁剪人脸
    #         face_img = vis[y1:y2, x1:x2]
    #         # 检查裁剪的图像是否为空
    #         if face_img.size == 0:
    #             continue
    #         # 保存人脸图片
    #         # 调整图像大小为112x112
    #         resized_face_img = cv.resize(face_img, (112, 112), interpolation=cv.INTER_AREA)
    #
    #         # 保存调整大小的人脸图片
    #         cv.imwrite(f'img_data/face_{i}.jpg', resized_face_img)
    #         img_ = cv.imdecode(np.fromfile(f'img_data/face_{i}.jpg', dtype=np.uint8), -1)
    #         try:
    #             names = test.recognition(img_)
    #         except IndexError as e:
    #             print(e)
    #             break
    #         draw_str(vis, (x1 + 20, y1 - 20), names[0])
    #
    #     draw_rects(vis, rects, (0, 255, 0))
    #     # 是否标注眼睛
    #     # if not nested.empty():
    #     #     for x1, y1, x2, y2 in rects:
    #     #         roi = gray[y1:y2, x1:x2]
    #     #         vis_roi = vis[y1:y2, x1:x2]
    #     #         subrects = detect(roi.copy(), nested)
    #     #         draw_rects(vis_roi, subrects, (255, 0, 0))
    #     dt = clock() - t
    #
    #     draw_str(vis, (20, 20), 'time: %.1f ms' % (dt * 1000))
    #     cv.imshow('facedetect', vis)
    #
    #     if cv.waitKey(5) == 27:
    #         break

    print('Done')


if __name__ == '__main__':
    main()
    # cv.destroyAllWindows()
