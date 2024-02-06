import os
import random
import time

import cv2 as cv
import face_recognition

# local modules
# from utils.video import create_capture
from utils.util1 import recognition, updateFaceDatabase
from utils.common import clock, draw_str
# from faceRecognition import FaceEval
import numpy as np
from PIL import Image, ImageDraw
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


# def main():
#     import sys, getopt
#
#     args, video_src = getopt.getopt(sys.argv[1:], '', ['cascade=', 'nested-cascade='])
#     try:
#         video_src = video_src[0]
#     except:
#         video_src = 0
#     args = dict(args)
#     # 检测人脸
#     cascade_fn = args.get('--cascade', "./model_detect/haarcascade_frontalface_alt.xml")
#     # 检测眼睛
#     nested_fn = args.get('--nested-cascade', "./model_detect/haarcascade_eye.xml")
#
#     cascade = cv.CascadeClassifier(cv.samples.findFile(cascade_fn))
#     nested = cv.CascadeClassifier(cv.samples.findFile(nested_fn))
#
#     cam = create_capture(video_src,
#                          fallback='synth:bg={}:noise=0.05'.format(cv.samples.findFile('./model_detect/lena.jpg')))
#     # test = FaceEval()
#     # test.update_face_data()
#     knownData = updateFaceDatabase('imgs')
#     # 这里是识别的单张图片，如果需要添加其他图片就可以在对应文件夹里增加，然后修改一下路径就可以了
#     for imgName in os.listdir('pic'):
#
#         img = cv.imread(f'pic/{imgName}')
#         try:
#             gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#         except:
#             continue
#         gray = cv.equalizeHist(gray)
#         # print(gray)
#         rects = detect(gray, cascade)
#         # print(rects)
#         vis = img.copy()
#         # 这里是人脸识别的入口文件，这个main函数是入口函数
#         draw_rects(vis, rects, (0, 255, 0))
#         for i, (x1, y1, x2, y2) in enumerate(rects):
#             face_img = vis[y1:y2, x1:x2]
#             if face_img.size == 0:
#                 print('no face detected')
#             resized_face_img = cv.resize(face_img, (112, 112), interpolation=cv.INTER_AREA)
#             t = random.randint(1000, 9999999999999)
#             cv.imwrite(f'img_data/{t}.bmp', resized_face_img)
#             try:
#                 # names = test.recognition(resized_face_img)
#                 # draw_str(vis, (x1 + 20, y1 - 20), names[0])
#                 name = recognition(f'img_data/{t}.bmp',knownData,0.6)
#                 draw_str(vis, (x1 + 20, y1 - 20), name)
#                 cv.imwrite(f'result-{imgName}.jpg', vis)
#             except IndexError as e:
#                 print(e)
#                 break
#     # while True:
#     #     _ret, img = cam.read()
#     #     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     #     gray = cv.equalizeHist(gray)
#     #
#     #     t = clock()
#     #     rects = detect(gray, cascade)
#     #     print(rects)  # 裁剪并保存人脸
#     #     vis = img.copy()
#     #     for i, (x1, y1, x2, y2) in enumerate(rects):
#     #         # 根据坐标裁剪人脸
#     #         face_img = vis[y1:y2, x1:x2]
#     #         # 检查裁剪的图像是否为空
#     #         if face_img.size == 0:
#     #             continue
#     #         # 保存人脸图片
#     #         # 调整图像大小为112x112
#     #         resized_face_img = cv.resize(face_img, (112, 112), interpolation=cv.INTER_AREA)
#     #
#     #         # 保存调整大小的人脸图片
#     #         cv.imwrite(f'img_data/face_{i}.jpg', resized_face_img)
#     #         img_ = cv.imdecode(np.fromfile(f'img_data/face_{i}.jpg', dtype=np.uint8), -1)
#     #         try:
#     #             names = test.recognition(img_)
#     #         except IndexError as e:
#     #             print(e)
#     #             break
#     #         draw_str(vis, (x1 + 20, y1 - 20), names[0])
#     #
#     #     draw_rects(vis, rects, (0, 255, 0))
#     #     # 是否标注眼睛
#     #     # if not nested.empty():
#     #     #     for x1, y1, x2, y2 in rects:
#     #     #         roi = gray[y1:y2, x1:x2]
#     #     #         vis_roi = vis[y1:y2, x1:x2]
#     #     #         subrects = detect(roi.copy(), nested)
#     #     #         draw_rects(vis_roi, subrects, (255, 0, 0))
#     #     dt = clock() - t
#     #
#     #     draw_str(vis, (20, 20), 'time: %.1f ms' % (dt * 1000))
#     #     cv.imshow('facedetect', vis)
#     #
#     #     if cv.waitKey(5) == 27:
#     #         break
#
#     print('Done')

def main(data):
    for imgName in os.listdir('pic'):
        unknown_image = cv.imread(f'pic/{imgName}')
        # Find all the faces and face encodings in the unknown image
        face_locations = face_recognition.face_locations(unknown_image)
        face_encodings = face_recognition.face_encodings(unknown_image)

        # Convert the image to a PIL-format image so that we can draw on top of it with the Pillow library
        # See http://pillow.readthedocs.io/ for more about PIL/Pillow
        img = cv.imread(f'pic/{imgName}')
        # Create a Pillow ImageDraw Draw instance to draw with
        vis = img.copy()

        # Loop through each face found in the unknown image
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # See if the face is a match for the known face(s)
            # matches = face_recognition.compare_faces([_[1] for _ in data], face_encoding)
            #
            # name = "Unknown"
            #
            # # If a match was found in known_face_encodings, just use the first one.
            # # if True in matches:
            # #     first_match_index = matches.index(True)
            # #     name = known_face_names[first_match_index]
            #
            # # Or instead, use the known face with the smallest distance to the new face
            # face_distances = face_recognition.face_distance([_[1] for _ in data], face_encoding)
            # best_match_index = np.argmin(face_distances)
            # if matches[best_match_index] and :
            #     name = data[best_match_index][0].split('.')[0]

            name = recognition(f'pic/{imgName}', data, 0.45)

            # Draw a box around the face using the Pillow module
            # draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

            # Draw a label with a name below the face
            text_width, text_height = 20, 20

            draw_str(vis, (left + text_width, top - text_height), name)
            draw_rects(vis, [(left, top, right, bottom)], (0, 255, 0))
        cv.imwrite(f'{imgName}.jpg', vis)
        # Display the resulting image
        # pil_image.show()

        # You can also save a copy of the new image to disk if you want by uncommenting this line
        # pil_image.save("image_with_boxes.jpg")


if __name__ == '__main__':
    data_ = updateFaceDatabase('imgs')
    main(data_)
    # cv.destroyAllWindows()
