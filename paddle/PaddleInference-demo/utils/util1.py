import os
import face_recognition

import numpy as np


def updateFaceDatabase(imgBaseRoot) -> list[tuple[str, np.array]]:
    listArr = []
    for class_ in os.listdir(imgBaseRoot):
        classImgPath = os.path.join(imgBaseRoot, class_)
        for imgPath in os.listdir(classImgPath):
            known_image = face_recognition.load_image_file(os.path.join(classImgPath, imgPath))
            listArr.append((imgPath,face_recognition.face_encodings(known_image)[0]))

    return listArr


def recognition(unknown_image_path:str,data: list, tolerance: float) -> str:
    print(1)
    unknown_img = face_recognition.load_image_file(unknown_image_path)
    try:
        unknown_encoding = face_recognition.face_encodings(unknown_img)[0]
    except IndexError:
        return 'unknown'
    distances = face_recognition.face_distance([_[1] for _ in data], unknown_encoding)
    resultIndex, name, resultDistance = None, 'unknown', float('inf')
    for i, distance in enumerate(distances):
        if resultDistance < distance or distance > tolerance:
            continue
        resultDistance = distance
        resultIndex = i
        name = data[resultIndex][0]
    # print(distances)
    if resultDistance > tolerance:
        print(distances)
        print('unknown image')
        return 'unknown'
    return name.split('.')[0]
