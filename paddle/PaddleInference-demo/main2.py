import base64
import time
import urllib
import cv2 as cv
import os

import face_recognition
import requests
import json
from utils.common import clock, draw_str


def draw_rects(img, rects, color):
    for x1, y1, x2, y2 in rects:
        cv.rectangle(img, (x1, y1), (x2, y2), color, 2)


def face_recognition_func(path: str) -> dict:
    url = ("https://aip.baidubce.com/rest/2.0/face/v3/search?access_token=24.1280653a60902ca24dc88e91c401ab28.2592000"
           ".1709839676.282335-49977564")

    # image 可以通过 get_file_content_as_base64("C:\fakepath\Grand Theft Auto V 2023_7_23 20_04_42.png",False) 方法获取
    payload = json.dumps({
        "group_id_list": "24345354",
        "image": get_file_content_as_base64(path),
        "image_type": "BASE64"
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    print(response.text)
    if response.status_code == 200:
        content = response.json()
        if content['error_code'] == 0:
            return content['result']['user_list'][0]
        elif content['error_code'] == 18:
            time.sleep(0.3)
            return face_recognition_func(path)
    return {}


def getName(user_id: str) -> str:
    with open(r'name.json', 'r', encoding='utf-8') as fp:
        data = json.load(fp)
        return data.get(user_id)


def main():
    for imgName in os.listdir('pic'):
        unknown_image = cv.imread(f'pic/{imgName}')
        # Find all the faces and face encodings in the unknown image
        face_locations = face_recognition.face_locations(unknown_image)
        vis = unknown_image.copy()
        # Draw a label with a name below the face
        text_width, text_height = 20, 20
        # Loop through each face found in the unknown image
        print(face_locations)
        for _, (top, right, bottom, left) in enumerate(face_locations):
            face_img = vis[top:bottom, left:right]
            path = f'img_data/{_}_{imgName}'
            cv.imwrite(path, face_img)
            result = face_recognition_func(path)
            if result:
                draw_str(vis, (left + text_width, top - text_height), f'{getName(result["user_id"])}-{result["score"]}' if result['score'] > 80 else 'unknown')
            draw_rects(vis, [(left, top, right, bottom)], (0, 255, 0))
        cv.imwrite(f'{imgName}.jpg', vis)


def get_file_content_as_base64(path, urlencoded=False):
    """
    获取文件base64编码
    :param path: 文件路径
    :param urlencoded: 是否对结果进行urlencoded
    :return: base64编码信息
    """
    with open(path, "rb") as f:
        content = base64.b64encode(f.read()).decode("utf8")
        if urlencoded:
            content = urllib.parse.quote_plus(content)
    return content


if __name__ == '__main__':
    main()
