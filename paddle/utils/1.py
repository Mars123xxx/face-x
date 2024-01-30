import os
import subprocess

def resizeImg():
    for item in os.listdir(r'D:\projects\face.evoLVe\paddle\FaceDatabase'):
        # 指定你的exe文件路径
        exe_path = r"E:\OpenFace_2.2.0_win_x64\FaceLandmarkImg.exe" + ' -f ' + ("D:\\projects\\face.evoLVe\\paddle"
                                                                                "\\FaceDatabase\\")+f'{item}'
        # 使用subprocess.run来运行exe文件
        # 注意：如果你的exe文件需要管理员权限，这样的运行方式可能会失败
        result = subprocess.run(exe_path, capture_output=True, text=True)