import os
import shutil
import subprocess


def resizeImg(basePath):
    if not os.path.exists(os.path.join(basePath, 'processed')):
        os.makedirs(os.path.join(basePath, 'processed'))
    for item in os.listdir(basePath):
        # 指定你的exe文件路径
        exe_path = r"E:\OpenFace_2.2.0_win_x64\FaceLandmarkImg.exe" + ' -f ' + os.path.join(basePath,item)
        # 使用subprocess.run来运行exe文件
        # 注意：如果你的exe文件需要管理员权限，这样的运行方式可能会失败
        result = subprocess.run(exe_path, capture_output=True, text=True)

        for item_ in os.listdir('./processed'):
            name = item_.split('.')[0].strip('_aligned')
            if '.' in item_:
                os.remove('./processed/' + item_)
                continue
            for img in os.listdir('./processed/' + item_):
                # 文件的新名字，这里使用了原文件名，你可以根据需要修改

                # 源文件的完整路径
                source_file = os.path.join(f'./processed/{item_}', img)

                # 目标文件的完整路径
                target_file = os.path.join(basePath,'processed',name+'.bmp')

                # 复制并重命名文件
                shutil.copy(source_file, target_file)
    if os.path.exists('./processed'):
        shutil.rmtree('./processed')
