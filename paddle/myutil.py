import os


for index,item in enumerate(os.listdir('FaceDatabase')[:10]):
    root = 'FaceDatabase'
    os.rename(os.path.join(root,item),os.path.join(root,'lxz'+str(index+1)+'.bmp'))