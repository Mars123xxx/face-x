import os


for index,item in enumerate(os.listdir('../data/imgs/1')):
    root = '../data/imgs/1'
    os.rename(os.path.join(root,item),os.path.join(root,str(index+1)+'.jpg'))