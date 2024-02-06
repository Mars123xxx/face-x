import os

import face_recognition

from utils import updateFaceDatabase





if __name__ == '__main__':
    knownData = updateFaceDatabase('imgs')
    print(main(knownData, tolerance=0.45))
