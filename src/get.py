import cv2, os
import numpy as np
import image
import sqlite3
from PIL import Image

def get_data(path, detector):
    # Lấy tất cả các file trong thư mục
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    ids=[]
    faceSamples=[]
    for imagePath in imagePaths:
        if (imagePath[-3:]=="jpg"):
            pilImage=Image.open(imagePath).convert('L')
            imageNp=np.array(pilImage,'uint8')
            id = int(os.path.split(imagePath)[-1].split("_")[0])
            faces=detector.detectMultiScale(imageNp)
            for (x,y,w,h) in faces:
                faceSamples.append(imageNp[y:y+h,x:x+w])
                ids.append(id)
    return faceSamples, ids

# Hàm lấy thông tin người dùng qua ID
def get_profile(id):
    conn=sqlite3.connect("./datasets/face_database.db")
    cursor=conn.execute("SELECT * FROM People WHERE ID=" + str(id))
    profile=None
    for row in cursor:
        profile=row
    conn.close()
    return profile
