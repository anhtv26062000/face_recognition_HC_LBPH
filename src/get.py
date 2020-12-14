import cv2, os
import numpy as np
import image
import sqlite3
from PIL import Image
from datetime import datetime

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

def save_attendance(id_, name):
    with open("./attendance.csv", "r+") as f:
        myDatalist = f.readlines()
        id_list = []
        for line in myDatalist:
            entry = line.split(",")
            id_list.append(entry[0])
        if id_ not in id_list:
            now = datetime.now()
            date = now.strftime("%A %d-%b-%Y %H:%M:%S")
            f.writelines(f'\n{id_}, {name}, {date}')