import os
import cv2
import pickle
import sqlite3
import numpy as np
from src.add import *
from src.get import *
from PIL import Image

detector=cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Chỗ này chỉnh tay nếu chạy trên VSCode (1 là phải chỉnh mode cho phép nhập - 2 là set cứng giá trị n tại đây)
# n = 1
n = int(input())
while n!=1 and n!=2 and n!=3:
    n = int(input("Nhập lại n (1-Thêm thông tin sinh viên / 2-Train lại model / 3-Chạy code nhận diện face"))

if n==1:
    id = input('Nhập mã số sinh viên: ')
    name = input('Nhập tên sinh viên: ')
    # id = "18520471"
    # name = "To Viet Anh"
    print("Bắt đầu chụp ảnh sinh viên cho đến khi màn hình tắt\n*Nhấn q để thoát ngay lập tức!*")
    insert_update(id, name)
    cam = cv2.VideoCapture(0)
    add_user(id, name, cam, detector)
elif n==2:
    # Lấy các khuôn mặt và id từ thư mục dataSet
    faceSamples, ids = get_data('./datasets/data', detector)
    # Train model để trích xuất đặc trưng các khuôn mặt và gán với từng nhân viên
    recognizer.train(faceSamples, np.array(ids))
    # Lưu model
    recognizer.save('./models/trainner.yml')
    print("Trained!")
else:
    recognizer.read('./models/trainner.yml')
    id=0
    #set text style
    fontface = cv2.FONT_HERSHEY_SIMPLEX
    fontscale = 0.5
    fontcolor = (0,255,0)
    fontcolor1 = (0,0,255)

    # Khởi tạo camera
    cam=cv2.VideoCapture(0)
    while(True):
        # Đọc ảnh từ camera
        ret, img=cam.read()
        # Lật ảnh cho đỡ bị ngược
        img = cv2.flip(img, 1)

        # # Vẽ khung chữ nhật để định vị vùng người dùng đưa mặt vào
        # centerH = img.shape[0] // 2;
        # centerW = img.shape[1] // 2;
        # sizeboxW = 300;
        # sizeboxH = 400;
        # cv2.rectangle(img, (centerW - sizeboxW // 2, centerH - sizeboxH // 2),
        #               (centerW + sizeboxW // 2, centerH + sizeboxH // 2), (255, 255, 255), 5)

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiScale(gray,1.3,5)
        # Lặp qua các khuôn mặt nhận được để hiện thông tin
        for(x, y, w, h) in faces:
            # Vẽ hình chữ nhật quanh mặt
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # Nhận diện khuôn mặt, trả ra 2 tham số id: mã sinh viên và dist (độ sai khác)
            id, dist=recognizer.predict(gray[y:y+h, x:x+w])
            profile=None

            # Nếu độ sai khác < 25% thì lấy profile
            # if (dist<=25):
            if (dist<=55):
                profile=get_profile(id)

            # Hiển thị thông tin tên người hoặc Unknown nếu không tìm thấy
            if(profile!=None):
                cv2.putText(img, "Name: " + str(profile[1]), (x,y+h+30), fontface, fontscale, fontcolor, 2)
                cv2.putText(img, "Score: " + str(round(dist, 2)), (x,y+h+50), fontface, fontscale, fontcolor, 2)
                print(profile[1], round(dist, 2))
            else:
                cv2.putText(img, "Name: Unknown", (x, y + h + 30), fontface, fontscale, fontcolor1, 2)

        cv2.imshow('Face Recognition - To Viet Anh - 18520471', img)

        # Nếu nhấn q thì thoát
        if cv2.waitKey(1)==ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
