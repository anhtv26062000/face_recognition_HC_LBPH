import cv2
import numpy as np
import sqlite3

# Hàm cập nhật tên và ID vào CSDL
def insert_update(id, name):
    conn=sqlite3.connect("./datasets/face_database.db")
    cursor=conn.execute('SELECT * FROM People WHERE ID='+str(id))
    isRecordExist=0
    for row in cursor:
        isRecordExist = 1
        break

    if isRecordExist==1:
        cmd="UPDATE people SET Name=' "+ str(name) +" ' WHERE ID=" + str(id)
    else:
        cmd="INSERT INTO people(ID, Name) Values(" + str(id) +",' " + str(name) + " ' )"

    conn.execute(cmd)
    conn.commit()
    conn.close()

def add_user(id, name, cam, detector):
    sampleNum=1
    while(True):
        ret, img = cam.read()

        # Flip camera
        img = cv2.flip(img,1)

        # Kẻ khung chuẩn để detect
        centerH = img.shape[0] // 2;
        centerW = img.shape[1] // 2;
        mask = np.zeros_like(img)
        mask = cv2.ellipse(mask, (centerW, centerH), (90, 150), 0, 0, 360, (255, 255, 255), (-1))
        img = cv2.bitwise_and(img, mask)

        # Chuyển về ảnh mức xám
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Nhận diện khuôn mặt
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            # Vẽ hình chữ nhật quanh mặt nhận được
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            # Ghi dữ liệu khuôn mặt vào thư mục dataSet
            cv2.imwrite("datasets/data/" + id + '_' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
            sampleNum = sampleNum + 1

        cv2.imshow('frame', img)
        # Check xem có bấm q hoặc trên 100 ảnh sample thì thoát
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        elif sampleNum>100:
            break

    cam.release()
    cv2.destroyAllWindows()
