# -*- coding: utf-8 -*-#
# PROJECT_NAME: opencv
# Name:   mainwin
# Author: YunXi
# Date:   2025/7/22
import sys
import numpy as np
import os
import cv2
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *


class mainwin(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(900,600)
        self.setWindowTitle("opcv车流系统")
        self.setWindowIcon(QIcon("./img/Icon.png"))

        self.openFileBtn=QPushButton("打开文件",self)
        self.openFileBtn.setGeometry(20,20,100,30)

        self.grayBtn=QPushButton("灰度处理",self)
        self.grayBtn.setGeometry(140,20,100,30)

        self.imgCheckBtn=QPushButton("车辆图片识别",self)
        self.imgCheckBtn.setGeometry(260,20,140,30)

        self.videoCheckBtn=QPushButton("车辆视频识别",self)
        self.videoCheckBtn.setGeometry(420,20,140,30)

        self.collectFaceBtn = QPushButton("人脸采集",self)
        self.collectFaceBtn.setGeometry(580,20,100,30)

        self.trainFaceBtn = QPushButton("人脸训练", self)
        self.trainFaceBtn.setGeometry(700, 20, 100, 30)

        self.checkFaceBtn = QPushButton("人脸识别", self)
        self.checkFaceBtn.setGeometry(820, 20, 100, 30)

        self.leftlab=QLabel("原图",self)
        self.leftlab.setGeometry(20,80,400,400)
        self.leftlab.setStyleSheet("background-color:white")

        self.rightlab = QLabel("新图", self)
        self.rightlab.setGeometry(440, 80, 400, 400)
        self.rightlab.setStyleSheet("background-color:white")

        self.openFileBtn.clicked.connect(self.openFile)
        self.grayBtn.clicked.connect(self.grayImg)
        self.imgCheckBtn.clicked.connect(self.imgCheck)
        self.videoCheckBtn.clicked.connect(self.videoCheck)
        self.collectFaceBtn.clicked.connect(self.collectFace)
        self.trainFaceBtn.clicked.connect(self.trainFace)
        self.checkFaceBtn.clicked.connect(self.checkFace)

    def openFile(self):
        print("打开文件")
        self.file_path, imgtype = QFileDialog.getOpenFileName(self, "打开文件", "", "图片文件 (*.png *.jpg *.jpeg)")  # 只保留路径
        if self.file_path:  # 检查文件是否选择成功
            self.leftlab.setPixmap(QPixmap(self.file_path))
            self.leftlab.setScaledContents(True)


    def grayImg(self):
        print("灰度处理")
        img=cv2.imread(self.file_path)
        imggray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        newImage="./img/gray.jpg"
        cv2.imwrite(newImage,imggray)
        self.rightlab.setPixmap(QPixmap(newImage))
        self.rightlab.setScaledContents(True)

    def imgCheck(self):
        print("车辆识别")
        img=cv2.imread(self.file_path)
        grayimage=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        car_detector=cv2.CascadeClassifier("./cars.xml")
        cars=car_detector.detectMultiScale(grayimage,1.1,1,cv2.CASCADE_SCALE_IMAGE,(10,60),(120,120))
        for (x,y,w,h) in cars:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        newImage = "./img/jiance.jpg"
        cv2.imwrite(newImage,img)

        self.rightlab.setPixmap(QPixmap(newImage))
        self.rightlab.setScaledContents(True)

    def videoCheck(self):
        print("打开车流视频")
        video,videotype=QFileDialog.getOpenFileName(self,"打开视频","","")
        cap=cv2.VideoCapture(video)
        car_detector = cv2.CascadeClassifier("./cars.xml")
        while True:
            status,img=cap.read()
            if status:
                grayimage = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                cars = car_detector.detectMultiScale(grayimage, 1.1, 3, cv2.CASCADE_SCALE_IMAGE, (50,70), (120, 300))
                for (x, y, w, h) in cars:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 3)

                string="real time traffic flow:"+str(len(cars))
                cv2.putText(img,string,(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255))
                cv2.imshow("opencv",img)
            else :
                break
            key=cv2.waitKey(10)
            if key==27:
                break

        cap.release()
        cv2.destroyAllWindows()

    def collectFace(self):
        print("人脸采集")
        cap = cv2.VideoCapture(0)
        face_detector = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
        i = 1
        while True:
            status,img = cap.read()
            if status:
                img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(img_gray,1.1,2,cv2.CASCADE_SCALE_IMAGE,(200,200),(350,350))
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255))
                    filename="./face_img/zzh{}.jpg".format(i)
                    cv2.imwrite(filename,img_gray[y:y+h,x:x+w])
                    i+=1
                cv2.imshow("opencv",img)
            else:
                break
            if i>100:
                break
            key = cv2.waitKey(2)
            if key ==27:
                break
        cap.release()
        cv2.destroyAllWindows()

    def trainFace(self):
        try:
            print("人脸训练")
            path = "./face_img/"
            if not os.path.exists(path):
                os.makedirs(path)
                print("已创建face_img目录，请先进行人脸采集")
                return
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            facedata = []
            ids = []
            file_list = os.listdir(path)
            if not file_list:
                print("错误：face_img目录下没有图片文件")
                return
            for file in file_list:
                img_path = os.path.join(path, file)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"警告：无法读取图片 {img_path}，已跳过")
                    continue
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 注意：cv2默认读取为BGR格式
                img_gray = cv2.resize(img_gray, (160, 160))
                facedata.append(img_gray)
                ids.append(0)  # 所有图片标记为同一人（可根据需求修改）
            if not facedata:
                print("错误：没有有效的训练图片数据")
                return
            recognizer.train(facedata, np.array(ids))
            recognizer.write("./train.yml")  # 保存模型
            print("训练完毕，模型已保存至train.yml")
        except Exception as e:
            print(f"训练失败：{str(e)}")  # 打印具体错误信息

    def checkFace(self):
        print("人脸识别")
        cap = cv2.VideoCapture(0)
        face_detector = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("./train.yml")

        count=0
        while True:
            status,img = cap.read()
            if status:
                img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(img_gray,1.1,2,cv2.CASCADE_SCALE_IMAGE,(200,200),(350,350))
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255))
                    user_id,confidence=recognizer.predict(img_gray[y:y+h,x:x+w])
                    print(user_id,confidence)
                    chance = round(100-confidence)
                    if chance>65:
                        count+=1

                cv2.imshow("opencv",img)
            else:
                break
            if count>10:
                print("识别成功")
                break

            key = cv2.waitKey(2)
            if key ==27:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app=QApplication(sys.argv)
    mainWin=mainwin()
    mainWin.show()
    sys.exit(app.exec_())