#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:02:45 2023

@author: lihongcheng
"""

import cv2
import numpy as np
from api_temp import *

# 開啟網路攝影機
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('rtsp://root:Admin1234@59.125.76.241:5540/live1s1.sdp')
cap = cv2.VideoCapture('/Users/lihongcheng/Desktop/heatstress/0110_heat.mkv')
# cap = cv2.VideoCapture('rtsp://admin:Foxconn_88@59.125.76.241:5541/Streaming/Channels/201?transportmode=unicast&profile=Profile_201')

# 設定影像尺寸
width = 1280
height = 960

# 設定擷取影像的尺寸大小
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
count = 8000
cap.set(cv2.CAP_PROP_POS_FRAMES, count)
# 計算畫面面積
area = width * height



object_detector = cv2.createBackgroundSubtractorMOG2()

# 初始化平均影像
ret, frame = cap.read()
# cv2.imwrite("output_heat.jpg",frame)
avg = cv2.blur(frame, (4, 4))
# avg = cv2.GaussianBlur(frame, (3, 3),1)
avg_float = np.float32(avg)

#氣象站參數
url='https://59.125.76.241:4431/ISAPI/Thermal/channels/2/thermometry/1/rulesTemperatureInfo?format=json'
user='admin'
password='Foxconn_88'
header={
    'User-Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Mobile Safari/537.36'
    }
method='get'

sss
while(cap.isOpened()):
    
    # 讀取一幅影格
    ret, frame = cap.read()
    
    # 若讀取至影片結尾，則跳出
    if ret == False:
        break
    #獲取氣象站資料
    # method='get'
    # getdata=Get_temp(url,user,password,header)
    # connect=getdata.connect_type(method)
    # temp=getdata.temp(connect)
    # maxtemp=temp[0]
    # mintemp=temp[1]
    
    maxtemp=33
    mintemp=4
    
    #對照顏色
    ap = np.linspace(maxtemp,mintemp,52)
    colorbar=frame[100:617,1268:1269].reshape([517,3])[::10 ]
    color_map = {str(x):y for x, y in zip(ap.tolist(), colorbar.tolist())}
    
    
    # 模糊處理
    blur = cv2.blur(frame, (4, 4))
    # blur = cv2.GaussianBlur(frame, (3, 3),1)

    # 計算目前影格與平均影像的差異值
    diff = cv2.absdiff(avg, blur)

    # 將圖片轉為灰階
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    cv2.imshow('gray', gray)
    # 篩選出變動程度大於門檻值的區域(25,255)
    ret, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
    # thresh= cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 使用型態轉換函數去除雜訊
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 產生等高線
    # cntImg, cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntImg, cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.RETR_TREE)
    if type(cnts) == type(None):
        continue
    else:
        for c in cntImg:
            # 忽略太小的區域
            if cv2.contourArea(c) < 2500:
                continue

            # # 偵測到物體，可以自己加上處理的程式碼在這裡...

            # 計算等高線的外框範圍
            (x, y, w, h) = cv2.boundingRect(c)

            # 畫出外框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            colorframe=frame[x:x+w,y:y+h]
            
            max_temp=max({str(x):y for x, y in zip(colorframe.tolist()[0], color_map)}.values())
            # print(max_temp)
            max_temp=str(round(float(max_temp),2))
            # max_temp=str(round(maxtemp,2))
            cv2.putText(frame, max_temp, (int(x+w/2), y), cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 1, cv2.LINE_AA)
            
            # # 畫出等高線（除錯用）
            # cv2.drawContours(frame, cntImg, -1, (0, 255, 255), 2)

        # 顯示偵測結果影像
    cv2.imshow('frame', frame)
    # cv2.imshow('frame', thresh)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # 更新平均影像
    cv2.accumulateWeighted(blur, avg_float, 0.1)
    avg = cv2.convertScaleAbs(avg_float)

# cap.release()
# cv2.destroyAllWindows()

# 修正影片關閉問題
cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)