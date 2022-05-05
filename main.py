import cv2
import numpy as np

thres = 0.5
nms_threshold = 0.5

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

classNames = []
classFile = 'resources/coco.names'
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'resources/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightPath = 'resources/frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

while True:
    success, img = cap.read()
    classIds, confs, bbox = net.detect(img, confThreshold = thres)
    bbox = list(bbox)
    confs = list(confs.reshape(1,-1)[0])
    confs = list(map(float,confs))

    indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    # for i in indices:
    #     i = i[0]
    #     box = bbox[i]
    #     x,y,w,h = box[0],box[1],box[2],box[3]
    #     cv2.rectangle(img, (x,y), (x + w, y + h), color = (0,255,0), thickness = 2)
    #     cv2.putText(img, classNames[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
    #                 cv2.FONT_ITALIC, 2, (0, 0, 255))

    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs, bbox):
            cv2.rectangle(img, box, color = (0,255,0), thickness = 2)
            cv2.putText(img, classNames[classId - 1].upper(), (box[0]+10, box[1]+30),
                        cv2.FONT_ITALIC,2,(0,0,255))
            cv2.putText(img, str(round(confidence*100)), (box[0] + 330, box[1]+30),
                        cv2.FONT_ITALIC,2,(0,0,255))

    cv2.imshow("Output", img)
    cv2.waitKey(1)