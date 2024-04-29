# from nis import match

import cv2
import numpy as np
import pytesseract
import os
from datetime import datetime,timedelta
from pymongo import MongoClient


#Connection to store dada into database
client = MongoClient('mongodb+srv://purtidpatil:admin@renttrack-db.al3r5mc.mongodb.net/?retryWrites=true&w=majority&appName=renttrack-db')
db = client['renttrack-db']
collection = db['tenant']


per=25
pixelThreshold=500
roi=[[(264, 322), (892, 358), 'text', 'Name'],
     [(264, 398), (892, 436), 'text', 'College'],
     [(532, 468), (892, 504), 'text', 'mail'],
     [(260, 542), (892, 582), 'text', 'Address'],
     [(260, 616), (572, 650), 'text', 'Contact'],
     [(264, 762), (562, 794), 'text', 'nationality'],
     [(262, 838), (566, 870), 'text', 'dob'],
     [(264, 918), (290, 942), 'box', '2 sharing'],
     [(348, 920), (372, 936), 'box', '3 sharing'],
     [(260, 988), (284, 1006), 'box', 'ac'],
     [(346, 988), (370, 1010), 'box', 'non ac']]




pytesseract.pytesseract.tesseract_cmd='C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
# imgQ=cv2.imread('Query-image.jpg')
#give here actual path of query image
imgQ=cv2.imread("C:\\Users\\Asus\\Desktop\\PBL project\\pythonProject1\\Query-image.jpg")

if imgQ is not None:
    h, w, c = imgQ.shape
    print(f'Image shape: {h}x{w}x{c}')
else:
    print('Error loading the image. Please check the file path.')



# h,w,c=imgQ.shape


# imgQ=cv2.resize(imgQ,(w//3,h//3))
#
orb=cv2.ORB_create(1000)
kp1,des1=orb.detectAndCompute(imgQ,None)
# impKp1=cv2.drawKeypoints(imgQ,kp1,None)
# path='userforms'
path="C:\\Users\\Asus\\Desktop\\PBL project\\pythonProject1 - Copy\\userform1"
myPicList=os.listdir(path)
print(myPicList)
for j,y in enumerate(myPicList):
    img=cv2.imread(path+"/"+y)
    img = cv2.resize(img, (w // 3, h // 3))
    cv2.imshow(y,img)
    kp2,des2=orb.detectAndCompute(img,None)
    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    matches=bf.match(des2,des1)
 
    matches=sorted(matches, key=lambda x:x.distance)



    good=matches[:int(len(matches)*(per/100))]
    imgMatch=cv2.drawMatches(img,kp2,imgQ,kp1,good[:100],None,flags=2)
    # imgMatch=cv2.drawMatches(img,kp2,imgQ,kp1,matches[:10],None,flags=2)
    # imgMatch = cv2.resize(imgMatch, (w // 3, h // 3))
    # cv2.imshow(y, imgMatch)

    srcPoints=np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints=np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, _ =cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC,5.0)
    imgScan=cv2.warpPerspective(img, M, (w, h))
    # print(w,h)
    # imgScan = cv2.resize(imgScan, (w // 3, h // 3))
    # cv2.imshow(y, imgScan)

    imgShow=imgScan.copy()
    imgMask=np.zeros_like(imgShow)
    myData=[]
    print(f'*************Extraction Data from Form : {j}*************')
    for x,r in enumerate(roi):
        cv2.rectangle(imgMask,(r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,255,0),cv2.FILLED)
        imgShow=cv2.addWeighted(imgShow,0.99,imgMask,0,1,0)

        imgCrop=imgScan[r[0][1]:r[1][1],r[0][0]:r[1][0]]
        # print(imgCrop.shape)
        # cv2.imshow(str(x),imgCrop)

        if r[2] == 'text':
            print(f'{r[3]}:{pytesseract.image_to_string(imgCrop)}')
            myData.append(pytesseract.image_to_string(imgCrop))
        if r[2]=='box':
            imgGray=cv2.cvtColor(imgCrop,cv2.COLOR_BGR2GRAY)
            imgThresh=cv2.threshold(imgGray,170,255,cv2.THRESH_BINARY_INV)[1]
            totalPixels=cv2.countNonZero(imgThresh)
            # print(totalPixels)
            if totalPixels>pixelThreshold : totalPixels=1
            else : totalPixels=0
            print(f'{r[3]}:{totalPixels}')
            myData.append(totalPixels)

    with open('dataInfo.csv','a+') as f:
        for data in myData:
            # f.write((str(data)+','))
            f.write((str(data)).replace("\n",", "))
        f.write('\n')


    print(myData)
    # print(myData[0])
    name=myData[0]
    college=myData[1]
    email=myData[2]
    address=myData[3]
    phone=myData[4]
    nationality=myData[5]
    dob=myData[6]
    if(myData[7]==1):
        two=myData[7]
    else:
        three=myData[8]    
    

    new_user = {
        'name': name,
        'college':college,
        'email': email,
        'address': address,
        'phone': phone,
        'nationality': nationality,
        'dob': dob,
        # 'city': city,
        'last_payment_timestamp': datetime.now()
    }
    collection.insert_one(new_user)
    

    # imgShow = cv2.resize(imgShow, (w // 3, h // 3))
    cv2.imshow(y+"2", imgShow)

# cv2.imshow("Keypoints",impKp1)
cv2.imshow("Output",imgQ)
cv2.waitKey(0)
