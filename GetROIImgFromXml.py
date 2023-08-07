import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2


classes = ["truckhead","num","sidenum"]

def image_file_name(file_dir):   
    L=[]   
    for root, dirs, files in os.walk(file_dir):  
        for file in files:  
            if os.path.splitext(file)[1] == '.jpg' or os.path.splitext(file)[1] == '.png':  
                L.append(file)  
    return L  


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

        
def convert_annotation(imageName,xmlName,imgSavePath):
    in_file = open(xmlName,encoding='gb18030')
    img = cv2.imread(imageName)
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    num=0
    
    for obj in root.iter('object'):
    
        objectDif=obj.find('difficult')
        difficult = '0'
        if objectDif is not None:
            difficult=obj.find('difficult').text
        cls = obj.find('name').text
        xmlbox = obj.find('bndbox')
        if (cls =="num" or cls=="sidenum"):
            b = [int(xmlbox.find('xmin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymin').text), int(xmlbox.find('ymax').text)]
            if b[0]<0:
                b[0]=0
            if b[1]>=w:
                b[1]=w-1
                
            if b[2]<0:
                b[2]=0
            if b[3]>=h:
                b[3]=h-1
                
            imageNameROI=os.path.splitext(imageName)[0]+str(num)+os.path.splitext(imageName)[-1]
            print(imageNameROI)
            cv2.imwrite(imageNameROI,img[b[2]:b[3],b[0]:b[1]])
        
            num=num+1
        

    tree=ET.ElementTree(root)
    tree.write(xmlName);   
   



imgPath='crnn.pytorch-master/images'
xmlPath='crnn.pytorch-master/xmls'
saveCRNNPath='crnn.pytorch-master/dataSave'

imageNames=image_file_name(imgPath)


for imageName in imageNames:
    xmlName=imageName.replace('jpg','xml').replace('png','xml')

    if os.path.exists(os.path.join(xmlPath,xmlName)):
        convert_annotation(os.path.join(imgPath,imageName),os.path.join(xmlPath,xmlName),saveCRNNPath)










