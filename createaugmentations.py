
import numpy as np
import cv2
import glob
import os

backgroundimgpath="backgroundimage/*.*"

imgaugsavelocation="aug/"

#Create aug directory if not present

if not os.path.exists(imgaugsavelocation):
    os.makedirs(imgaugsavelocation)

pascalvocannotationloc="hand/train/*.json"

def getpolygonpoints(annotatedjsonfile):
    import json
 
    # Opening JSON file
    f = open(annotatedjsonfile)
 
    # returns JSON object as
    # a dictionary
    data = json.load(f)
 
    # Iterating through the json
    # list


    for i in data['shapes']:
        polygon_cords=i['points']
 
    # Closing file
    f.close()
    return polygon_cords

def embedimage_to_bg(annotatedjsonfile,img,backgroundimg):

    height = img.shape[0]
    width = img.shape[1]
    
    polygon_cords=getpolygonpoints(annotatedjsonfile)

    backgroundimg=cv2.resize(backgroundimg, (width, height), interpolation = cv2.INTER_CUBIC)

    mask = np.zeros((height, width), dtype=np.uint8)
    #points = np.array([[[10,150],[150,100],[300,150],[350,100],[310,20],[35,10]]])
    points = np.array([polygon_cords])
    cv2.fillPoly(mask, points, (255))

    res = cv2.bitwise_and(img,img,mask = mask)

    rect = cv2.boundingRect(points) # returns (x,y,w,h) of the rect
    #im2 = np.full((res.shape[0], res.shape[1], 3), (0, 255, 0), dtype=np.uint8 ) # you can also use other colors or simply load another image of the same size
    im2=backgroundimg
    maskInv = cv2.bitwise_not(mask)
    colorCrop = cv2.bitwise_or(im2,im2,mask = maskInv)
    finalIm = res + colorCrop
    #cropped = finalIm[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]
    return finalIm
    
def createannotatedjsonfile(inputfile,outputfile,imagename):
    import json
 
    # Opening JSON file
    f = open(inputfile)
 
    # returns JSON object as
    # a dictionary
    data = json.load(f)
    data['imagePath']=imagename 
    # Closing file
    f.close()
    with open(outputfile, "w") as outfile:
        json.dump(data, outfile)
    


#Get all background images
cntaug=0
for bgm in glob.glob(backgroundimgpath):
    print(bgm)
    backgroundimg=cv2.imread(bgm)
    #Apply object - embed to bg
    #get from json
    for jsonfile in glob.glob(pascalvocannotationloc):
        imgfile=jsonfile.replace('.json','.jpg')
        img=cv2.imread(imgfile)
        #polygoncord=getpolygonpoints(jsonfile)
        imageann=embedimage_to_bg(jsonfile,img, backgroundimg)
        cntaug+=1
        #save aug image
        cv2.imwrite(imgaugsavelocation+'aug_'+str(cntaug)+".jpg",imageann)
        print("image saved -",imgaugsavelocation+'aug_'+str(cntaug)+".jpg")
        #save aug json
        #createannotatedjsonfile('hand/train/1.json','aug/aug_1.json','aug_1.jpg')
        createannotatedjsonfile(jsonfile,imgaugsavelocation+'aug_'+str(cntaug)+".json",'aug_'+str(cntaug)+".jpg")
