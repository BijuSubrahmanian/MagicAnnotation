import numpy as np
import sys
import cv2
import time
import imutils
import os
import time

Base_path="C:/biju/Experiments/MagicAnnotation/"

#Create Json Annotation template

try:
    objecttoannotate=sys.argv[1]
    videofile = sys.argv[2]
except:
    print('Arguments missing')
    print('@Usage : python ObjectDetectionImageLive.py <<object name>> << video file>>')
    exit()

directory =objecttoannotate
if not os.path.exists(Base_path+"/"+directory):
    os.makedirs(Base_path+"/"+directory)



pointfactor=4

relevant_path = "MagicAnnotation/" + objecttoannotate
FrameNo=1
#included_extensions = ['jpg','jpeg', 'bmp', 'png', 'gif']
#file_names = [fn for fn in os.listdir(relevant_path)
#              if any(fn.endswith(ext) for ext in included_extensions)]

jsontemplateMaster="""{
  "version": "3.16.7",
  "flags": {
    "__ignore__": false,
    "$object": true
      },
  "shapes": [
    {
      "label": "$object",
      "line_color": null,
      "fill_color": null,
      "points": [
$points
],
      "shape_type": "polygon",
      "flags": {}
    }
  ],
  "lineColor": [
    0,
    255,
    0,
   128
  ],
  "fillColor": [
    255,
    0,
    0,
   128
  ],
  "imagePath": "$imfile",
  "imageData": null,
  "imageHeight":$imheight,
  "imageWidth":$imwidth
}"""



# Creating a VideoCapture object
# This will be used for image acquisition later in the code.
cap = cv2.VideoCapture(videofile)
ret = True
while (ret):

    # Capturing the live frame
    ret, img = cap.read()
    #time.sleep(0.5)
#for imfile in file_names:

        # Capturing the live frame
    #imfile='cat.png'
    jsontemplate=jsontemplateMaster

    #print(img.shape)
    try:
        imageheight=img.shape[0]
        imagewidth=img.shape[1]
    except:
        print('End of file')
        exit()
    imfile=str(FrameNo) + '.jpg'
    jsonfilename=str(FrameNo) +'.json'
    FrameNo=FrameNo +1

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        #thresh=gray
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    #cv2.imshow('thresh',thresh)
    # find contours in thresholded image, then grab the largest
    # one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    #print('cnts',cnts)
    c = max(cnts, key=cv2.contourArea)
    a=True
    pointsstr=""
    pointlength=len(c)
    cntr=0
    for i in c:
        cntr= cntr + 1
        if cntr%pointfactor == 0:
            a=True
        else:
            a=False
        if a:
            for j in i:
                if pointlength ==cntr:
                    pointsstr=pointsstr + "\n" + '[' + str(j[0]) + ',' + str(j[1]) +']'
                else:
                    if cntr + pointfactor <= pointlength:
                        pointsstr = pointsstr + "\n" + '[' + str(j[0]) + ',' + str(j[1]) + '],'
                    else:
                        pointsstr = pointsstr + "\n" + '[' + str(j[0]) + ',' + str(j[1]) + ']'
                #print ('[',j[0],',',j[0],']',',')
        a= not a
            #a= not a
    print('pointstr', pointsstr)

        #determine the most extreme points along the contour
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

        # show the output image
    #cv2.imshow("Image", img)

    jsontemplate=jsontemplate.replace('$points',pointsstr)
    jsontemplate=jsontemplate.replace('$object',objecttoannotate)
    jsontemplate=jsontemplate.replace('$imfile',imfile)
    jsontemplate=jsontemplate.replace('$imheight',str(imageheight))
    jsontemplate=jsontemplate.replace('$imwidth',str(imagewidth))

    cv2.imwrite(relevant_path +'/'+ imfile,img)
    #print(jsontemplate)
    f = open(relevant_path +'/'+jsonfilename, "w")
    f.write(jsontemplate)
    imgtodisplay=img
    cv2.drawContours(imgtodisplay, [c], -1, (0, 255, 255), 2)
    cv2.imshow('image', cv2.resize(imgtodisplay, (1280, 960)))
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        cap.release()
        break

    #cv2.imwrite('processedimg.jpg',img)