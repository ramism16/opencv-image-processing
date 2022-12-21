"""
cv2.VideoCapture class used to read webcam input from computer
and operations on this input to track and detect some object of choice

cv2.VideoCapture() constructor can be provided with video source or IP stream
or (an integer for) opening (one of the) attached cameras for video capturing

cv2.VideoCapture.get(propID) will return properties of the video if applicable
propID is in the range [0,18]

cap = cv2.VideoCapture(0), sometimes cap may have not initialized the capture
in that case, check if not cap.isOpened() then cap.open()

cap.set(cap.get(4),240) will set the height in spatial resolution to 240

as always, to show video loop cv2.imshow and use appropriate time with cv2.waitKey()
if too less, video will be very fast, and if high, video will be slow
25ms is average value for normal cases, this is the capturing response time

To save a video, we create and use a VideoWriter object, specify codec in FourCC
"""

import cv2
import numpy as np
from imutils import perspective

#open video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened(): cap.open()

print("Press escape to end video capturing")

#blank callable for trackbars
def dont(x):
    pass

#function to find the midpoint of line
def midpoint(a,b):
    return ( int((a[0] + b[0]) * 0.5), int((a[1] + b[1]) * 0.5) )

#function to find the midpoint of a contour box
def boxMidpoint(box):
    tltr = midpoint( box[0], box[1])
    blbr = midpoint( box[3], box[2])
    tlbl = midpoint( box[0], box[2])
    trbr = midpoint( box[1], box[2])
    return [ midpoint(tlbl,trbr), midpoint(tltr,blbr) ] 

#HSV channel bounds trackbars to define desired HSV range
wname = "S & V Trackbars"
cv2.namedWindow(wname)
cv2.createTrackbar("Low Hue", wname, 0, 255, dont)
cv2.createTrackbar("High Hue",wname, 0, 255, dont)
cv2.createTrackbar("Low Sat", wname, 0, 255, dont)
cv2.createTrackbar("High Sat", wname, 0, 255, dont)
cv2.createTrackbar("Low Value", wname, 0, 255, dont)
cv2.createTrackbar("High Value", wname, 0, 255, dont)
cv2.createTrackbar("Min Contour Area", wname, 10, 3000, dont)

#while the VideoCapture is reading (while recording)
while(True):
    # Take each frame
    ret, frame = cap.read()

    #if frame captured from output
    if ret:
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        #get trackbar positions for channel bounds
        lowH = cv2.getTrackbarPos("Low Hue", wname)
        highH = cv2.getTrackbarPos("High Hue", wname)
        lowS = cv2.getTrackbarPos("Low Sat", wname)
        highS = cv2.getTrackbarPos("High Sat", wname)
        lowV = cv2.getTrackbarPos("Low Value", wname)
        highV = cv2.getTrackbarPos("High Value", wname)
        minContourBar = cv2.getTrackbarPos("Min Contour Area", wname)
        
        #desired HSV range defined in trackbar window
        lowerHSV = np.array([lowH,lowS,lowV])
        upperHSV = np.array([highH,highS,highV])

        # Threshold the HSV image to get only blue colors
        mask = cv2.inRange(hsv, lowerHSV, upperHSV)

        #finding the contours on the thresolded image(mask) to find object-bounding box
        #cv2.RETR_EXTERNAL retrieves only the extereme outer contours
        #cv2.CHAIN_APPROX_SIMPLE defines horizontal, vertical and diagonal segments with end points only
        contours, heirarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)             
                
        # Bitwise-AND mask and original image
        res = cv2.bitwise_and(frame,frame, mask= mask)
        
        #Remove noise from the frame by removing unnecessary objects in the foreground
        cv2.morphologyEx(res, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))

        #Remove noise from the result frame by filling in the image holes
        cv2.morphologyEx(res, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations = 3)


        selectedContours = []
        #iterate through the contours to find the one with the larger area (i.e. remove noise in brighter areas)
        #and filter the contours with higher area to calculate their midpoint and output to terminal
        for c in contours:
            if cv2.contourArea(c) > minContourBar:
                #model and save a box from the contour endpoint coordinates
                box = np.array( cv2.boxPoints(cv2.minAreaRect(c)) , dtype ="int")
                #order the coordinates (topleft, topright, bottomright, bottomleft)
                box = perspective.order_points(box)
                #find the midpoint pixel coordinate of the larger contours(detected objects)
                mid = boxMidpoint(box)
                #display the midpoint
                print("Contour midpoint: ", mid)
                #add to selected contours, to draw bounding box around detected objects and filter noise
                selectedContours.append(c)

        #drawing the contours received from the mask
        #drawContours(image, contours, -1(all contours)/n(color list), bounding box color(BGR), thickness)
        #white bounding box, selected contours, and 3 pt thickness
        cv2.drawContours(res, np.array(selectedContours), -1, (0,0,128), 3)
        cv2.drawContours(frame, np.array(selectedContours), -1, (0,0,128), 3)
        
        #displaying the stream output
        cv2.imshow('Camera',frame)
        cv2.imshow('mask', mask)
        cv2.imshow('Result',res)

    #5-sec response time, ESC key to break loop
    k = cv2.waitKey(15) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()

#destroying the video capture object
del cap
