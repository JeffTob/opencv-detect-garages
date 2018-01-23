#packages
import cv2
import numpy as np
import imutils

#load the image, create a temporarily work image and convert it to grayscale/blur
image = cv2.imread('houses/house19.jpg')
work_image = image.copy()
image_gray = cv2.cvtColor(work_image, cv2.COLOR_BGR2GRAY)

#to a better thresholding and edge detection let's blurred the gray image
image_gray = cv2.GaussianBlur(image_gray, (3, 3), 0)

#load the template image, convert it to grayscale
template = cv2.imread('templates/template3.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
w, h = template.shape[::-1]

#create mask from original image
mask = np.zeros(image.shape[:2], dtype = "uint8")

#create bookkeeping variable
found = None
matched_locations = None

#Now loop over the scales of the working image
for scale in np.linspace(0.2, 1.0, 20)[::-1]:
    # resize the image according to the scale and keep track
    # of the ratio of the resizing
    resized = imutils.resize(image_gray, width=int(image_gray.shape[1] * scale))
    r = image_gray.shape[1] / float(resized.shape[1])
    # if the resized image is smaller than the template, then break
    # from the loop
    if resized.shape[0] < h or resized.shape[1] < w:
        break
    #apply template Matching
    result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
    #threshold 0.8 for 80% only works with TM_CCOEFF_NORMED comparison method
    threshold = 0.8
    #result is greater than or equal to 80%
    loc = np.where(result >= threshold)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    #if you want to see the resizing iteration just uncomment 
    #draw a rectangle around the detected region
    #----------------------------------------------------------------------
    #clone = np.dstack([resized, resized, resized])
    #for pt in zip(*loc[::-1]):
    #    cv2.rectangle(clone, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)
    #cv2.imshow("Iteration", clone)
    #cv2.waitKey(0)
    #----------------------------------------------------------------------
    
    # if have found a new maximum correlation value, then
    # pass to the bookkeeping variables
    if found is None or max_val > found[0]:
        found = (max_val, max_loc, r)
        matched_locations = loc
        
#unpack the bookkeed varaible found        
(the_maxVal, maxLoc, r) = found

#compute the (x, y) coordinates of the bounding box based on the resized ratio
#draw a filled bounding box around in the detected region creating contours
for pt in zip(*matched_locations[::-1]):
    (startX, startY) = (int(pt[0] * r), int(pt[1] * r))
    (endX, endY) = (int((pt[0] + w) * r), int((pt[1] + h) * r))
    rect_rec = work_image[(startX, startY)[1]: (endX, endY)[1], (startX, startY)[0]: (endX, endY)[0]]
    rect_rec[:] = (0, 0, 0)
    #create a mask image with bounding box to detect contours 
    cv2.rectangle(mask, (startX, startY), (endX, endY), 255, -1)
    
#get edges
edged_image = cv2.Canny(mask, 50, 200)
    
#find contours in the edged image
(_, cnts , _) = cv2.findContours(edged_image.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

#loop over the contours and draw a rectangle on original image 
for contour in cnts:
    rect = cv2.boundingRect(contour)
    cv2.rectangle(image, (rect[0],rect[1]), (rect[2]+rect[0],rect[3]+rect[1]), (0,0,0), 2)
    
#compute the center of the contour rectangle area
#write a text in the original image 
for (i, c) in enumerate(cnts):
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Gar:' + str(i+1), (cX - 20, cY), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    
#show the final Result
cv2.imshow("Result Image", image)
cv2.waitKey(0)
