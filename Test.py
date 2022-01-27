import cv2 as cv
import numpy as np
from realsense_camera import *
from mask_rcnn import *




rs= RealsenseCamera()
mrcnn=MaskRCNN()

#You might need to adjust path for mask mrcnn

while True:
    #Get info in real time
    ret, bgr_frame, depth_frame = rs.get_frame_stream()
    
    boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)

    # print("boxes"   ,boxes)
    # print("contours",contours)
    # print("classes" ,classes)
    # print("centers" ,centers)

   

    #Draw
    bgr_frame = mrcnn.draw_object_mask(bgr_frame)

    mrcnn.draw_object_info(bgr_frame,depth_frame)

#Keep to play with 
    # point_x, point_y = 250, 100
    # dist_Frame=depth_frame[point_y, point_x]

    # cv.circle(bgr_frame, (point_x, point_y), 8, (0, 0, 255), -1)

    # cv.putText(bgr_frame, "{} mm".format(dist_Frame), (point_x, point_y-10), 0, 1, (0,0,255), 2)
    # print(dist_Frame)

#Seperate stuff below for ROS package
    #So depth Frame Values is a 2D Matrix 720, 1280
    #So BGR   Frame Values is a 3D Matrix thing 720 by 1280 by 3

    cv.imshow("depth frame", depth_frame)
    cv.imshow("Bgr frame", bgr_frame)
    

    key = cv.waitKey(1)
    if key == 27:
        break

