import cv2
import numpy as np
import torch.nn.functional as F

# coordinates flattened list
def annotate_flat_image(coordinates, image=None, image_path=None):
    # image passed directly -> convert to [x,y,color] - > numpy
    # needs to (h,w,c)

    # image from path
    if image_path != None:
        image = cv2.imread(image_path)

    # Window name in which image is displayed
    window_name = 'Image'
    # Radius of circle
    radius = 1 # 7

    # Blue color in BGR (RBG)
    circle_color = (0,0,0) # 245, 201, 59
    line_color = (240, 214, 22)  # 22, 214, 240

    color1 = (240, 214, 22)
    color2 = (59, 200, 245)
    color3 = (0, 0, 255)
    # Line thickness of 2 px
    thickness = -1
    line_thickness = 2 # 4

    check = 0

    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    #for i, point in enumerate(coordinates):
    for i in range(0, len(coordinates), 2):
        point = (coordinates[i],coordinates[i+1])
        #if type(point[0]) != int or type(point[1]) != int: point = point_toInt(point)
        # draw connecting line
        # [0-5]
        if i > 0: check = i/2

        if check > 0 and check < 6:
            start_point, end_point = start_end_line((coordinates[i-2],coordinates[i-1]), point)
            if valid_points(start_point, point):
                cv2.line(image, start_point, end_point, color1, line_thickness)
                #cv2.circle(image, point, radius, circle_color, thickness)
        # [6-9]
        elif check > 6 and check < 10:
            start_point, end_point = start_end_line((coordinates[i-2],coordinates[i-1]), point)
            if valid_points(start_point, point):
                cv2.line(image, start_point, end_point, color2, line_thickness)
                #cv2.circle(image, point, radius, circle_color, thickness)
        # [10-15]
        elif check > 10:
            start_point, end_point = start_end_line((coordinates[i-2],coordinates[i-1]), point)
            if valid_points(start_point, point):
                cv2.line(image, start_point, end_point, color3, line_thickness)
                #cv2.circle(image, point, radius, circle_color, thickness)

        if valid_point(point):
            cv2.circle(image, point, radius, circle_color, thickness)
            #add text
            cv2.putText(image, str(int(check)), point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    # Displaying the image
    cv2.startWindowThread()
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_annotate_flat_image(coordinates, image=None, image_path=None):
    # image passed directly -> convert to [x,y,color] - > numpy
    # needs to (h,w,c)

    # image from path
    if image_path != None:
        image = cv2.imread(image_path)

    # Window name in which image is displayed
    window_name = 'Image'
    # Radius of circle
    radius = 7 # 7

    # Blue color in BGR (RBG)
    circle_color = (245,201,59) # 245, 201, 59
    line_color = (240, 214, 22)  # 22, 214, 240

    color1 = (240, 214, 22)
    color2 = (59, 200, 245)
    color3 = (0, 0, 255)
    # Line thickness of 2 px
    thickness = -1
    line_thickness = 2 # 4

    check = 0

    # for now
    # coordinates = coordinates[16:]


    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    #for i, point in enumerate(coordinates):
    masks = coordinates[0:16]
    coordinates = coordinates[16:]

    for i in range(0, len(coordinates), 2):
        point = (coordinates[i], coordinates[i + 1])
        # if type(point[0]) != int or type(point[1]) != int: point = point_toInt(point)
        # draw connecting line
        # [0-5]
        if i > 0: check = i / 2

        if check > 0 and check < 6:
            start_point, end_point = start_end_line((coordinates[i - 2], coordinates[i - 1]), point)
            if valid_points(start_point, point):
                cv2.line(image, start_point, end_point, color1, line_thickness)
                # cv2.circle(image, point, radius, circle_color, thickness)
        # [6-9]
        elif check > 6 and check < 10:
            start_point, end_point = start_end_line((coordinates[i - 2], coordinates[i - 1]), point)
            if valid_points(start_point, point):
                cv2.line(image, start_point, end_point, color2, line_thickness)
                # cv2.circle(image, point, radius, circle_color, thickness)
        # [10-15]
        elif check > 10:
            start_point, end_point = start_end_line((coordinates[i - 2], coordinates[i - 1]), point)
            if valid_points(start_point, point):
                cv2.line(image, start_point, end_point, color3, line_thickness)
                # cv2.circle(image, point, radius, circle_color, thickness)

        if valid_point(point):
            cv2.circle(image, point, radius, circle_color, thickness)
            # add text
            #cv2.putText(image, str(int(check)), point, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)

    return image

def annotate_image(image_path, coordinates):
    # Reading an image in default mode
    image = cv2.imread(image_path)
    # Window name in which image is displayed
    window_name = 'Image'
    # Radius of circle
    radius = 7

    # Blue color in BGR (RBG)
    circle_color = (0,0,0) # 245, 201, 59
    line_color = (240, 214, 22)  # 22, 214, 240

    color1 = (240, 214, 22)
    color2 = (59, 200, 245)
    color3 = (0, 0, 255)
    # Line thickness of 2 px
    thickness = -1
    line_thickness = 4

    # Using cv2.circle() method
    # Draw a circle with blue line borders of thickness of 2 px
    for i, point in enumerate(coordinates):
        if type(point[0]) != int or type(point[1]) != int: point = point_toInt(point)
        # draw connecting line
        # [0-5]
        if i > 0 and i < 6:
            start_point, end_point = start_end_line(coordinates[i-1], point)
            if valid_points(start_point, point):
                cv2.line(image, start_point, end_point, color1, line_thickness)
                #cv2.circle(image, point, radius, circle_color, thickness)
        # [6-9]
        elif i > 6 and i < 10:
            start_point, end_point = start_end_line(coordinates[i-1], point)
            if valid_points(start_point, point):
                cv2.line(image, start_point, end_point, color2, line_thickness)
                #cv2.circle(image, point, radius, circle_color, thickness)
        # [10-15]
        elif i > 10:
            start_point, end_point = start_end_line(coordinates[i-1], point)
            if valid_points(start_point, point):
                cv2.line(image, start_point, end_point, color3, line_thickness)
                #cv2.circle(image, point, radius, circle_color, thickness)

        if valid_point(point):
            cv2.circle(image, point, radius, circle_color, thickness)

    # Displaying the image
    cv2.startWindowThread()
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def point_toInt(point):
    a = int(point[0])
    b = int(point[1])
    return (a, b)

def start_end_line(start_point, point):
    if type(start_point[0]) != int or type(start_point[1]) != int: start_point = point_toInt(start_point)
    return start_point, point

def valid_point(point):
    if (point[0] == -1 and point[1] == -1): return False
    return True

def valid_points(point1, point2):
    return valid_point(point1) and valid_point(point2)
