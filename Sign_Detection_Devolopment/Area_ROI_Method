# Prototype of Area on IGVC - ROI Method
# This prototype uses a face detection AI to show ROI method of getting area of contours in region of interest (ROI) determined by bounding box
import cv2
import numpy as np

def contour_frame(frame, threshold1, threshold2):
    #face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
    imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
    imgCanny = cv2.Canny(imgBlur, threshold1, threshold2)
    #kernel = np.ones((5, 5))
    #imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
    #faces = face_cascade()
    processed_frame = imgCanny
    return processed_frame

def get_bbox_vals(imgIn):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    bboxes = tuple(face_cascade.detectMultiScale(imgIn))
    return bboxes


def get_contours(img, roi, bbox_info):
    total_area_roi = 0
    for bbox_tuple in bbox_info:
        bbox = bbox_tuple
        xb, yb, wb, hb = bbox
        roi = img[yb:yb+hb, xb:xb+wb]
        contoursROI, hierarchyROI = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contoursROI:
            area_roi = cv2.contourArea(cnt)
            #cv2.drawContours(roi, cnt, -1, (255, 0, 255), 7)
            if area_roi > 500:
                cv2.drawContours(roi, cnt, -1, (255, 0, 255), 7)
                # calc area of all contours with area > threshold
                total_area_roi = total_area_roi + area_roi 
    return total_area_roi


def gen_AI_image(imgInput): # FOR TROUBLESHOOTING
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(imgInput)
    for (x, y, w, h) in faces:
        cv2.rectangle(imgInput, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return imgInput

def main():
    # Open the default camera (usually the first webcam connected)
    cap = cv2.VideoCapture(0)

    # Threshold adjustment
    cv2.namedWindow('Parameters')
    cv2.resizeWindow('Parameters', 640, 240)
    def empty(a):
        pass
    threshold1 = cv2.createTrackbar('Threshold1', 'Parameters', 185, 255, empty)
    threshold2 = cv2.createTrackbar('Threshold2', 'Parameters', 144, 255, empty)


    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame.")
            break

        imgContour = frame.copy()

        # Bounding box location (x, y, w, h)
        bbox_vals = get_bbox_vals(frame)    
        #print(bbox_vals)

        # Process the frame
        contoured_frame = contour_frame(frame, threshold1, threshold2)

        # Image with contours & area inside contours in bbox
        # Image with contours at imgContour
        area_in_contours = get_contours(contoured_frame, imgContour, bbox_vals)
        print(area_in_contours)


        AI_gen_img = gen_AI_image(frame)

        # Display the processed frame
        cv2.imshow('Processed Frame', contoured_frame)
        cv2.imshow('Contoured Image', imgContour)
        cv2.imshow('AI Image', AI_gen_img)

        # Check for 'q' key press to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
