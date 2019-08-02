import cv2
import numpy as np
from pynput.keyboard import Key, Controller

keyboard = Controller()


# init part
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
detector_params = cv2.SimpleBlobDetector_Params()
detector_params.filterByArea = True
detector_params.maxArea = 1500
detector = cv2.SimpleBlobDetector_create(detector_params)


def detect_faces(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cascade.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        cv2.rectangle(img, (x, y), (x + w, y + h), (230, 230, 0), 2)
        frame = img[y:y + h, x:x + w]
    return frame


def detect_eyes(img, cascade):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = cascade.detectMultiScale(gray_frame, 1.3, 5)  # detect eyes
    width = np.size(img, 1)  # get face frame width
    height = np.size(img, 0)  # get face frame height
    left_eye = None
    right_eye = None
    
    for (x, y, w, h) in eyes:
        if y > height / 2:
            pass
        eyecenter = x + w / 2  # get the eye center
        color = (255, 0, 0)
        if eyecenter < width * 0.5:
            left_eye = img[y:y + h, x:x + w]
            color = (0, 0, 255)
        else:
            right_eye = img[y:y + h, x:x + w]
            color = (255, 0, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)        
    
    if left_eye is None and right_eye is not None:
        keyboard.press('r')
        keyboard.release('r')

    if left_eye is not None and right_eye is None:
        keyboard.press('l')
        keyboard.release('l')


    return left_eye, right_eye


def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 5)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)

    return img


def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('eye1', gray_frame)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2)
    img = cv2.dilate(img, None, iterations=4)
    img = cv2.medianBlur(img, 5)
    # cv2.imshow('eye2', img)
    keypoints = detector.detect(img)
    return keypoints


def nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('image')
    # cv2.namedWindow('eye1')
    # cv2.namedWindow('eye2')
    cv2.createTrackbar('threshold', 'image', 0, 255, nothing)
    while True:
        _, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
        face_frame = detect_faces(frame, face_cascade)
        if face_frame is not None:
            eyes = [detect_eyes(face_frame, eye_cascade)[1]]
            for eye in eyes:
                if eye is not None:
                    threshold = r = cv2.getTrackbarPos('threshold', 'image')
                    eye = cut_eyebrows(eye)
                    keypoints = blob_process(eye, threshold, detector)
                    eye = cv2.drawKeypoints(eye, keypoints, eye, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        frame = cv2.flip(frame, +1)
        cv2.imshow('image', frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
