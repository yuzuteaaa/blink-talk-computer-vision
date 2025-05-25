import dlib
import cv2
import numpy as np
import time
import pyttsx3
from translate import Translator
import os
import datetime



def eye_on_mask(mask, side, shape):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    l = points[0][0]
    t = (points[1][1] + points[2][1]) // 2
    r = points[3][0]
    b = (points[4][1] + points[5][1]) // 2
    return mask, [l, t, r, b]


def midpoint(p1, p2):
    return int((p1.x + p2.x) / 2), int((p1.y + p2.y) / 2)


def euclidean_distance(leftx, lefty, rightx, righty):
    return np.sqrt((leftx - rightx) ** 2 + (lefty - righty) ** 2)


def get_EAR(eye_points, facial_landmarks):
    left_point = [facial_landmarks.part(
        eye_points[0]).x, facial_landmarks.part(eye_points[0]).y]
    right_point = [facial_landmarks.part(
        eye_points[3]).x, facial_landmarks.part(eye_points[3]).y]
    center_top = midpoint(facial_landmarks.part(
        eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(
        eye_points[5]), facial_landmarks.part(eye_points[4]))
    hor_line_lenght = euclidean_distance(
        left_point[0], left_point[1], right_point[0], right_point[1])
    ver_line_lenght = euclidean_distance(
        center_top[0], center_top[1], center_bottom[0], center_bottom[1])
    EAR = ver_line_lenght / hor_line_lenght
    return EAR


def find_eyeball_position(end_points, cx, cy):
    width = end_points[2] - end_points[0]
    height = end_points[3] - end_points[1]

    if width == 0 or height == 0:
        return 0  # cegah pembagian nol

    relative_x = (cx - end_points[0]) / width
    relative_y = (cy - end_points[1]) / height

    if relative_x < 0.35:
        return 1  # kiri
    elif relative_x > 0.65:
        return 2  # kanan
    elif relative_y < 0.35:
        return 3  # atas
    else:
        return 0  # tengah/normal


def contouring(thresh, mid, img, end_points, right=False):
    cnts, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key=cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)

        pos = find_eyeball_position(end_points, cx, cy)
        return pos
    except:
        pass


def process_thresh(thresh):
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    thresh = cv2.medianBlur(thresh, 3)
    thresh = cv2.bitwise_not(thresh)
    return thresh


def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def search_word(lis, dic):
    for i in lis:

        try:
            if (dic[i]):
                return (dic[i])
        except:
            return 'Eye Movement States are not Recognized by Our System'
        else:
            return "End of text"


def print_eye_pos(img, left, right, lis):
    if left == right and left != 0:
        text = ''
        if left == 1:
            print('Looking left')
            text = 'left'
            lis.append(text)

        elif left == 2:
            print('Looking right')
            text = 'right'
            lis.append(text)

        elif left == 3:
            print('Looking up')
            text = 'up'
            lis.append(text)

    return lis


detector = dlib.get_frontal_face_detector()
path = os.getcwd() + '/models/shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(path)

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]
cap = cv2.VideoCapture(0)
ret, img = cap.read()
thresh = img.copy()

cv2.namedWindow('image')
kernel = np.ones((9, 9), np.uint8)


def nothing(x):
    pass


t1 = datetime.datetime.now()
print("Local Time: ", datetime.datetime.now())

print("Pengambilan vidio akan segera dimulai , arahkan kedua mata ke kamera")


cv2.createTrackbar('threshold', 'image', 75, 255, nothing)

lis = []
words = []


blink_counter = 0
previous_ratio = 100

while (True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    for rect in rects:
        shape = predictor(gray, rect)

        left_eye_ratio = get_EAR(left, shape)
        right_eye_ratio = get_EAR(right, shape)

        blinking_ratio = (left_eye_ratio + right_eye_ratio) / 2
        blinking_ratio_1 = blinking_ratio * 100
        blinking_ratio_2 = np.round(blinking_ratio_1)
        blinking_ratio_rounded = blinking_ratio_2 / 100
        if blinking_ratio < 0.20:
            if previous_ratio > 0.20:
                blink_counter = blink_counter + 1
                print('Blink')
                lis.append('Blink')

        previous_ratio = blinking_ratio

        shape = shape_to_np(shape)
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        mask, end_points_left = eye_on_mask(mask, left, shape)
        mask, end_points_right = eye_on_mask(mask, right, shape)
        mask = cv2.dilate(mask, kernel, 5)

        eyes = cv2.bitwise_and(img, img, mask=mask)
        mask = (eyes == [0, 0, 0]).all(axis=2)
        eyes[mask] = [255, 255, 255]
        mid = int((shape[42][0] + shape[39][0]) // 2)
        eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
        threshold = cv2.getTrackbarPos('threshold', 'image')
        _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        thresh = process_thresh(thresh)
        eyeball_pos_left = contouring(
            thresh[:, 0:mid], mid, img, end_points_left)
        eyeball_pos_right = contouring(
            thresh[:, mid:], mid, img, end_points_right, True)

        dic = {'left left left': 'bey wan dekk ??',
               }

        if (len(lis) > 3):
            lis = []

        word = print_eye_pos(img, eyeball_pos_left, eyeball_pos_right, lis)

        if (len(word) == 3):
            words.append(' '.join(word))
            print('words =', words)
            text = search_word(words, dic)
            word.clear()
            cv2.putText(img, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1)
            print(text)
            speech(text, 'en')

        words.clear()

        for (x, y) in shape[36:48]:
            cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

    cv2.imshow('eyes', img)
    cv2.imshow("image", thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
