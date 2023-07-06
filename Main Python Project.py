def distance(a1,b1,a2,b2):
    x = abs(a2 - a1) * abs(a2 - a1)
    y = abs(b2 - b1) * abs(b2 - b1)
    return math.sqrt(x+y)

    
import math
import cv2
import dlib
from playsound import playsound
import time


cap = cv2.VideoCapture(0)

hog_face_detector = dlib.get_frontal_face_detector()

dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
Detection = []
while True:
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = hog_face_detector(gray)
    
    for face in faces:
        face_landmarks = dlib_facelandmark(gray, face)
        
        E1 = []
        n = 37
        while n < 42 :
            x1 = face_landmarks.part(n).x
            y1 = face_landmarks.part(n).y
            cv2.circle(frame, (x1, y1), (1), (0, 255, 255), 1)
            E1.append(x1)
            E1.append(y1)
            
            if n < 38:     
                a = n + 1
                x2 = face_landmarks.part(a).x
                y2 = face_landmarks.part(a).y                
                cv2.line(frame, (x1,y1), (x2,y2), (0, 255, 255), 1)
            
            if n == 38:     
                a = 40
                x2 = face_landmarks.part(a).x
                y2 = face_landmarks.part(a).y                
                cv2.line(frame, (x1,y1), (x2,y2), (0, 255, 255), 1)
            
            if n > 39:     
                a = n + 1
                if n == 41:
                    a = 37
                x2 = face_landmarks.part(a).x
                y2 = face_landmarks.part(a).y                
                cv2.line(frame, (x1,y1), (x2,y2), (0, 255, 255), 1)
              
             
            n = n + 1
            if n == 39:
                n = n + 1
        
        
        L = distance(E1[0],E1[1],E1[2],E1[3])
        B = distance(E1[4],E1[5],E1[6],E1[7])
        print("L:", L)
        print("B:", B)
        A1 = L*B
        print("Area of eye 1:", A1)
        
            
        E2 = []
        n = 43
        while n < 48: 
            x1 = face_landmarks.part(n).x
            y1 = face_landmarks.part(n).y
            cv2.circle(frame, (x1, y1), (1), (0, 255, 255), 1)
            E2.append(x1)
            E2.append(y1)
            
            if n < 44:     
                a = n + 1
                x2 = face_landmarks.part(a).x
                y2 = face_landmarks.part(a).y                
                cv2.line(frame, (x1,y1), (x2,y2), (0, 255, 255), 1)
            
            if n == 44:     
                a = 46
                x2 = face_landmarks.part(a).x
                y2 = face_landmarks.part(a).y                
                cv2.line(frame, (x1,y1), (x2,y2), (0, 255, 255), 1)
            
            if n > 45:     
                a = n + 1
                if n == 47:
                    a = 43
                x2 = face_landmarks.part(a).x
                y2 = face_landmarks.part(a).y                
                cv2.line(frame, (x1,y1), (x2,y2), (0, 255, 255), 1)
            
            n = n + 1
            if n == 45:
                n = n + 1
        
        L = distance(E2[0],E2[1],E2[2],E2[3])
        B = distance(E2[4],E2[5],E2[6],E2[7])
        print("L:", L)
        print("B:", B)
        A2 = L*B
        print("Area of eye 2:", A2)
        
        Ave_area = (A1+A2)/2
        print("\n\nAverage area:", Ave_area)
        print("\n\n")
        
        Detection.append(Ave_area)
        print(Detection)
        i = len(Detection)
        print(i)
        if i>6:
            value = Detection[i-1]+Detection[i-4]+Detection[i-5]
            print(value, "\n")
            Average = value/3
            print(Average, "\n")
            if Average < 134:
                playsound("alarm.mp3")
                print("Hi you closed your eyes!")
               
    
    cv2.imshow("Face Landmarks", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
    time.sleep(0.2)
cap.release()

cv2.destroyAllWindows()