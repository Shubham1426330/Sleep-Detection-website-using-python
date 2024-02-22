# Import necessary libraries
import math
import cv2
import dlib
from playsound import playsound
import time

# Define a function to calculate the Euclidean distance between two points
def distance(a1, b1, a2, b2):
    x = abs(a2 - a1) * abs(a2 - a1)
    y = abs(b2 - b1) * abs(b2 - b1)
    return math.sqrt(x + y)

# Open the video capture device (0 represents the default camera)
cap = cv2.VideoCapture(0)

# Initialize face detector using Histogram of Oriented Gradients (HOG)
hog_face_detector = dlib.get_frontal_face_detector()

# Load the pre-trained facial landmarks predictor
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize an empty list to store detected areas
Detection = []

# Main loop for video processing
while True:
    # Read a frame from the video feed
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = hog_face_detector(gray)
    
    # Loop through detected faces
    for face in faces:
        # Get facial landmarks for the current face
        face_landmarks = dlib_facelandmark(gray, face)
        
        # Extract coordinates of eye landmarks for the first eye (left eye)
        E1 = []
        n = 37
        while n < 42:
            # Get coordinates of the current landmark
            x1 = face_landmarks.part(n).x
            y1 = face_landmarks.part(n).y
            # Draw a circle at the landmark position
            cv2.circle(frame, (x1, y1), (1), (0, 255, 255), 1)
            E1.append(x1)
            E1.append(y1)
            
            # Draw a line connecting consecutive landmarks
            if n < 38:
                a = n + 1
                x2 = face_landmarks.part(a).x
                y2 = face_landmarks.part(a).y                
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            
            # Special case for connecting last landmark to the first one
            if n == 38:
                a = 40
                x2 = face_landmarks.part(a).x
                y2 = face_landmarks.part(a).y                
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            
            # Draw lines for the remaining landmarks
            if n > 39:
                a = n + 1
                if n == 41:
                    a = 37
                x2 = face_landmarks.part(a).x
                y2 = face_landmarks.part(a).y                
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
              
            # Increment the landmark index
            n = n + 1
            if n == 39:
                n = n + 1
        
        # Calculate the area of the first eye using the distance function
        L = distance(E1[0], E1[1], E1[2], E1[3])
        B = distance(E1[4], E1[5], E1[6], E1[7])
        print("L:", L)
        print("B:", B)
        A1 = L * B
        print("Area of eye 1:", A1)
        
        # Extract coordinates of eye landmarks for the second eye (right eye)
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
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            
            if n == 44:
                a = 46
                x2 = face_landmarks.part(a).x
                y2 = face_landmarks.part(a).y                
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            
            if n > 45:
                a = n + 1
                if n == 47:
                    a = 43
                x2 = face_landmarks.part(a).x
                y2 = face_landmarks.part(a).y                
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
            
            n = n + 1
            if n == 45:
                n = n + 1
        
        # Calculate the area of the second eye
        L = distance(E2[0], E2[1], E2[2], E2[3])
        B = distance(E2[4], E2[5], E2[6], E2[7])
        print("L:", L)
        print("B:", B)
        A2 = L * B
        print("Area of eye 2:", A2)
        
        # Calculate the average area of both eyes
        Ave_area = (A1 + A2) / 2
        print("\n\nAverage area:", Ave_area)
        print("\n\n")
        
        # Add the average area to the list of detections
        Detection.append(Ave_area)
        print(Detection)
        i = len(Detection)
        print(i)
        
        # Check if enough detections are available for making a decision
        if i > 6:
            value = Detection[i-1] + Detection[i-4] + Detection[i-5]
            print(value, "\n")
            Average = value / 3
            print(Average, "\n")
            
            # If the average area is below a threshold, play an alarm sound
            if Average < 134:
                playsound("alarm.mp3")
                print("Hi you closed your eyes!")
    
    # Display the frame with facial landmarks
    cv2.imshow("Face Landmarks", frame)
    
    # Check for the 'ESC' key to exit the loop
    key = cv2.waitKey(1)
    if key == 27:
        break
    
    # Introduce a delay to make the video feed more manageable
    time.sleep(0.2)

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
