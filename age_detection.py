import cv2
import numpy as np

# Load pre-trained age detection model
model_path = 'D:\DOWNLOADS\haarcascade_frontalface_default.xml'
model = cv2.CascadeClassifier(model_path)

# Load age labels
age_labels = ['0-2', '4-6', '8-12', '15-20', '25-32', '38-43', '48-53', '60-100']

# Open the camera
cap = cv2.VideoCapture(0)

while True:
    # Read the current frame from the camera
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Perform face detection
    faces = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Process each detected face
    for (x, y, w, h) in faces:
        # Extract the face region of interest
        face = gray[y:y+h, x:x+w]

        # Resize the face to a fixed size (optional)
        face = cv2.resize(face, (224, 224))

        # Perform age detection prediction
        # TODO: Apply age estimation model inference here
        # You can use pre-trained deep learning models or other age estimation techniques
        
        # Example: Randomly assign an age label for demonstration purposes
        predicted_age = np.random.choice(age_labels)

        # Draw bounding box and predicted age on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'Age: {predicted_age}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame with age estimation
    cv2.imshow('Age Detection', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
