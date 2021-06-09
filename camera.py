import cv2

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_recognizer1 = cv2.face.LBPHFaceRecognizer_create()
face_recognizer1.read("lbph_saksham.yml")
face_recognizer2 = cv2.face.LBPHFaceRecognizer_create()
face_recognizer2.read("lbph_yash.yml")
width, height = 128, 128
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = face_detector.detectMultiScale(image_gray, minSize=(100, 100),
                                                minNeighbors=5)

    # Draw a rectangle around the faces
    for (x, y, w, h) in detections:
        image_face = cv2.resize(image_gray[y:y + w, x:x + h], (width, height))
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        ids1, confidence1 = face_recognizer1.predict(image_face)
        ids2, confidence2 = face_recognizer2.predict(image_face)

        name = ""
        if confidence1 > confidence2:
            name = "Saksham"
        elif confidence1 < confidence2:
            name = "Yash"

        cv2.putText(frame, name, (x, y +(w+30)), font, 2, (0, 0, 255))
        cv2.putText(frame, str(max(confidence1, confidence2)), (x, y +(h+50)), font, 1, (0, 0, 255))
    # Display the resulting frame
    cv2.imshow("Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
camera.release()
cv2.destroyAllWindows()