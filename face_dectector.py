import cv2

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# This will take in the image.
img = cv2.imread('dadi.jpg') # Change it's name to the image you want to detect...

# This will set the image to grayscaled_image
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# This will detect faces. I will do this using the cv2 xml file.
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)

# This will Print out the image.
cv2.imshow('Face Detection', img)
cv2.waitKey()

print("Program Done")