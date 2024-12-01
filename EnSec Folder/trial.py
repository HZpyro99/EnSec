import cv2
import numpy as np
import os
import sys

def read_images(path, sz=None):
  """
  Reads images from a directory structure where each subfolder represents a class (person)

  Args:
      path: Path to the directory containing subfolders for each class.
      sz: Optional size for resizing images.

  Returns:
      A list of images (X) and a list of corresponding labels (y).
  """
  c = 0
  X, y = [], []

  for dirname, dirnames, filenames in os.walk(path):
    # Skip the main directory
    if dirname == path:
      continue

    label = int(os.path.basename(dirname))  # Extract class label from subfolder name

    for filename in filenames:
      if filename.endswith(('.jpg', '.png', '.jpeg')):
        try:
          filepath = os.path.join(dirname, filename)
          im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

          if sz is not None:
            im = cv2.resize(im, (sz, sz))

          X.append(np.asarray(im, dtype=np.uint8))
          y.append(label)

        except Exception as e:
          print(f"Error reading {filename}: {e}")

  return [X, y]

def face_rec():
  # Replace with actual names corresponding to your folder names
  names = ["Renz", "Evans", "Ivan"]

  image_path = sys.argv[1] if len(sys.argv) > 1 else 'D:/Docs/J/EnSec Folder/dataset'

  if not os.path.exists(image_path):
    print(f"Error: Image path '{image_path}' does not exist.")
    sys.exit()

  [X, y] = read_images(image_path)
  y = np.asarray(y, dtype=np.int32)

  # Create EigenFaceRecognizer
  model = cv2.face.EigenFaceRecognizer_create()
  model.train(X, y)

  camera = cv2.VideoCapture(1)
  face_cascade = cv2.CascadeClassifier('C:/Users/Ivan Klein Alegre/AppData/Roaming/Python/Python313/site-packages/cv2/data/haarcascade_frontalface_default.xml')

  while True:
    ret, img = camera.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
      cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
      roi = gray[y:y+h, x:x+w]
      roi = cv2.resize(roi, (200, 200))

      label, confidence = model.predict(roi)

      # Set a threshold for confidence
      threshold = 70

      if confidence < threshold:
        if 0 <= label < len(names):
          name = names[label]
          cv2.putText(img, f"{name} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
          print(f"Warning: Predicted label {label} is out of range.")
          cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  camera.release()
  cv2.destroyAllWindows()

if __name__ == "__main__":
  face_rec()