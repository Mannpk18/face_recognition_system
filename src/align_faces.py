import os
import cv2
import dlib

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("src/shape_predictor_68_face_landmarks.dat")

def align_face(image):
    faces = detector(image, 1)
    if len(faces) == 0:
        return None
    face = faces[0]
    shape = predictor(image, face)
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cropped = image[y:y+h, x:x+w]
    return cv2.resize(cropped, (160, 160))

def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for person in os.listdir(input_dir):
        person_dir = os.path.join(input_dir, person)
        out_person_dir = os.path.join(output_dir, person)
        os.makedirs(out_person_dir, exist_ok=True)

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            aligned = align_face(img)
            if aligned is not None:
                cv2.imwrite(os.path.join(out_person_dir, img_name), aligned)

if __name__ == "__main__":
    process_images("C:/Users/pinal/Downloads/face_recognition_project/data", 
               "C:/Users/pinal/Downloads/face_recognition_project/aligned")

