import urllib.request

url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
filename = "shape_predictor_68_face_landmarks.dat.bz2"

print("Downloading model...")
urllib.request.urlretrieve(url, filename)
print("Extracting...")
import bz2
with bz2.BZ2File(filename) as fr, open("shape_predictor_68_face_landmarks.dat", "wb") as fw:
    fw.write(fr.read())
print("Done.")
