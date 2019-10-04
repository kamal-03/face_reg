import face_recognition
import cv2,glob
import os
import numpy as np
import os.path,traceback
import pickle

f_encodings=[]
f_names=[]
directory='./padding'
c=0
try:
    for file in sorted(os.listdir(directory)):
        person_path1 = os.path.basename(file)
        person_name=person_path1.split('.')[0]
        
        
        f_names.append(person_name)
        print(c)
        person_path="./padding/"+str(person_path1)
        c=c+1
        f_image=face_recognition.load_image_file(person_path)
        # print(f_image)
        F=face_recognition.face_encodings(f_image)[0]
        f_encodings.append(F)
        # c=c+1
        print(person_name)
        
        
except:
    pass
print(len(f_encodings))


embeddings= dict(zip(f_names, f_encodings))

filename='image_db_embeddings'
outfile = open(filename,'wb')
pickle.dump(embeddings,outfile)
outfile.close()

known_face_names =embeddings.keys()


known_face_encodings = embeddings.values()

print(known_face_names)
