import face_recognition
import cv2,glob
import os
import numpy as np
import os.path,traceback
import pickle
from imutils.video import WebcamVideoStream

#video_capture = cv2.VideoCapture(-1)

video_capture=WebcamVideoStream(src=-1).start()


f_encodings=[]
f_names=[]




# directory='./padding'
# c=0
# try:
#     for file in sorted(os.listdir(directory)):
#         person_path1 = os.path.basename(file)
#         person_name=person_path1.split('.')[0]
        
        
#         f_names.append(person_name)
#         print(c)
#         person_path="./padding/"+str(person_path1)
#         c=c+1
#         f_image=face_recognition.load_image_file(person_path)
#         # print(f_image)
#         F=face_recognition.face_encodings(f_image)[0]
#         f_encodings.append(F)
#         # c=c+1
#         print(person_name)
        
        
# except:
#     pass
# print(len(f_encodings))



# embeddings= dict(zip(f_names, f_encodings))

filename='image_db_embeddings'
# filename='embeddings_cropped_new'
infile = open(filename,'rb')
embeddings = pickle.load(infile)
infile.close()

keys, values = zip(*embeddings.items()) 
keys=list(keys)
values=list(values)

known_face_names =keys
known_face_encodings = values

# known_face_names =f_names

# known_face_encodings = f_encodings


# known_face_names = [
#     "Shitij",
#     "Kamal",
#     "Bharat"
# ]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
#print('known_face_encodings',known_face_encodings)


while True:
    # Grab a single frame of video
    #ret, 
    frame = video_capture.read()
    # small_frame=frame
    # Resize frame of video to 1/4 size for faster face recognition processing
    #small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)


    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]
    #rgb_small_frame = frame[:, :, ::-1]


    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)

        print('face_locations',face_locations)
        for i in face_locations:
            print("Diagonal Value:",i[1]-i[3])
        
       	print(len(face_locations))






        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        #print('face_encodings',face_encodings)
        
        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            #matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.44999997)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding,tolerance=0.449999)
            # print(matches)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            #print(face_distances)
            # print(known_face_names[face_distances)
            best_match_index = np.argmin(face_distances)
            #print(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            #print(face_names)
            print(name)
    #process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # print((top, right, bottom, left))
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        if right-left>94:
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            print(top,right,bottom,left)
        #if right-left>94:
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    # frame = cv2.resize(frame, (0, 0), fx=1.5, fy=1.5)
    #cv2.namedWindow('Video',cv2.WINDOW_NORMAL)
    cv2.imshow('Video', frame)
    

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
