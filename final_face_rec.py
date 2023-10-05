import cv2
import numpy as np
import face_recognition as face_rec
import os
from datetime import datetime
import pyttsx3

talk=pyttsx3.init()


# resize function
def resize(img,size):
    width=int(img.shape[1]*size)
    height=int(img.shape[0]*size)
    dimension=(width,height)
    return cv2.resize(img,dimension, interpolation=cv2.INTER_AREA)

path="student_images"
student_image=[]
student_name=[]
mylist=os.listdir(path)
print(mylist)

for i in mylist:
    curr_img=cv2.imread(f"{path}/{i}")
    student_image.append(curr_img)
    student_name.append(os.path.splitext(i)[0])

print(student_name)

def find_encodings(images):
    encoding_list=[]
    for img in images:
        img=resize(img,0.50)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encoding=face_rec.face_encodings(img)[0]
        encoding_list.append(encoding)
    return encoding_list

def attendence(name):
    with open("attendence.csv","r+") as f:
        datalist=f.readlines()
        namelist=[]   #class me jo present honge unke naam excel sheet me
        for i in datalist:
            entry=i.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            
            now=datetime.now()
            time=now.strftime("%H:%M")
            f.writelines(f"\n{name},{time}")
            # talk ka refernece hai pyttsx3
            welcome_statemet=str("welcome to the class hope you have a great day ahead"+name)
            talk.say(welcome_statemet)
            talk.runAndWait()
        

encoded_list=find_encodings(student_image)
vid=cv2.VideoCapture(0)
while True:
    success,frame=vid.read()
    frames=cv2.resize(frame,(0,0),None,0.25,0.25)
    frames=cv2.cvtColor(frames,cv2.COLOR_BGR2RGB)
    
    faces_in_frame=face_rec.face_locations(frames)
    encode_in_frame=face_rec.face_encodings(frames,faces_in_frame)     
    
    for encodeface,faceloc in zip(encode_in_frame,faces_in_frame):
        matches=face_rec.compare_faces(encoded_list,encodeface)  
        facedistance=face_rec.face_distance(encoded_list,encodeface)
        print(facedistance)
        
        matchIndex=np.argmin(facedistance)
        
        # agr face match hota hai
        if matches[matchIndex]:
            name=student_name[matchIndex].upper()
            y1,x2,y2,x1=faceloc
            y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(frame,(x1,y2-25),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            attendence(name)
        
    cv2.imshow("video",frame)
    cv2.waitKey(1)
        
            
        