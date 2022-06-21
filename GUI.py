from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import Open
import tkinter 

import cv2
import PIL.Image,PIL.ImageTk
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam

model_architecture = "RecognizeWood_256.json"
model_weights = "RecognizeWood_256.h5"
model = model_from_json(open(model_architecture).read())
model.load_weights(model_weights)
opt = Adam(learning_rate=0.0008)
model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics=['accuracy'])
labels = ["Anh dao","Bach dang","Bo de","Cam xe","Cao su","Cate","Hoang dan","Keo lai","Lat hoa","Mo","Mun","Muong den","Que","Soi","Tau mat",""]

fl = ""
onCamera = 0
imgtest = np.zeros((1,256,256), dtype=np.float32)
result = labels[np.argmax(model.predict(imgtest))]

def Image_Recognization():
    global onCamera
    if onCamera ==1:
        global cap
        onCamera = 0
        cap.release()
    global imgshow
    global GUI_Imageshow
    ftypes = [('Images', '*.jpg *.tif *.bmp *.gif *.png')]
    dlg = Open( filetypes = ftypes)
    fl = dlg.show()    
    if fl!= "":
        imgtest = np.zeros((1,256,256), dtype=np.float32)
        imgtest[0][::][::] = cv2.resize(cv2.imread(fl,cv2.IMREAD_GRAYSCALE),[256,256])/255.0
        result = labels[np.argmax(model.predict(imgtest))]
        imgshow = cv2.resize(cv2.imread(fl,cv2.IMREAD_COLOR),(600,450))
        imgshow = cv2.putText(imgshow,result,(20,40),cv2.FONT_HERSHEY_SIMPLEX,fontScale = 1,color = (255,0,0),thickness = 2)
        imgshow = cv2.cvtColor(imgshow,cv2.COLOR_BGR2RGB)
        imgshow = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(imgshow))
        GUI_Imageshow.create_image(0,0,image=imgshow,anchor=tkinter.NW)

def Camera_Recognization():
    global onCamera
    global cap
    onCamera = 1
    cap = cv2.VideoCapture(0)

root = Tk()
root.title("Wood Recognize App")
root.geometry("620x470")
menubar = Menu(root)
root.config(menu=menubar) 
fileMenu = Menu(menubar)
menubar.add_cascade(label="Mode", menu=fileMenu)
fileMenu.add_command(label="Image", command=Image_Recognization)
fileMenu.add_command(label="Camera", command=Camera_Recognization)
btnOn = Button(root,text="ON CAMERA",command=onCamera)
GUI_Imageshow = Canvas(root,width=600,height=450,bg="white")
GUI_Imageshow.place(x=10,y=10)
def update_camera():
    global imgshow
    global GUI_Imageshow
    global cap
    if onCamera == 1:
        ret, frame = cap.read()
        if ret == True:
            imgtest[0][::][::] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),[256,256])/255.0
            result = labels[np.argmax(model.predict(imgtest))]
            imgshow = cv2.resize(frame,(600,450))
            imgshow = cv2.putText(imgshow,result,(20,40),cv2.FONT_HERSHEY_SIMPLEX,fontScale = 1,color = (255,0,0),thickness = 2)
            imgshow = cv2.cvtColor(imgshow,cv2.COLOR_BGR2RGB)
            imgshow = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(imgshow))
            GUI_Imageshow.create_image(0,0,image=imgshow,anchor=tkinter.NW)
    root.after(40,update_camera)
update_camera()
root.mainloop()

