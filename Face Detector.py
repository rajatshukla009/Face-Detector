import PySimpleGUI as sg
import cv2

layout = [[sg.Image(key="-IMAGE-")],
          [sg.Text("people in picture: ", key="-TEXT-", expand_x=True, justification="c")]]

window = sg.Window("face detector", layout)

video = cv2.VideoCapture(0)
faceCASCADE = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    event, values = window.read(timeout= 0)

    if event == sg.WIN_CLOSED:
        break

    _,frame = video.read()
    graySACLE = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = faceCASCADE.detectMultiScale(graySACLE,
                                         scaleFactor=1.3,
                                         minNeighbors=7,
                                         minSize=(50,50))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    imgBYTES = cv2.imencode(".png", frame)[1].tobytes()
    window["-IMAGE-"].update(data=imgBYTES)

    window["-TEXT-"].update(f'People in Picture: {len(faces)}')

window.close()