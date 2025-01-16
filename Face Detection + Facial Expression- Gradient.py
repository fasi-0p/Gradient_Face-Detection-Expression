import cv2
import mediapipe as mp

mpmesh=mp.solutions.face_mesh
mesh=mpmesh.FaceMesh(max_num_faces=2) 
mpDraw=mp.solutions.drawing_utils
drawspec=mpDraw.DrawingSpec(thickness=1, circle_radius=0) #to calibrate node's size

cap=cv2.VideoCapture(0)

while True:
    success,img=cap.read()
    if not success:
        print('Ignoring Camera')
        continue
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=mesh.process(imgRGB)
    
    if(result.multi_face_landmarks):
        for i in result.multi_face_landmarks:

            for id,lm in enumerate(i.landmark):
                h,w,c=img.shape
                cx,cy= int(lm.x*w), int(lm.y*h)

                #print(f"lower lip y:{int(i.landmark[14].y*img.shape[1])}  ||  Upper Lip y:{int(i.landmark[11].y*img.shape[1])}")  #to find coordinate difference of landmarks
                if( i.landmark[61].y*img.shape[1] <i.landmark[11].y*img.shape[1]> i.landmark[291].y*img.shape[1] ):
                    print('Smiling')
                if(15<(i.landmark[14].y*img.shape[1] - i.landmark[11].y*img.shape[1])<30):
                    print('\t','Laughing')
                if(25<(i.landmark[14].y*img.shape[1] - i.landmark[11].y*img.shape[1])<35):
                    print('\t','\t','Surprise')
                

            mpDraw.draw_landmarks(img,i,mpmesh.FACEMESH_TESSELATION,drawspec,drawspec)
            cv2.imshow('camera feed',img)

        if(cv2.waitKey(1) & 0xFF == ord('q')):
            break
cap.release()
cv2.destroyAllWindows()