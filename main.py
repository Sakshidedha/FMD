# Import necessary libraries
import streamlit as st
from keras.models import load_model#To load the pre-trained mask detection model
from keras.utils import load_img, img_to_array  # For image preprocessing
import numpy as np  # For numerical operations
import cv2 # For image processing and face detection
import tempfile

# Load face detection model and mask detection mode
facemodel=cv2.CascadeClassifier("face.xml")
maskmodel=load_model('mask.h5')

st.set_page_config(page_title="Face Mask Detection System", page_icon="https://encrypted-tbn0.gstatic.com/images?q.JPG=tbn:ANd9GcRybwVSLyTqJ_wWE24LK6ZZiBR4B61pJ11wbR6PWio_nqk5AzC-cS_aehcwXDgAPB3GshM&usqp=CAU")
st.title("FACE MASK DETECTION SYSTEM") # Set the title for the web application
st.image("https://5.imimg.com/data5/PI/FD/NK/SELLER-5866466/images-500x500.jpg", width=500)
choice=st.sidebar.selectbox("MY MENU",["HOME","IMAGE","VIDEO","CAMERA"])# Sidebar menu for navigation

if(choice=="HOME"): #Instruction on Home Section
    st.write("""Welcome to the Face Mask Detection System.This application enables automated detection of face masks from various input sources, supporting images, videos, and IP cameras for flexible and real-time monitoring.""")

    st.subheader("How to Use:")

    st.markdown("""
    - **Upload an Image**  
     Upload an image from your device. The system will analyze and detect faces, indicating whether each person is wearing a mask.

    - **Upload a Video**   
     Upload a video file to perform frame-by-frame face mask detection. Results will be displayed with bounding boxes and labels.

    - **Use an IP Camera**  
     Connect to an IP camera by entering the camera’s URL. The system will stream and analyze live footage for face mask detection.""")

elif(choice=="IMAGE"):
    file=st.file_uploader("Upload Image")
    
    if file:
        b=file.getvalue()# Convert the uploaded image into an array
        d=np.frombuffer(b,np.uint8)
        img=cv2.imdecode(d,cv2.IMREAD_COLOR)# Decode the image
        face=facemodel.detectMultiScale(img)# Detect faces in the uploaded image
        
        for(x,y,w,h) in face:  # Iterate over each detected face
            crop_face=img[y:y+h,x:x+w]
            cv2.imwrite('temp.jpg',crop_face)
             # Load the cropped face image for prediction
            crop_face=load_img('temp.jpg',target_size=(150,150,3))
            crop_face=img_to_array(crop_face)
            crop_face=np.expand_dims(crop_face,axis=0)
            pred=maskmodel.predict(crop_face)[0][0]
            
            if pred==1:  # Draw a rectangle around the face
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4) # Red rectangle for no mask
            else:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),4)  # Green rectangle for no mask
                
        st.image(img,channels='BGR',width=400)         # Display the resulting image with rectangles around faces

elif(choice=="VIDEO"):
    file=st.file_uploader("Upload Video") # Upload a video file
    window = st.empty() # Create an empty space for displaying video frames
    
    if file:
        tfile=tempfile.NamedTemporaryFile()# Save uploaded video temporarily
        tfile.write(file.read())
        vid=cv2.VideoCapture(tfile.name)# Load video from temporary file
        i=1
        
        while(vid.isOpened()):  # Read video frame by frame
            flag, frame=vid.read()
            if flag:
                face=facemodel.detectMultiScale(frame)  # Detect faces in the frame
                
                for (x, y, l, w) in face: 
                    crop_face1=frame[y:y+w, x:x+l] #Crop detected face region
                    cv2.imwrite('temp.jpg', crop_face1) # Save temporary cropped face
                    # Preprocess face for prediction
                    crop_face=load_img('temp.jpg', target_size=(150, 150, 3)) 
                    crop_face=img_to_array(crop_face)
                    crop_face=np.expand_dims(crop_face, axis=0)
                    pred=maskmodel.predict(crop_face)[0][0]
                    
                    if pred==1:  # Predict mask (1 = No Mask, 0 = Mask)
                        cv2.rectangle(frame, (x, y), (x+l, y+w), (0, 0, 255), 4)
                        path = "data/" + str(i) + ".jpg"
                        cv2.imwrite(path, crop_face1) # Save face with no mask
                        i +=1
                    else:
                         #Draw green rectangle for mask
                        cv2.rectangle(frame, (x, y), (x+1, y+w), (0, 255, 0), 4)
                
                window.image(frame, channels='BGR')# Display the frame with rectangles
                                
elif(choice=="CAMERA"): # Button to start the camera
    btn=st.button("Start Camera")
    window=st.empty()   # Empty container for displaying frames
    btn2=st.button("Stop camera") # Button to stop the camera
    if btn2: # If 'Stop camera' is clicked, rerun the script to stop video capture
        st.rerun()
    if btn: # If 'Start Camera' is clicked
        vid=cv2.VideoCapture("https://192.168.29.13:8080/video")
        i=1  # Counter for saving images
        while(vid.isOpened()):  # Loop until the video is open
            flag, frame = vid.read()# Read a frame from the camera

            if flag:   # Detect faces in the frame

                face=facemodel.detectMultiScale(frame)

                for (x, y, l, w) in face:# Crop the detected face from the frame
                    crop_face1=frame[y:y+w, x:x+l]
                    cv2.imwrite('temp.jpg', crop_face1) # Save the cropped face temporarily

                    # Load and preprocess the cropped face for prediction
                    crop_face=load_img('temp.jpg', target_size=(150, 150, 3))
                    crop_face=img_to_array(crop_face)
                    crop_face=np.expand_dims(crop_face, axis=0)

                    pred=maskmodel.predict(crop_face)[0][0]# Predict mask status (1 = No Mask, 0 = Mask)

                    if pred==1:
                        cv2.rectangle(frame, (x, y), (x+l, y+w), (0, 0, 255), 4) # Draw red rectangle for no mask
                        path = "data/" + str(i) + ".jpg"# Save the face image with no mask in 'data' folder
                        cv2.imwrite(path, crop_face1)
                        i +=1
                    else:
                        cv2.rectangle(frame, (x, y), (x+l, y+w), (0, 255, 0), 4)# Draw green rectangle for mask

                window.image(frame, channels='BGR')  # Display the video frame with rectangles









        
