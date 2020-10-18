import numpy as np
import cv2, os
import tensorflow as tf
from keras.models import load_model
from django.conf import settings
import math
import dlib
from sklearn.preprocessing import StandardScaler
from numpy import load,  asarray, sum, sqrt, multiply
import math

from PIL import Image
from cheating import pre_processing_facenet as facenet


#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
labels_emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

#0=exchange Paper, 1=looking at friend, 2=Talking Friend, 3=Use Cheat Sheet, 4=No Cheat
labels_cheats = ("exchange Paper" , "looking at friend" , "Talking Friend", "Use Cheat Sheet", "No Cheat")


def image_processing(img):
    # Convert image to grayscale image
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Modify image contrast
    #B, G, R = cv2.split(img)
    #R = cv2.equalizeHist(R.astype(np.uint8))
    #G = cv2.equalizeHist(G.astype(np.uint8))
    #B = cv2.equalizeHist(B.astype(np.uint8))
    
    #img = cv2.merge((B, G, R))
    
    # Remove noising
    img = cv2.medianBlur(img, 3)
    
    #-----Converting image to LAB Color model----------------------------------- 
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #-----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    #-----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    #-----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl,a,b))

    #-----Converting image from LAB Color model to RGB model--------------------
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    #_____END_____#
    return img 


def image_resize2(path, input_size=224, is_color=False):
    img = tf.keras.preprocessing.image.load_img(path, grayscale=not(is_color), color_mode='rgb',
                                                 target_size=(input_size, input_size),
                                                 interpolation='nearest')
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    img = image_processing(img_arr)
    img = np.array(img_arr)
    return img

def image_resize(img, img_size=(299, 299)):
    img = cv2.resize(img,img_size)
    img = np.asarray(img)
    return img



# compute face landmarks distance      

def compute_features(face_points):
    assert (len(face_points) == 68), "len(face_points) must be 68"
    
    face_points = np.array(face_points)
    features = []
    for i in range(68):
        for j in range(i+1, 68):
            features.append(np.linalg.norm(face_points[i]-face_points[j]))
            
    return np.array(features).reshape(1, -1)


def label_head(cible):
   
    if cible < 0:
        return "left"
    else:
        return "right"
  

model1 = load_model(os.path.join(settings.BASE_DIR, 'models/head_pose_model.h5'))

# head pose train
data= np.load(os.path.join(settings.BASE_DIR, 'models/head-train.npz'))
data_train = data['arr_0']
std = StandardScaler()
std.fit(data_train)

def model_head(model ,marks):
            
    features = compute_features(marks)
   
    features = std.transform(features)
    
    y_pred = model.predict(features)

    roll_pred, pitch_pred, yaw_pred = y_pred[0]
    # Get the absolute value for each number
    absValues = [math.fabs(number) for number in y_pred[0]]
    absValues = np.array(absValues)
    max_val = max(absValues)
    print(roll_pred, pitch_pred, yaw_pred)
    label = "Normal"
    if max_val == math.fabs(roll_pred): 
        label = label_head(roll_pred)
    
    if max_val == math.fabs(yaw_pred): 
        label = label_head(yaw_pred)
        
    return label

######### fin head


########## fr debut ###############

# load facenet model
model2 = facenet.facenet_model()

# load face embeddings
data = load(os.path.join(settings.BASE_DIR, 'models/lfw-faces-embeddings.npz'))
trainX_embedding, trainy = data['arr_0'], data['arr_1']


def l2_normalize(x):
 return x / sqrt(sum(multiply(x, x)))

def euclidian_distance(list_source, cible, treshold = 0.7):
    cible = cible.reshape((cible.shape[0],1))
    cible = l2_normalize(cible)
   
    for index, embedding in enumerate(list_source):
        embedding = embedding.reshape((embedding.shape[0],1))
        embedding = l2_normalize(embedding)
        dist = embedding - cible
        dist = sum(multiply(dist,dist))
       
        dist = sqrt(dist)
        if dist <= treshold:
            print("distance euclidian verified: ", dist)
            return True, index
        
    print("distance euclidian unverified: ", dist)
    return False, "UNKNOWN"

def findCosineSimilarity(list_source, cible, treshold = 0.09):
     cible = cible.reshape((cible.shape[0],1))
     
     for index, embedding in enumerate(list_source):
        
         embedding = embedding.reshape((embedding.shape[0],1))
        
         a = np.matmul(np.transpose(embedding), cible)
         b = np.sum(np.multiply(embedding, embedding))
         c = np.sum(np.multiply(cible, cible))
         dist = 1 - (a / (np.sqrt(b) * np.sqrt(c)))
         if dist <= treshold:
            print("Cosinus distance verified: ", dist)
            return [True, index]
        
     print("Cosinus distance unverified: ", dist)
     return [False, "UNKNOWN"]
        


def model_fr(model, face):

    name = "UNKNOWN"
    img = Image.fromarray(face)
    img = facenet.image_resize(img)
    embedding = facenet.face_embedding(model, img)
    results = findCosineSimilarity(trainX_embedding, embedding)
    #print(results)
    if results[0] == True:
        name = str(trainy[results[1]])
        #print(name)
    
    return name

############ fin fr #######################

####### debut emotions ###########

#0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
labels_emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

model3 = load_model(os.path.join(settings.BASE_DIR, 'models/ER_model.h5'))

def model_ER(model, face):
    
    detected_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) #transform to gray scale
    detected_face = cv2.resize(detected_face, (48, 48)) #resize to 48x48
    
    img_pixels = np.array(detected_face)
    img_pixels = np.expand_dims(img_pixels, axis = 0)
    img_pixels = img_pixels.reshape(img_pixels.shape[0], 48, 48, 1)
    img_pixels = img_pixels.astype('float32')
    
    img_pixels /= 255
    
    predictions = model.predict(img_pixels)
    
    #find max indexed array
    max_index = np.argmax(predictions[0])
    
    emotion = labels_emotions[max_index]
   
    return emotion

##### emotion fin ############

######## Cheat debut #######################

#0=exchange Paper, 1=looking at friend, 2=Talking Friend, 3=Use Cheat Sheet, 4=No Cheat
labels_cheats = ("exchange Paper" , "looking at friend" , "Talking Friend", "Use Cheat Sheet", "No Cheat")

model4 = load_model(os.path.join(settings.BASE_DIR, 'models/H_A_R_model_xception.hdf5'))

def image_resize(img, img_size=(299, 299)):
    img = cv2.resize(img,img_size)
    img = asarray(img)
    return img


def model_cheat(model, frame):
    frame = image_resize(frame)
    #print(test_img.shape)
    test_img= np.expand_dims(frame, axis=0)
    test_img = test_img.astype('float32')
    test_img /= 255
    #print(test_img.shape)
    pred = model.predict(test_img)

    pred_res = pred[0]
    pred_max = max(pred_res)

    # index
    indice_max = -1
    label = "None"
    for i in range(len(pred_res)):
        if pred_res[i] == pred_max:
            indice_max = i
            label = labels_cheats[indice_max]
    #print(indice_max)
    
    return label

########### fin cheat ##########################







