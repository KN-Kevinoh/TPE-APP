""" 
    In this module, we perform face detector and schedule ours models 
    over each frame through video streaming.
    For face detector, we caffe dnn the most powerful between MTCNN, Hardcascase, dlib
"""

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import tensorflow as tf
import imutils
import cv2,os
import numpy as np
import time
from threading import Thread
from django.conf import settings
from .utils import image_processing, model_head, model_fr, model_ER, model_cheat, model1, model2, model3, model4

######### implements thread
import sys
from builtins import super    # https://stackoverflow.com/a/30159479

if sys.version_info >= (3, 0):
    _thread_target_key = '_target'
    _thread_args_key = '_args'
    _thread_kwargs_key = '_kwargs'
else:
    _thread_target_key = '_Thread__target'
    _thread_args_key = '_Thread__args'
    _thread_kwargs_key = '_Thread__kwargs'

class ThreadSchedule(Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._return = None

    def run(self):
        target = getattr(self, _thread_target_key)
        if not target is None:
            self._return = target(
                *getattr(self, _thread_args_key),
                **getattr(self, _thread_kwargs_key)
            )

    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self._return



# load our serialized face detector model from disk
prototxtPath = os.path.sep.join([settings.BASE_DIR, "models/face_detector/deploy.prototxt"])
weightsPath = os.path.sep.join([settings.BASE_DIR,"models/face_detector/res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
#maskNet = load_model(os.path.join(settings.BASE_DIR,'models/face_detector/mask_detector.model'))
#VIDEO_PATH = os.path.join(settings.BASE_DIR, 'media/multimedia/test_exam.mp4')


class MaskDetect(object):
	def __init__(self, VIDEO_PATH):
		
		self.vs = cv2.VideoCapture(VIDEO_PATH)
		#time.sleep(2.0)
		
  
	def __del__(self):
		#self.vs.stream.release()
		self.vs.release()
		cv2.destroyAllWindows()
	
	
	def detect_and_predict_mask(self,frame, faceNet, threshold=0.5):
		rows, cols, _ = frame.shape

		confidences = []
		faceboxes = []

		faceNet.setInput(cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
		detections = faceNet.forward()
		
		for result in detections[0, 0, :, :]:
			confidence = result[2]
			if confidence > threshold:
				x_left_bottom = int(result[3] * cols)
				y_left_bottom = int(result[4] * rows)
				x_right_top = int(result[5] * cols)
				y_right_top = int(result[6] * rows)
				confidences.append(confidence)
				faceboxes.append([x_left_bottom, y_left_bottom, x_right_top, y_right_top])

		self.detection_result = [faceboxes, confidences]

		return confidences, faceboxes

	
	def detect_marks(self, image_np):
		"""Detect marks from image"""
		model = load_model(os.path.join(settings.BASE_DIR, 'models/pose_model'))
		# # Actual detection.
		predictions = model.signatures["predict"](
			tf.constant(image_np, dtype=tf.uint8))

		# Convert predictions to landmarks.
		marks = np.array(predictions['output']).flatten()[:136]
		marks = np.reshape(marks, (-1, 2))

		return marks
     			

	def get_frame(self):
		_,frame = self.vs.read()
		#time.sleep(2.0)
		#frame.astype(np.float32)
		frame = image_processing(frame)
		frame = imutils.resize(frame, width=650)
		frame = cv2.flip(frame, 1)
  
		faceboxes =  self.detect_and_predict_mask(frame, faceNet, threshold=0.5)
		if len(faceboxes) > 0:
			for facebox in faceboxes:
				face_img = frame[facebox[1]: facebox[3],
						facebox[0]: facebox[2]]
				face_img = cv2.resize(face_img, (128, 128))
				face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
				marks = self.detect_marks([face_img])
				marks *= (facebox[2] - facebox[0])
				marks[:, 0] += facebox[0]
				marks[:, 1] += facebox[1]
				shape = marks.astype(np.uint)
				print(marks.shape)
	
				t1 = ThreadSchedule(target=model_head, args=(model1, marks))
				t2 = ThreadSchedule(target=model_fr, args=(model2, face_img))
				t3 = ThreadSchedule(target=model_ER, args=(model3, face_img))
				t4 = ThreadSchedule(target=model_cheat, args=(model4, face_img))
				
				# Threads started
				t1.start()
				t2.start()
				t3.start()
				t4.start()
				
				# waiting threads results
				results_1 = t1.join()
				results_2 = t2.join()
				results_3 = t3.join()
				results_4 = t4.join()
				
				cv2.rectangle(frame, (facebox[0], facebox[1]),
							(facebox[2], facebox[3]), (0, 255, 0), 5)
				label = results_2
				label_size, base_line = cv2.getTextSize(
					label, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)

				cv2.rectangle(frame, (facebox[0], facebox[1] + 10 - label_size[1]),
								(facebox[0] + 90 + label_size[0],
								facebox[1] - 75 + base_line),
								(0, 255, 0), cv2.FILLED)
				cv2.putText(frame, 'Name: '+ label, (facebox[0], facebox[1] - 55),
							cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
				
				cv2.putText(frame,'Head: '+ results_1, (facebox[0], facebox[1] - 35),
							cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
				
				cv2.putText(frame,'Emotion: '+ results_3, (facebox[0], facebox[1] - 15),
							cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
				
				cv2.rectangle(frame, (0, 0), (350, 40), (0, 0, 0), -1)
				cv2.putText(frame,'Activity: ' + results_4 , (10, 25), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 255, 255), 2)
			
		ret, jpeg = cv2.imencode('.jpg', frame)
		return jpeg.tobytes()
