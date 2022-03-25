from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib.mailer import Mailer
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest

#0.00987987987
#python run.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel 

t0 = time.time()

def run():

	
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=False,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	
	ap.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip frames between detections")
	args = vars(ap.parse_args())

	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	if not args.get("input", False): # IF you find a camera, webcam, IP cameera, video
		print("[INFO] Starting the live stream..")
		vs = VideoStream(config.url).start()
		time.sleep(2.0)

	else:
		print("[INFO] Starting the video..") #if you have a video file, use this
		vs = cv2.VideoCapture(args["input"])


	writer = None
	frame_width = int(vs.get(3))
	frame_height = int(vs.get(4))
	
	size = (frame_width, frame_height)
	# #result = cv2.VideoWriter('filename.avi', 
    #                      cv2.VideoWriter_fourcc(*'MJPG'),
    #                      10,size)
	#result = cv2.VideoWriter('output.avi', -1, 20.0, (640,480))
	fourcc = cv2.VideoWriter_fourcc(*'MJPG')
	print("size",size)
	result = cv2.VideoWriter('ossudtput.avi', fourcc, 20.0, size)
	W = None
	H = None
	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}
	totalFrames = 0
	totalDown = 0
	totalUp = 0
	x = []
	empty=[]
	empty1=[]

	fps = FPS().start()

	if config.Thread:
		vs = thread.ThreadingClass(config.url)

	while True: # Infinite Loop
		# 60 fps ->
		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame

		if args["input"] is not None and frame is None:
			break

        # Change Width here if CPU is low
		frame = imutils.resize(frame, width = 500) # Width of the frame
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		#result.release()
		# Init Dimensions
		if W is None or H is None:
			(H, W) = frame.shape[:2]
   #frame = image

		# Output writer
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)

		# Status Init
		status = "Waiting"
		rects = []

		# slow PC?????
		if totalFrames % args["skip_frames"] == 0:
			
			status = "Detecting"
			trackers = []
			# blob creation
   # 00101010101010101010 
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
   # iamge -> pixels -> detections -> mapped
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associate with the prediction
				confidence = detections[0, 0, i, 2]
				# confidence
				if confidence > args["confidence"]:
					# extract the index of the class label from detection 
					idx = int(detections[0, 0, i, 1])

					if CLASSES[idx] != "person":
						continue #get out of the [position]

					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")

					# construct a dlib rectangle object from the bounding box coordinates and start tracker
					tracker = dlib.correlation_tracker()
     
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					# add the tracker to our list of trackers so we can utilize it during skip frames
					trackers.append(tracker)

		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"

				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

		# draw a horizontal line in the center of the frame -- once an
		# object crosses this line we will determine whether they were
		# moving 'up' or 'down'
		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
		# #cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
		# 	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the y-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'up' and positive for 'down')
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)

				# check to see if the object has been counted or not
				if not to.counted:
					# if the direction is negative (indicating the object
					# is moving up) AND the centroid is above the center
					# line, count the object
					if direction < 0 and centroid[1] < H // 2:
						totalUp += 1
						empty.append(totalUp)
						to.counted = True

					# if the direction is positive (indicating the object
					# is moving down) AND the centroid is below the
					# center line, count the object
					elif direction > 0 and centroid[1] > H // 2:
						totalDown += 1
						empty1.append(totalDown)
						#print(empty1[-1])
						# if the people limit exceeds over threshold, send an email alert
						if sum(x) >= config.Threshold:
							cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
								cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
							if config.ALERT:
								print("[INFO] Sending email alert..")
								Mailer().send(config.MAIL)
								print("[INFO] Alert sent")

						to.counted = True
						
					x = []
					# compute the sum of total people inside
					x.append(len(empty1)-len(empty))
					#print("Total people inside:", x)


			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		# construct a tuple of information we will be displaying on the
		doers="Phalcon"
		info = [
        ("Project By",doers),
		("Exit", totalUp),
		("Enter", totalDown),
		("Status", status),
		
		]

		info2 = [
		("Total people inside", x),
		]

                # Display the output
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

		for (i, (k, v)) in enumerate(info2):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		# Initiate a simple log to save data at end of the day
		if config.Log:
			datetimee = [datetime.datetime.now()]
			d = [datetimee, empty1, empty, x]
			export_data = zip_longest(*d, fillvalue = '')

			with open('Log.csv', 'w', newline='') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				wr.writerow(("End Time", "In", "Out", "Total Inside"))
				wr.writerows(export_data)
				
		# check to see if we should write the frame to disk
		if writer is not None:
			writer.write(frame)
		result.write(frame)
		# show the output frame
		cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# increment the total number of frames processed thus far and
		# then update the FPS counter
		totalFrames += 1
		fps.update()

		if config.Timer:
			# Automatic timer to stop the live stream. Set to 8 hours (28800s).
			t1 = time.time()
			num_seconds=(t1-t0)
			if num_seconds > 28800:
				break

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


	# # if we are not using a video file, stop the camera video stream
	# if not args.get("input", False):
	# 	vs.stop()
	#
	# # otherwise, release the video file pointer
	# else:
	# 	vs.release()
	
	# issue 15
	if config.Thread:
		vs.release()
	result.release()
	# close any open windows
	cv2.destroyAllWindows()


##learn more about different schedules here: https://pypi.org/project/schedule/
if config.Scheduler:
	##Runs for every 1 second
	#schedule.every(1).seconds.do(run)
	##Runs at every day (9:00 am). You can change it.
	schedule.every().day.at("9:00").do(run)

	while 1:
		schedule.run_pending()

else:
	run()
