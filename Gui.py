from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths
import math

class Root(Tk):
	def __init__(self):
		super(Root,self).__init__()
		self.title("Size Calculator")
		self.minsize(640,400)

		self.labelFrame=ttk.LabelFrame(self,text="Open The File",height=60,width=500)
		self.labelFrame.grid(column=0,row=0,padx=20,pady=20,ipadx=0,ipady=0)

		self.info="Please upload the picture"

		self.button()

		self.laFrame=ttk.LabelFrame(self,text="Information",height=100,width=500)
		self.laFrame.grid(column=0,row=2,padx=20,pady=20,ipadx=0,ipady=0)
		self.la()

	def la(self):
		self.label1=Label(self.laFrame,text=self.info)
		self.label1.place(x=20,y=40,anchor="w")

	def button(self):
		self.button=ttk.Button(self.labelFrame,text="Browse",command=self.fileDialog)
		self.button.grid(column=0,row=1)
		self.button.place(x=10,y=20,anchor="w")
		self.fileName="No file selected"
		self.label=Label(self.labelFrame,text=self.fileName)
		self.label.place(x=100,y=20,anchor="w")
		self.upload()

	def upload(self):
		self.button=ttk.Button(self.labelFrame,text="Upload",command=self.script)
		self.button.grid(column=1,row=1)
		self.button.place(x=350,y=20,anchor="w")

	def fileDialog(self):
		self.fileName=filedialog.askopenfilename(initialdir="~/Documents/hackathon/hack_v15/",title="Select a file",filetypes=(("jpeg","*.jpeg"),("All Files","*.*")))
		self.label.configure(text=self.fileName)


	def script(self):

		# construct the argument parse and parse the arguments
		# ap = argparse.ArgumentParser()
		# ap.add_argument("-i", "--image", required=True,
		# 	help="path to the input image")
		# ap.add_argument("-w", "--width", type=float, required=True,
		# 	help="width of the left-most object in the image (in inches)")
		# args = vars(ap.parse_args())

		# load the image, convert it to grayscale, and blur it slightly
		image = cv2.imread(self.fileName)
		image = cv2.resize(image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)

		# perform edge detection, then perform a dilation + erosion to
		# close gaps in between object edges
		edged = cv2.Canny(gray, 10, 10)
		edged = cv2.dilate(edged, None, iterations=1)
		edged = cv2.erode(edged, None, iterations=1)

		# find contours in the edge map
		cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)

		# sort the contours from left-to-right and initialize the
		# 'pixels per metric' calibration variable
		(cnts, _) = contours.sort_contours(cnts)
		pixelsPerMetric = None
		orig = None

		for c in cnts:
			# if the contour is not sufficiently large, ignore it
			if cv2.contourArea(c) < 100:
				continue

			# compute the rotated bounding box of the contour
			orig = image.copy()
			box = cv2.minAreaRect(c)
			box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
			box = np.array(box, dtype="int")

			# order the points in the contour such that they appear
			# in top-left, top-right, bottom-right, and bottom-left
			# order, then draw the outline of the rotated bounding
			# box
			box = perspective.order_points(box)
			cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

			# loop over the original points and draw them
			for (x, y) in box:
				cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

			# unpack the ordered bounding box, then compute the midpoint
			# between the top-left and top-right coordinates, followed by
			# the midpoint between bottom-left and bottom-right coordinates
			(tl, tr, br, bl) = box
			(tltrX, tltrY) = (tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5
			(blbrX, blbrY) = (bl[0] + br[0]) * 0.5, (bl[1] + br[1]) * 0.5

			# compute the midpoint between the top-left and top-right points,
			# followed by the midpoint between the top-righ and bottom-right
			(tlblX, tlblY) = (tl[0] + bl[0]) * 0.5, (tl[1] + bl[1]) * 0.5
			(trbrX, trbrY) = (tr[0] + br[0]) * 0.5, (tr[1] + br[1]) * 0.5

			# draw the midpoints on the image
			cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
			cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
			cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
			cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

			# draw lines between the midpoints
			cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),
				(255, 0, 255), 2)
			cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),
				(255, 0, 255), 2)

			# compute the Euclidean distance between the midpoints
			dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
			dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

			# if the pixels per metric has not been initialized, then
			# compute it as the ratio of pixels to supplied metric
			# (in this case, inches)
			if pixelsPerMetric is None:
				pixelsPerMetric = dB / 7.87
			# compute the size of the object
			dimA = dA / pixelsPerMetric
			dimB = dB / pixelsPerMetric

			# draw the object sizes on the image
			cv2.putText(orig, "{:.1f}in".format(dimA),
				(int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (255, 255, 255), 2)
			cv2.putText(orig, "{:.1f}in".format(dimB),
				(int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
				0.65, (255, 255, 255), 2)

			# show the output image
			cv2.imshow("Image", orig)
			break



		# Specify the paths for the 2 files
		protoFile = "pose/coco/pose_deploy_linevec.prototxt"
		weightsFile = "pose/coco/pose_iter_440000.caffemodel"
		nPoints = 18
		POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]
		 
		# Read the network into Memory
		net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

		frameWidth = orig.shape[1]
		frameHeight = orig.shape[0]
		threshold = 0.1

		net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

		# Specify the input image dimensions
		inWidth = 368
		inHeight = 368

		# Prepare the frame to be fed to the network
		inpBlob = cv2.dnn.blobFromImage(orig, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)
		 
		# Set the prepared object as the input blob of the network
		net.setInput(inpBlob)

		output = net.forward()

		H = output.shape[2]
		W = output.shape[3]
		# Empty list to store the detected keypoints
		points = []
		for i in range(1,nPoints):
			if (i==9 or i==10):
				continue

			# confidence map of corresponding body's part.
			probMap = output[0, i, :, :]

			# Find global maxima of the probMap.
			minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

			# Scale the point to fit on the original image
			x = (frameWidth * point[0]) / W
			y = (frameHeight * point[1]) / H
			if prob > threshold : 
				cv2.circle(orig, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
				cv2.putText(orig, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, lineType=cv2.LINE_AA)
		 
				# Add the point to the list if the probability is greater than the threshold
				points.append((int(x), int(y)))
			else :
				points.append(None)
			if (i==11):
				break

		Shoulder = (points[4][0] - points[1][0])/pixelsPerMetric
		Length = (points[7][1] - points[1][1])/pixelsPerMetric
		Arm = (points[3][1]-points[1][1])/pixelsPerMetric
		self.info= "shoulder "+str(Shoulder)+"\nlength "+str(Length)+"\nArm-length "+str(Arm)+"\n"

		if (Shoulder < 12):
			self.info+="Your T-shirt size is Small"
		elif (Shoulder < 13):
			self.info+="Your T-shirt size is Medium"
		else:
			self.info+="Your T-shirt size is Large"

		self.label1.configure(text=self.info)
		print(self.info)

		cv2.imshow("Image",orig)
		cv2.waitKey(0)
		cv2.destroyAllWindows()



if __name__ == '__main__':
	root=Root()
	root.mainloop()

