from imageai.Detection.Custom import CustomObjectDetection
from tkinter import Tk, filedialog

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("detection_model-ex-035--loss-0002.056.h5")
detector.setJsonPath("detection_config.json")
detector.loadModel()
root = Tk()
root.filename = filedialog.askopenfilename(initialdir='Downloads', title= "Select an Oil_Filter", filetypes= (("jpg files","*.jpg"),("all files","*.*")))
print (root.filename)
detections = detector.detectObjectsFromImage(input_image=root.filename, output_image_path="8-detected.jpeg")
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
