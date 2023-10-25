# Description: This file is used to test the model on a local image from a video file.
from roboflow import Roboflow
import cv2

# access your model
rf = Roboflow(api_key="SJ3ihoEVndxBlymkSMfe")
project = rf.workspace().project("glass_bottles-rlphj")
model = project.version(1).model

# access the video file and open it
file = "C:/Users/Glenn/Desktop/ARP_TEST_VID.mp4"
vid = cv2.VideoCapture(file)

# Check if camera opened successfully
if not vid.isOpened():
    print("Error opening video stream or file")
num = 0
# Skip the first 200 frames
for x in range(0, 200):
    frame = vid.read()[1]

# Read until video is completed
while vid.isOpened():
    num += 1
    # Capture frame-by-frame
    frame = vid.read()[1]
    frame = cv2.resize(frame, (640, 640))

    print('\nlocal img:\n')
    # infer on a local image
    preds = model.predict(frame, confidence=80, overlap=30).json()
    # print('size of preds: '+str(len(preds)))
    obnum = -1

    # For each object in the image, draw a rectangle and label it with the confidence
    for obj in preds['predictions']:
        obnum += 1
        # print(obj)
        # remap the coordinates to the corners of the bounding box
        x = int(obj['x'])
        y = int(obj['y'])
        w = int(obj['width'])
        h = int(obj['height'])
        x1 = int(x-w/2)
        x2 = int(x+w/2)
        y1 = int(y-h/2)
        y2 = int(y+h/2)
        # insert the confidence into the image
        cv2.putText(frame, str(round(obj['confidence']*100, 2))+'%',
                    (x1, y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, )
        # draw the bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('image2', frame)

    # visualize your prediction
    print('\nvisualization:\n')
    cv2.waitKey(10)
    # model.predict(frame, confidence=40, overlap=30).save('prediction'+str(num)+'.jpg')

    # infer on an image hosted elsewhere
    # print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())
