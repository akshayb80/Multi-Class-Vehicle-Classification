from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
                help="path to trained model model")
# ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# image = cv2.imread(args["image"])
cap = cv2.VideoCapture("videoplayback.mp4")
fourcc = cv2.VideoWriter_fourcc(*'XVID') # writes video frame by frame
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480)) # 20.0 fps and resolution is (640, 480)
# orig = image.copy()
labels = ["Bike", "Car"]



while True:
    ret, frame = cap.read()
    orig = frame.copy()
    frame = cv2.resize(frame, (28, 28))
    frame = frame.astype("float") / 255.0
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)

    print("[INFO] loading network...")
    model = load_model(args["model"])

    (Bike, Car) = model.predict(frame)[0]

    # print(Bike)
    # print(Car)
    # print(Truck)

    # build the label
    # label = np.argmax([Bike, Car])
    label = np.argmax([Bike, Car])
    label = labels[label]
    proba = max([Bike, Car])
    label = "{}: {:.2f}%".format(label, proba * 100)
    # draw the label on the image
    output = imutils.resize(orig, width=400)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)
    # show the output image
    cv2.imshow("Output", output)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
