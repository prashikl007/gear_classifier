import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import cv2
import time
import serial.tools.list_ports


port_data = []
for port in serial.tools.list_ports.comports():
    port_data.append(port.name)
    port_data.append(port.description)

if 'Silicon Labs CP210x USB to UART Bridge (COM4)' in port_data:
    index = port_data.index('Silicon Labs CP210x USB to UART Bridge (COM4)')
    selected_port = port_data[index - 1]

arduino = serial.Serial(port=selected_port,  baudrate=115200, timeout=.1)

model_name ="D:/Projects/pythonProject/detector_for_industry/cnn1.h5"
#detection using model
model = tf.keras.models.load_model(model_name)


def detect_obj(image):
    #test_image_conv = image.img_to_array(test_image)
    test_image = cv2.resize(image, (64, 64))
    test_image_conv = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image_conv)
    res = result.tolist()
    # training_set.class_indices
    prediction = 'no prediction'
    x = '0'
    if res[0][0] == 1:
        prediction = 'blank'
        x = '0'
    if res[0][1] == 1:
        prediction = 'herringbone'
        x = '1'
    if res[0][2] == 1:
        prediction = 'spur'
        x = '2'
    print(prediction)
    return x


cap = cv2.VideoCapture(0)
while True:
    ret, image = cap.read()
    cv2.imshow("image", image)
    key = cv2.waitKey(20)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break
    x = detect_obj(image)
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)

cv2.destroyAllWindows()
cap.release()
