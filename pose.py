'''
file: pose.py
brief: estimate the pose of the Imam in prayer
author: Mohammed Nurul Hoque
Created: 2019-12-29
'''
import tflite_runtime.interpreter as tf
import cv2
import numpy as np
import math

PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

IDX = {part:i for i, part in enumerate(PART_NAMES)}

CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"), 
    ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), 
    ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), 
    ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), 
    ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), 
    ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), 
    ("leftHip", "rightHip")
]

# sigmoid function
def sigmoid(x):
  return 1 / (1 + math.exp(-x))
sigmoid_v = np.vectorize(sigmoid)

def visualize_heatmap(hm):
    h, w = hm.shape[1:3]
    hms = list(map(lambda a: a.reshape(h,w), np.split(hm, 17, 3)))
    view = np.zeros([h*4, w*5])
    for i, hm in enumerate(hms):
        y, x = np.unravel_index(i, (4, 5))
        view[y*h:(y+1)*h, x*w: (x+1)*w] = hm

    # upscale image bc heatmaps are tiny (9x9)
    view = cv2.resize(view, (w*50,h*40))

    # draw grid lines
    for i in range(1, 4):
        view[h*i*10,:] = 0.5
    for i in range(1, 5):
        view[:,w*i*10] = 0.5
    
    # put part names
    for i in range(17):
        y, x = np.unravel_index(i, (4, 5))
        cv2.putText(view, PART_NAMES[i], (w*x*10+5, h*y*10+10), 
            cv2.FONT_HERSHEY_PLAIN, 0.75, 128)
            
    cv2.imshow('Visualize', view, )

# video stream
cap = cv2.VideoCapture('/dev/video0')
def get_frame():
    ret, frame = cap.read()
    return frame

# change raw OpenCV frame to match model input
def frame_reshape(frame, shape):
    cv2.normalize
    frame = cv2.resize(frame, (shape[1], shape[2]))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype('float32')/255
    frame = frame.reshape(shape)
    return frame 

# init tf model
def init_model():
    interpreter = tf.Interpreter(model_path='posenet_mobilenet_v1_100_257x257.tflite')
    interpreter.allocate_tensors()
    return interpreter

interpreter = init_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']
# shape of model input frame
iy = input_shape[1]
ix = input_shape[2]

while True:
    frame = get_frame()
    # shape of original frame
    fy, fx = frame.shape[0], frame.shape[1]
    input_data = frame_reshape(frame, input_shape)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    heatmap = sigmoid_v(interpreter.get_tensor(output_details[0]['index']))
    offsets = interpreter.get_tensor(output_details[1]['index'])

    hmheight = heatmap.shape[1]
    hmwidth = heatmap.shape[2]
    n_points = heatmap.shape[3]

    # find max confidence in heatmap per part
    keypoints = [(0, 0) for i in range(n_points)]
    max_vals = [0 for i in range(n_points)]
    for i in range(hmheight):
        for j in range(hmwidth):
            for p in range(n_points):
                if heatmap[0][i][j][p] > max_vals[p]:
                    max_vals[p] = heatmap[0][i][j][p]
                    keypoints[p] = (i, j)

    # calculate coords in input
    coords = [(0, 0) for i in range(n_points)]
    for i, point in enumerate(keypoints):
        y, x = point
        coords[i] = (y*32 + offsets[0][y][x][i], x*32 + offsets[0][y][x][i+n_points])

    # convert to coords in original and flip to (x, y)
    real_coords = [(int(x*fx/ix), int(y*fy/iy)) for y, x in coords]

    # draw parts
    cv2.circle(frame, real_coords[0], 5, (0, 255, 0))
    for i, point in enumerate(real_coords[1:]):
        if max_vals[i] > 0.1:
            cv2.circle(frame, point , 5, (0, 0, 255))
        cv2.putText(frame, str(i+1), point, cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 0, 255))

    # connect edges
    for ex, ey in CONNECTED_PART_NAMES:
        i, j = IDX[ex], IDX[ey]
        if max_vals[i] > 0.1 and max_vals[j] > 0.1:
            cv2.line(frame, real_coords[PART_NAMES.index(ex)],
                real_coords[PART_NAMES.index(ey)], (255, 255, 0), 2)

    # draw grid
    for i in range(n_points):
        for j in range(n_points):
            cv2.circle(frame, (int((i+0.5)*(frame.shape[1]/hmwidth)),
                int((j+0.5)*(frame.shape[0]/hmheight))), 2, (255, i*15, j*15), 1)

    cv2.imshow('Pose', frame)
    visualize_heatmap(heatmap)
    if cv2.waitKey(30) == ord('q'):
        break

cv2.destroyAllWindows()