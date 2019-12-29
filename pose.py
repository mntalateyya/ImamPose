import tflite_runtime.interpreter as tf
import cv2
import copy

PART_NAMES = [
    "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
    "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
    "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
]

CONNECTED_PART_NAMES = [
    ("leftHip", "leftShoulder"), ("leftElbow", "leftShoulder"),
    ("leftElbow", "leftWrist"), ("leftHip", "leftKnee"),
    ("leftKnee", "leftAnkle"), ("rightHip", "rightShoulder"),
    ("rightElbow", "rightShoulder"), ("rightElbow", "rightWrist"),
    ("rightHip", "rightKnee"), ("rightKnee", "rightAnkle"),
    ("leftShoulder", "rightShoulder"), ("leftHip", "rightHip")
]

# switch section with commented lines to use host webcam instead
im = cv2.imread('pose3.png')
#cap = cv2.VideoCapture('/dev/video0')
def get_frame():
    return im
    #ret, frame = cap.read()
    #return frame

# change raw OpenCV frame to match model input
def frame_reshape(frame, shape):
    frame = cv2.resize(frame, (shape[1], shape[2]))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype('float32')
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
iy = input_shape[1]
ix = input_shape[2]

while True:
    frame = copy.deepcopy(get_frame())
    fy, fx = frame.shape[0], frame.shape[1]
    input_data = frame_reshape(frame, input_shape)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    heatmap = interpreter.get_tensor(output_details[0]['index'])
    offsets = interpreter.get_tensor(output_details[1]['index'])

    hmheight = heatmap.shape[1]
    hmwidth = heatmap.shape[2]
    n_points = heatmap.shape[3]

    # find max confidence in heatmap per part
    keypoints = [(0, 0) for i in range(n_points)]
    for p in range(n_points):
        maxval = -1000.0
        for i in range(hmheight):
            for j in range(hmwidth):
                if heatmap[0][i][j][p] > maxval:
                    maxval = heatmap[0][i][j][p]
                    keypoints[p] = (i, j)

    # calculate coords in input
    coords = [(0, 0) for i in range(n_points)]
    for i, point in enumerate(keypoints):
        y, x = point
        coords[i] = (y*32 + offsets[0][y][x][i], x*32 + offsets[0][y][x][i+n_points])

    # convert to coords in original and flip to (x, y)
    real_coords = list(map(lambda c: (int(c[1]*fx/ix), int(c[0]*fy/iy)), coords))

    # draw parts
    cv2.circle(frame, real_coords[0], 5, (0, 255, 0), 2)
    for i, point in enumerate(real_coords[1:]):
        cv2.circle(frame, point , 5, (0, 0, 255), 2)
        cv2.putText(frame, str(i+1), point, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1, cv2.LINE_AA)

    # connect edges
    for ex, ey in CONNECTED_PART_NAMES:
        cv2.line(frame, real_coords[PART_NAMES.index(ex)],
            real_coords[PART_NAMES.index(ey)], (0, 0, 0), 1)

    # draw grid
    for i in range(n_points):
        for j in range(n_points):
            cv2.circle(frame, (int((i+0.5)*(frame.shape[1]/hmwidth)),
                int((j+0.5)*(frame.shape[0]/hmheight))), 3, (255, i*15, j*15), 2)

    cv2.imshow('Pose', frame)

    # switch section with commented lines to use host webcam instead
    cv2.waitKey(0)
    break
    #if cv2.waitKey(30) == ord('q'):
    #    break

cv2.destroyAllWindows()