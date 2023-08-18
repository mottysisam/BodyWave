import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_dangerously_set_inner_html
import mediapipe as mp
import BodyWavePosture as bw
from flask import Flask, Response
import cv2
import tensorflow as tf
import numpy as np
from utils import landmarks_list_to_array, label_params, label_final_results

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

model = tf.keras.models.load_model("bodywave_model")

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

def gen(camera):
    cap = camera.video
    i=0
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            image = cv2.flip(image, 1)

            if not success:
                print("Ignoring empty camera frame.")
                break

            image_height, image_width, _ = image.shape
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            dim=(image_width//5, image_height//5)
            resized_image = cv2.resize(image, dim)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            results = pose.process(resized_image)

            params = bw.get_params(results)
            flat_params = np.reshape(params, (5, 1))
            output = model.predict(flat_params.T)
            output_name = ['c', 'k', 'h', 'r', 'x', 'i']
            label = ""
            for i in range(1, 4):
                label += output_name[i] if output[0][i] > 0.5 else ""
            if label == "":
                label = "c"
            label += 'x' if output[0][4] > 0.15 and label=='c' else ''
            label_final_results(image, label)
            i+=1
            ret, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

server = Flask(__name__)
app = dash.Dash(__name__, server=server)
app.title = "BodyWave Posture"

@server.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = html.Div(className="main", children=[
    html.Link(
        rel="stylesheet",
        href="/assets/stylesheet.css"
    ),
    dash_dangerously_set_inner_html.DangerouslySetInnerHTML("""
        <div class="main-container">
            <table cellspacing="20px" class="table">
                <tr class="row">
                    <td> <img src="/assets/animation_for_web.gif" class="logo" /> </td>
                </tr>
                <tr class="choices">
                    <td> Your personal AI Gym Trainer </td>
                </tr>
                <tr class="row">
                    <td> <img src="/video_feed" class="feed"/> </td>
                </tr>
                <tr class="disclaimer">
                    <td> Please ensure that the scene is well lit and your entire body is visible </td>
                </tr>
            </table>
        </div>
    """),
])

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run_server(debug=True, host='0.0.0.0', port=port)
