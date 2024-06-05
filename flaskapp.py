from flask import Flask, render_template, Response,jsonify,request,session,url_for
import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np
from flask import Flask, render_template,url_for,request,redirect

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Initialize Firebase with the credentials JSON file
cred = credentials.Certificate("iot-project-6b313-firebase-adminsdk-l3dnb-b8239e308b.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://iot-project-6b313-default-rtdb.firebaseio.com/'
})

# Get a reference to the Firebase Realtime Database
ref = db.reference('/Counter')


# Update
def update_data(data_id, updated_data):
    ref.child(data_id).update(updated_data)


app = Flask(__name__)
app.config['SECRET_KEY'] = ['KIM']



ZONE_POLYGON = np.array([
    [0,0],
    [1280 // 2,0],
    [1250 // 2,720],
    [0,720],
])


def parse_arguments()->argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280,720],
        nargs=2,
        type=int
    )
    args = parser.parse_args()
    return args


def main():
    #frame resolution
    args = parse_arguments()
    frame_width,frame_height = args.webcam_resolution

    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)
    x,y,w,h = 10,10,200,50

    model = YOLO("yolov8n.pt")

    box_annotator = sv.BoxAnnotator(
        thickness=2,
        text_thickness=2,
        text_scale=1
    )

    zone = sv.PolygonZone(polygon=ZONE_POLYGON,frame_resolution_wh=tuple(args.webcam_resolution))
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color = sv.Color.blue(),
        thickness=2,
        text_thickness=4,
        text_scale=2
        )

    while True:
        global total_passenger
        ret, frame = cap.read()
        #result = model(frame) #with no supervision
        #result = model(frame)[0] #with supervision
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.class_id == 0]
        #detections = sv.Detections.from_ultralytics(result)

        counter_frame = frame

        labels = [
            f"{model.model.names[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, _ 
            in detections
        ]

        frame = box_annotator.annotate(
            scene=frame,
            detections=detections,
            labels=labels
            )
        
        #sample = zone.trigger(detections=detections)
        sample = zone.trigger(detections=detections)
        #counting the number
        #print(len(sample))
        frame = zone_annotator.annotate(scene=frame)
        total = len(list(filter(None,sample)))

        cv2.rectangle(counter_frame, (x, x), (x + w, y + h), (0,0,0), -1)
        cv2.putText(counter_frame, f'Total:{total}', (x + int(w/10),y + int(h) - int(h/3)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 4)

        # Update data
        data_id = "-NzdK5eOSWNahdBKg3EP"  # Replace <data_id> with the ID of the data you want to update
        updated_data = {"Counter": total}
        update_data(data_id, updated_data)

        total_passenger = total
        
        yield frame
        #cv2.imshow("yolov8",frame)
        #cv2.imshow("image",frame)
        #print(frame.shape)

        #if cv2.waitKey(30) == 27:
        #    break

def generate_frames():
    yolo_output = main()
    for detection_ in yolo_output:
        ref,buffer=cv2.imencode('.jpg',detection_)
        frame = buffer.tobytes()
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame +b'\r\n')
        

@app.route("/", methods=['GET','POST'])
def index():
    session.clear()
    return render_template('video.html')


# To display the Output Video on Webcam page
@app.route('/webapp')
def webapp():
    #return Response(generate_frames(path_x = session.get('video_path', None),conf_=round(float(session.get('conf_', None))/100,2)),mimetype='multipart/x-mixed-replace; boundary=frame')
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host='0.0.0.0', port=5000, threaded=True)