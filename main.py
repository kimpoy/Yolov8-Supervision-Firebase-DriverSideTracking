import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

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

    cap = cv2.VideoCapture(0)
    #cap = cv2.VideoCapture("rtsp://yolov8:rmk2024@192.168.100.64")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)

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
        ret, frame = cap.read()
        #result = model(frame) #with no supervision
        #result = model(frame)[0] #with supervision
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        detections = detections[detections.class_id == 0]
        #detections = sv.Detections.from_ultralytics(result)

        for detection in detections:
            print(detection[0][0])

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
        
        cv2.imshow("yolov8",frame)
        #cv2.imshow("image",frame)
        #print(frame.shape)

        if cv2.waitKey(30) == 27:
            break

main()
#if __name__ == "__main__":
#    main()