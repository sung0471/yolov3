from __future__ import division
import argparse
import time
import cv2
import sys
import torch
from models.darknet import Darknet
from utils.detection_boxes_pytorch import DetectBoxes


def arg_parse():
    parser = argparse.ArgumentParser(description='Pytorch Yolov3')
    parser.add_argument('--video', help='video의 위치', default='assets/cars.mp4', type=str)
    parser.add_argument('--config', help='Yolov3 config file', default='darknet/yolov3.cfg')
    parser.add_argument('--weight', help='Yolov3 weight file', default='darknet/yolov3.weights')
    parser.add_argument('--conf', dest='confidence', help='Confidence threshold for predictions', default=0.5)
    parser.add_argument('--nms', dest='nmsThreshold', help='NMS threshold', default=0.4)
    parser.add_argument('--resolution', dest='resol', help='network에 들어가는 입력 해상도. 정확도 증가 and 속도 감소',
                        default=416, type=int)
    parser.add_argument('--webcam', help='web camera 탐지', default=False)

    return parser.parse_args()


def main():
    args = arg_parse()

    VIDEO_PATH = args.video if not args.webcam else 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Network...')
    model = Darknet(args.config, img_size=args.resol).to(device)
    model.load_darknet_weights(args.weight)
    print('Network successfully loaded')

    model.eval()

    PATH_TO_LABELS = 'labels/coco.names'

    # detection class 로드, 기본 정확도 threshold = 0.5
    detect = DetectBoxes(PATH_TO_LABELS, conf_threshold=args.confidence, nms_threshold=args.nmsThreshold)

    # Set window
    winName = 'YOLO-Pytorch'

    try:
        # read video file
        cap = cv2.VideoCapture(VIDEO_PATH)
    except IOError:
        print('input video file', VIDEO_PATH, "doesn't exist")
        sys.exit(1)

    while cap.isOpened():
        hasFrame, frame = cap.read()

        if not hasFrame:
            break

        start = time.time()
        detect.bounding_box_yolo(frame, args.resol, model)
        end = time.time()

        cv2.putText(frame, '{:.2f}ms'.format((end - start) * 1000), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (255, 0, 0), 2)

        cv2.imshow(winName, frame)
        print('FPS {:5.2f}'.format(1 / (end - start)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Video ended")

    # release video and 모든 window 창 제거
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
