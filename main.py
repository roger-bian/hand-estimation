import argparse
import glob

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2


def create_landmarker():
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=4,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    detector = vision.HandLandmarker.create_from_options(options)
    return detector
    
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


def infer_webcam():
    cap = cv2.VideoCapture(0)

    detector = create_landmarker()

    while True:
        ret, frame = cap.read()
        
        if ret:
            # convert to rgb
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # detect
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            detection_result = detector.detect(mp_image)
            
            # visualize
            frame = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            cv2.imshow('frame', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break        
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()


def infer_image(img: str):
    detector = create_landmarker()
    
    image = mp.Image.create_from_file(img)

    detection_result = detector.detect(image)

    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    
    cv2.waitKey(0)
    cv2.destroyAllWindows() 


def infer_dir(dir: str):
    images = glob.glob(dir + '/*')
    images = sorted(images)
    
    detector = create_landmarker()
    
    for image in images:
        image = mp.Image.create_from_file(image)

        detection_result = detector.detect(image)

        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
        
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()    


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--webcam', action='store_true')
    parser.add_argument('--image', type=str, default=None)
    parser.add_argument('--image-dir', type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = argument_parser()
    
    if args.webcam:
        infer_webcam()
    elif args.image:
        infer_image(args.image)
    elif args.image_dir:
        infer_dir(args.image_dir)
    else:
        print('Please set an appropriate flag')
        