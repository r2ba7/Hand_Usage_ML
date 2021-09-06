#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import pandas as pd
import time
import autopy
import mediapipe as mp


class HandDetectionClass():
    def __init__(self, mode=False, 
                 max_hands=2, 
                 min_det_con=0.5,
                 min_trck_con=0.5
                ):
        
            self.mode = mode
            self.max_hands = max_hands
            self.min_det_con = min_det_con
            self.min_trck_con = min_trck_con
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                                            static_image_mode=self.mode,
                                            max_num_hands=self.max_hands,
                                            min_detection_confidence=self.min_det_con,
                                            min_tracking_confidence=self.min_trck_con,
                                            )
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
        
        
        
    def findHands(self, img, draw=True):
        image = img.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        self.results = self.hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                for ID, point in enumerate(self.mp_hands.HandLandmark):
                    normalizedLandmark = hand_landmarks.landmark[point]
                    #pixelCoordinatesLandmark = mp_drawing._normalized_to_pixel_coordinates(normalizedLandmark.x, 
                    #                                                                        normalizedLandmark.y, 
                    #                                                                        width, 
                    #                                                                        height)
                    height, width, channels = image.shape
                    cx = int(normalizedLandmark.x * width)
                    cy = int(normalizedLandmark.y * height)
                    if draw:
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style()
                            ) 

        return image

    def findPosition(self, img, handNo=0, draw_indecis=True):
        self.landmark_list = []
        if self.results.multi_hand_landmarks:
            particular_hand = self.results.multi_hand_landmarks[handNo]
            for ID, landmark in enumerate(particular_hand.landmark):
                height, width, channels = img.shape
                relative_x, relative_y = int(landmark.x * width), int(landmark.y * height)
                self.landmark_list.append([ID, relative_x, relative_y])
                if draw_indecis:
                    cv2.circle(img, (relative_x, relative_y), 5,(0,0,0), 3)
                

        return self.landmark_list
    
            

def main():
    previous_time = 0
    current_time = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetectionClass()
    while True:
        retval, image = cap.read()
        image = detector.findHands(image)
        landmarks_list = detector.findPosition(image)
        if len(landmarks_list):
            print(landmarks_list[4])
        current_time = time.time()
        fps = 1/(current_time-previous_time)
        previous_time = current_time
        cv2.putText(image, str(int(fps)), (5,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)        
        cv2.imshow('Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
        

    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()
