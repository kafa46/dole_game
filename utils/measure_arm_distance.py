import cv2
import numpy as np
import mediapipe as mp
import math 

from utils.angle_calculaters import calculate_angle

mpDraw= mp.solutions.drawing_utils  #미디어 파이프 초록색 선 그리기
mpPose = mp.solutions.pose
pose = mp.solutions.pose.Pose()

# 팔 각도 120도 이상 확인, 팔 거리를 계산
def measure_arm_distance(frame):
    
    frame=cv2.flip(frame, 1) # filp()= 좌우반전

    cv2.putText(
        frame,
        str("Fully strach your left arm: more than 160 degree"),
        (10,20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0,0,255),
        2,
    )
    
    # l1, l2 거리 계산을 위한 관절 좌표 추출
    results = pose.process(frame)   
    
    try:
        landmark = results.pose_landmarks.landmark
        
        # 좌측 어깨 좌표
        LEFT_SHOULDER = [
            landmark[mpPose.PoseLandmark.LEFT_SHOULDER].x,
            landmark[mpPose.PoseLandmark.LEFT_SHOULDER].y
        ]
        
        # 좌측 팔꿈치 좌표
        LEFT_ELBOW = [
            landmark[mpPose.PoseLandmark.LEFT_ELBOW].x,
            landmark[mpPose.PoseLandmark.LEFT_ELBOW].y
        ]

        # 좌측 손목 좌표
        LEFT_WRIST = [
            landmark[mpPose.PoseLandmark.LEFT_WRIST].x,
            landmark[mpPose.PoseLandmark.LEFT_WRIST].y
        ]

        print(f'좌측 어깨   좌표(x, y): {LEFT_SHOULDER[0]},\t{LEFT_SHOULDER[1]}')
        print(f'좌측 팔꿈치 좌표(x, y): {LEFT_ELBOW[0]},\t{LEFT_ELBOW[1]}')
        print(f'좌측 손목   좌표(x, y): {LEFT_WRIST[0]},\t{LEFT_WRIST[1]}')

        # angle = calculate_angle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST) 
        angle = calculate_angle(LEFT_WRIST, LEFT_ELBOW, LEFT_SHOULDER) 
        print(angle)
        
        mpDraw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mpPose.POSE_CONNECTIONS
        )
        success = angle >=160
        
        # Comput arm distance
        distance_from_shoulder_to_elbow = math.sqrt(
            (LEFT_SHOULDER[0] - LEFT_ELBOW[0])**2 + (LEFT_SHOULDER[1] - LEFT_ELBOW[1])**2  
        ) 
        distance_from_elbow_to_wrist = math.sqrt(
            (LEFT_ELBOW[0] - LEFT_WRIST[0])**2 + (LEFT_ELBOW[1] - LEFT_WRIST[1])**2
        )
        distance = distance_from_shoulder_to_elbow + distance_from_shoulder_to_elbow

        # Display angle
        cv2.putText(
            frame,
            'Angle:' + str(int(angle)),
            (10,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,0,255),
            2,
        )

        cv2.imshow('output',frame)
    
    
    except:
        success = False
        distance = None
        angle = None
    
    return frame, success, distance, angle
