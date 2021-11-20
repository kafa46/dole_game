import cv2
import mediapipe as mp
import numpy as np

from mole_actions.moleOut import moleOut
from utils.angle_calculaters import calculate_angle, moleUp_decision_and_update_numCount
from utils.angle_gage import angleGage
from utils.estimate_arm_points import estimate_arm_coordinates
from utils.measure_arm_distance import measure_arm_distance
from utils.print_infomation import put_numCount

MAX_ANGLE = 160
MIN_ANGLE = 60
SUCCESS = False

mpDraw= mp.solutions.drawing_utils  #미디어 파이프 초록색 선 그리기
mpPose = mp.solutions.pose
pose = mpPose.Pose()
pose = mp.solutions.pose.Pose()

# 1번째 두더지
numCount_left = 0
SHRINED_LEFT = True
moleSwitch = True

# 2번째 두더지
numCount_right = 0
SHRINED_RIGHT = True
moleSwitch2 = True

#두더지 사이즈
moleY, moleX = 150, 150

def main(config):
    
    # img load & resize
    mole_img = cv2.imread(config['img_path'])
    mole_img = cv2.resize(mole_img, (moleX, moleY), interpolation=cv2.INTER_CUBIC)
    #cv2.imshow('mole_img', mole_img) # Check img loading, test use only
    
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    
    cap=cv2.VideoCapture(0) # 비디오 불러오기
    if cap.isOpened():
        print(f"\n웹캠 작동 상태: {cap.isOpened()}")
        print('width: {}, height : {}'.format(cap.get(3), cap.get(4)))

    #while cap.isOpened(): # 한 개의 프레임 마다 읽어오기
    #while True: # 한 개의 프레임 마다 읽어오기

    while cv2.waitKey(33) < 0:

        ret, frame = cap.read()
        
        if not SUCCESS:
            cv2.imshow('output', frame)
            
            print('\nMeasuring arm distance...')
            frame, success, distance, angle = measure_arm_distance(frame)
            print(f'success: {success} \t Arm distance: {distance} \t Arm angle: {angle}')

            if not success or not distance:
                continue

            elif success and distance:
                SUCCESS = True
            
            else:
                pass

        cv2.imshow('output',frame)    
        # l1, l2 거리 계산
        # input('pause...잠시만...')
            
    
        frame = cv2.resize(frame, (800,600)) 
        frameX = round(200) - round(moleY * 0.5)
        frameY = round(600 / 2) - round(moleX * 0.5)

        #frame=cv2.flip(frame, 1) # filp()= 좌우반전
        rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #oepn cv : bgr /mediapipe : rgb    
        results=pose.process(rgb)
        cv2.imshow('output', frame)
        
        try:
            # 왼팔, 오른팔 좌표 구하기
            landmark = results.pose_landmarks.landmark
            left_arm, right_arm = estimate_arm_coordinates(landmark)

            # 왼팔 각도 계산
            left_wrist, left_elbow, left_shoulder = left_arm
            angle_left_arm = round(
                calculate_angle(left_wrist, left_elbow, left_shoulder)
            )
            
            # 왼팔 팔각도를 기준으로 두더지를 나타낼지 판단하고, 운동 카운트 업데이트
            moleSwitch, SHRINED_LEFT, numCount_left = moleUp_decision_and_update_numCount(
                angle=angle_left_arm,
                max_angle=MAX_ANGLE,
                min_angle=MIN_ANGLE,
                moleSwitch=moleSwitch,
                shrinked=SHRINED_LEFT,
                numCount=numCount_left
            )

            # 오른팔 각도 계산
            right_wrist, right_elbow, right_shoulder = right_arm
            angle_right_arm = round(
                calculate_angle(right_wrist, right_elbow, right_shoulder)
            )
            
            # 오른팔 팔각도를 기준으로 두더지를 나타낼지 판단하고, 운동 카운트 업데이트
            moleSwitch2, SHRINED_RIGHT, numCount_right = moleUp_decision_and_update_numCount(
                angle=angle_right_arm,
                max_angle=MAX_ANGLE,
                min_angle=MIN_ANGLE,
                moleSwitch=moleSwitch2,
                shrinked=SHRINED_RIGHT,
                numCount=numCount_right,
            )

            #스켈레톤 그리기
            mpDraw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mpPose.POSE_CONNECTIONS
            )

            #프레임에 'Num_count: xx' 출력
            put_numCount(frame, numCount_left, counterID=1)
            put_numCount(frame, numCount_right, counterID=2)

            #두더지 화면 출력
            moleOut(moleSwitch, mole_img, frameX, frameY, frame)
            frameX = round(600) - round(moleY*0.5)
            moleOut(moleSwitch2, mole_img, frameX, frameY, frame)

            #각도별 그래프 그리기
            angleGage(angle_left_arm, 80, frame)                 
            angleGage(angle_right_arm, 720, frame)

        except Exception:
            pass

        key=cv2.waitKey(1)
        
        if key == ord('q') or key == ord('Q'):
            break

    cap.release() # 비디오 종료
    cv2.destroyAllWindows()

if __name__=='__main__':
    
    config = {
        'img_path': './imgs/mole-1024x853.png',
    }
    
    main(config)