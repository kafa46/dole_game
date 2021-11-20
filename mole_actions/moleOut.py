import cv2

#두더지 나타나기
def moleOut(moleSwitch, mole_img, frameX, frameY, frame):
    '''
    두더지 나타나기
    params:
        moleSwitch: boolian, if True
        moleShape:
        frameX:
        frameY:
        frame:
        img: image of mole which is resized
    
    returns:
        frame
    '''
    
    if moleSwitch:
        
        # 로고파일 픽셀값 저장
        rows, cols, _ = mole_img.shape

        #로고파일 필셀값을 관심영역(ROI)으로 저장
        roi = frame[
            frameY: rows + frameY, 
            frameX: cols + frameX
        ] 
        cv2.imshow("output", frame)
        
        #로고파일의 색상을 그레이로 변경
        gray = cv2.cvtColor(mole_img, cv2.COLOR_BGR2GRAY) 
        
        #배경은 흰색으로, 그림을 검정색으로 변경
        _, mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY) 
        mask_inv = cv2.bitwise_not(mask)

        #배경에서만 연산 = src1 배경 복사
        src1_bg = cv2.bitwise_and(roi, roi, mask=mask) 
        
        #로고에서만 연산
        src2_fg = cv2.bitwise_and(mole_img, mole_img, mask=mask_inv) 
        
        #src1_bg와 src2_fg를 합성
        dst = cv2.bitwise_or(src1_bg, src2_fg) 
        
        #src1에 dst값 합성
        frame[frameY:rows+frameY,frameX:cols+frameX] = dst 

    return frame