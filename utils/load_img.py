import cv2

def img_load_and_resize(img_path, moleX, moleY):
    
    # Load img from path
    mole_img = cv2.imread(img_path)

    # Resize the loaded mole img
    mole_img = cv2.resize(mole_img, dsize=(moleX, moleY), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('mole_img', mole_img)

    return mole_img