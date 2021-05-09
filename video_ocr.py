import cv2
import math
from tqdm import tqdm 
import easyocr
import sys
import json
import os


#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def main(imagesFolder,videoFile,outputFile):
    reader = easyocr.Reader(['en']) 
    cap = cv2.VideoCapture(videoFile)
    frameRate = cap.get(5) #frame rate
    totalFrames = cap.get(7) #total frames
    map_sec_to_text = {}
    count = 0
    confidence = 0.5
    pbar = tqdm(total=totalFrames//frameRate)
    while(cap.isOpened()):
        frameId = cap.get(1) #current frame number
        ret, frame = cap.read()
        if (ret != True):
            break
        if (frameId % math.ceil(frameRate) == 0):
            filename = imagesFolder + "/image_" +  str(int(frameId)) + ".jpg"
            filename_gray = imagesFolder + "/image_gray_" +  str(int(frameId)) + ".jpg"
            cv2.imwrite(filename, frame)
            frame = get_grayscale(frame)
            frame = thresholding(frame)
            cv2.imwrite(filename_gray, frame)  
            result = reader.readtext(filename_gray, detail = 0)
            map_sec_to_text[count] = result
            count+=1
            pbar.update(1)

    cap.release()
    print("Done!")


    with open(outputFile, 'w') as file:
        file.write(json.dumps(map_sec_to_text))


if __name__ == "__main__":
    images_path = "images"
    video_path = sys.argv[0]
    output_file_path = os.path.basename(video_path).split(".")[0] + ".txt" 
    print(f"Creating {output_file_path} for {video_path}")
    main(images_path,video_path,output_file_path)









