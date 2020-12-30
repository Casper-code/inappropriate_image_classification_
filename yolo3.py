#!/usr/bin/python

import numpy as np
import argparse
import time
import cv2 as cv
import os
from tensorflow import keras

def runYOLODetection(args):
    # load CASPER class labels that my YOLO model was trained on
    labelsPath = os.path.sep.join([args["yolo"], "casper.names"])
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    np.random.seed(0)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
        dtype="uint8")
    print(COLORS)
    #COLORS = np.array([255, 0, 0], dtype="uint8")

    # derive the paths to the YOLO weights and model configuration
    weightsPath = os.path.sep.join([args["yolo"], "casper_11000.weights"])
    configPath = os.path.sep.join([args["yolo"], "casper.cfg"])

    # load my YOLO object detector trained on my CASPER dataset (1 class)
    print("[INFO] loading YOLO from disk ...")
    net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
    
    # load Image Classification
    model_path= os.path.join(args["yolo"],"Image_classification","NSFW","")
    print(model_path)
    model = keras.models.load_model(model_path)
    # load input image and grab its spatial dimensions
    #print(args["image"])

    for filename in os.listdir(args["image"]):
        absolute_path_read=os.path.join(os.getcwd(),args["image"],filename)
        image = cv.imread(absolute_path_read)
        (H, W) = image.shape[:2]
        print(H,W)
    
    
    # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    # NOTE: (608, 608) is my YOLO input image size. However, using 
    # (416, 416) results in much accutate result. Pretty interesting.
        blob = cv.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
            swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layerOutputs = net.forward(ln)
        end = time.time()

    # show execution time information of YOLO
        print("[INFO] YOLO took {:.6f} seconds.".format(end - start))

    # initialize out lists of detected bounding boxes, confidences, and
    # class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []

    # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
            
                # filter out weak predictions by ensuring the detected
                # probability is greater then the minimum probability
                if confidence > args["confidence"]:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update out list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)

        # apply non-maxima suppression to suppress weark and overlapping bounding
        # boxes
        idxs = cv.dnn.NMSBoxes(boxes, confidences, args["confidence"],
            args["threshold"])
        im=0
        te=0
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[i]]]
                
                
                if LABELS[classIDs[i]].startswith('image'):
                    #crop images identified by yolo
                    croped=imcrop(image,x,y,x+w,y+h)
                    #print(croped.shape, type(croped))
                    #image_labels=predict_from_image(model,croped)
                    #Predict image label
                    image_labels=predict_from_image_NSFW(model,croped)
                    #plot box with label on original picture
                    cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv.putText(image, image_labels, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
                    0.5, color, 2)
                    im+=1
                    if not os.path.exists('CASPER/Images'):
                        os.makedirs('CASPER/Images')
                    
                    absolute_path_img= os.path.join(os.getcwd(),'CASPER/Images/',str(LABELS[classIDs[i]]) + '_' + str(im) + '_'+filename)
                    print(absolute_path_img)
                    #print(absolute_path_img)
                    #save cropped image - not needed for execution but for testing
                    cv.imwrite(absolute_path_img, croped)
                if LABELS[classIDs[i]].startswith('text'):
                    #crop images identified by yolo
                    croped=imcrop(image,x,y,x+w,y+h)
                    #insert text recognition (Teseract) and text classifier 
                    #
                    #
                    #
                    #plot box with label on original picture
                    #cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    #cv.putText(image, text_labels, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
                    #0.5, color, 2)
                    te+=1
                    if not os.path.exists('CASPER/Text'):
                        os.makedirs('CASPER/Text')
                    absolute_path_txt= os.path.join(os.getcwd(),'CASPER/Text/',str(LABELS[classIDs[i]]) + '_' + str(te) + '_'+filename)
                    print(absolute_path_txt)
                    #save cropped text - not needed for execution but for testing purposes - testing algorithms outside of main program
                    cv.imwrite(absolute_path_txt, croped)
            if not os.path.exists('CASPER/yolo_labeled'):
                os.makedirs('CASPER/yolo_labeled')
            absolute_path_labeled=os.path.join(os.getcwd(),'CASPER/yolo_labeled/',filename)
            #save CASPER results
            cv.imwrite(absolute_path_labeled, image)

    return image

def imcrop(img, xmin,ymin,xmax,ymax):
    if xmin < 0 or ymin < 0 or xmax > img.shape[1] or ymax > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, xmin, xmax, ymin, ymax)
    else:
        x1=xmin
        x2=xmax
        y1=ymin
        y2=ymax
    return img[y1:y2, x1:x2, :]

def pad_img_to_fit_bbox1(img, x1, x2, y1, y2):
    img = cv.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
        -min(0, x1), max(x2 - img.shape[1], 0),cv.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2
    
def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = np.pad(img, ((np.abs(np.minimum(0, y1)), np.maximum(y2 - img.shape[0], 0)),
               (np.abs(np.minimum(0, x1)), np.maximum(x2 - img.shape[1], 0)), (0,0)), mode="constant")
    y1 += np.abs(np.minimum(0, y1))
    y2 += np.abs(np.minimum(0, y1))
    x1 += np.abs(np.minimum(0, x1))
    x2 += np.abs(np.minimum(0, x1))
    return img, x1, x2, y1, y2

def predict_from_image(model,img):
    size = (200,200)
    resized = cv.resize(img, size, interpolation = cv.INTER_AREA)
    input_image = np.expand_dims(np.array(resized)/255, axis=0)
    predict_proba = model.predict(input_image)[0][0]
    text = ''
    if predict_proba > .5:
        text = 'NSFW:'+str(predict_proba*100)
    
    else:
        text = 'SFW:'+str(100-predict_proba*100)
    return text

def predict_from_image_NSFW(model,img):
    size = (224,224)
    resized = cv.resize(img, size, interpolation = cv.INTER_AREA)
    input_image = np.expand_dims(np.array(resized)/255, axis=0)
    Y_pred = model.predict(input_image)
    y_pred = np.argmax(Y_pred, axis=-1)
    text = ''
    if y_pred==0:
        text= 'drawings'
    elif y_pred==1:
        text='hentai'
    elif y_pred==2:
        text='neutral'
    elif y_pred==3:
        text='porn'
    else:
        text='sexy'
    return text

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
        help="path to directory with images")
    ap.add_argument("-y", "--yolo", required=True,
        help="base path to YOLO directory")
    ap.add_argument("-c", "--confidence", type=float, default=0.25,
        help="minimum probability to filter weak detections")
    ap.add_argument("-t", "--threshold", type=float, default=0.45,
        help="threshold when applying non-maxima suppression")
    args = vars(ap.parse_args())

    image = runYOLODetection(args)

    # show the output image
    #cv.namedWindow("Image", cv.WINDOW_NORMAL)
    #cv.resizeWindow("image", 1920, 1080)
    #cv.imshow("Image", image)
    
    cv.waitKey(0)
