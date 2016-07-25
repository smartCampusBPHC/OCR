import glob, os, sys
import numpy as np
from PIL import Image as Img
from PIL import ImageEnhance as ImgEnh
from PIL import ImageFilter as ImgFil
#import pytesseract
import cv2
def add_mask(file_name ):
    img  = cv2.imread(file_name)

    img_final = cv2.imread(file_name)
    img2gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #ret, mask = cv2.threshold(img2gray, 10, 0, cv2.THRESH_BINARY)
    mask = cv2.adaptiveThreshold(img2gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,159,30)
    mask1 = mask
    edges = cv2.Canny(mask1,100,200)
    #image_final = cv2.bitwise_and(img2gray , img2gray , mask =  mask)
    #ret, new_img = cv2.threshold(image_final, 100 , 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
    '''
            line  8 to 12  : Remove noisy portion 
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(1 , 1)) # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more 
    #dilated = cv2.dilate(mask,kernel,iterations = 2) # dilate , more the iteration more the dilation

    img_ret, contours, hierarchy = cv2.findContours(img2gray,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE) # get contours
    index = 0 
    for contour in contours:
        # get rectangle bounding contour
        [x,y,w,h] = cv2.boundingRect(contour)

        #Don't plot small false positives that aren't text
        if w < 2:
            if h>15:
                continue

        # draw rectangle around contour on original image
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

        '''
        #you can crop image and send to OCR  , false detected will return no text :)
        cropped = img_final[y :y +  h , x : x + w]

        s = file_name + '/crop_' + str(index) + '.jpg' 
        cv2.imwrite(s , cropped)
        index = index + 1

        '''
    # write original image with added contours to disk  
    #cv2.imshow('hopefully text' , img)
    masked_path = path + '_masked' + '.jpg'
    cv2.imwrite(path + '_masked' + '.jpg', mask)
    #cv2.imshow('dilated',dilated)
    #cv2.imshow('cannyedges', edges)
    #cv2.imshow('img2gray', img2gray)
    #cv2.imshow('new_image', new_img)
    #cv2.imshow('image_final', image_final)
    #cv2.imshow('mask', mask)
    #cv2.imshow('hierarchy@findContours', hierarchy)
    #cv2.waitKey()
    return path + '_masked' + '.jpg'


def enhanceImage(path):
    maskedImage = Img.open(path)
    maskedImage = maskedImage.convert('RGBA')
    pix = maskedImage.load()
    for y in range(maskedImage.size[1]):
        for x in range(maskedImage.size[0]):
            if pix[x, y][0] < 102 or pix[x, y][1] < 102 or pix[x, y][2] < 102:
                pix[x, y] = (0, 0, 0, 255)
            else:
                pix[x, y] = (255, 255, 255, 255)

    maskedImage.save('temp.jpg')

    im = Img.open('temp.jpg')
    #im = im.filter(ImgFil.MedianFilter())
    enhancer = ImgEnh.Contrast(im)

#    im = enhancer.enhance(2)
#   im = im.convert('1')
    im.save('temp2.jpg')

    ExtractedText = pytesseract.image_to_string(Img.open('temp.jpg'))

