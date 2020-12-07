#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import imutils
import pprint
from matplotlib import pyplot as plt
import streamlit as st
from PIL import Image
import os


st.write("""
# Find Your Dominant Skincolour 

Have you struggled to find the right foundation and concealer because you don't know your exact skintone? Or perhaps, you weren't able to find the right colour tights to go with your unique skin tone?

""")

    
def detectSkin(image):
    
    img = image.copy()
    
    # Converting from BGR Colours Space to HSV
    img = np.array(img)
    
    img = cv2.cvtColour(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower = np.array([0, 48, 80], dtype=np.uint8)
    upper = np.array([12, 255, 255], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower, upper)

    # Cleaning up mask using Gaussian Filter
    
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    
        # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColour(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each colour
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring colour
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        colour = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the colour is [0,0,0] that if it is black
        if compare(colour, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)


def getColourData(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each colour predicted
    occurance_counter = None

    # Output list variable to return
    colourInformation = []

    # Check for Black
    hasBlack = False

    # If a mask has be applied, remove th black
    if hasThresholding == True:

        (occurance, cluster, black) = removeBlack(
            estimator_labels, estimator_cluster)
        occurance_counter = occurance
        estimator_cluster = cluster
        hasBlack = black

    else:
        occurance_counter = Counter(estimator_labels)

    # Get the total sum of all the predicted occurances
    totalOccurance = sum(occurance_counter.values())

    # Loop through all the predicted colours
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the colour number into a list
        colour = estimator_cluster[index].tolist()

        # Get the percentage of each colour
        colour_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colourInfo = {"cluster_index": index, "colour": colour,
                     "colour_percentage": colour_percentage}

        # Add the dictionary to the list
        colourInformation.append(colourInfo)

    return colourInformation


def extractDominantTone(image, number_of_colours=3, hasThresholding=False):

    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colours += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColour(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colours, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colourInformation = getColourData(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colourInformation


def graphColourBar(colourInformation):
    # Create a 500x100 black image
    colour_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colourInformation:
        bottom_x = top_x + (x["colour_percentage"] * colour_bar.shape[1])

        colour = tuple(map(int, (x['colour'])))

        cv2.rectangle(colour_bar, (int(top_x), 0),
                      (int(bottom_x), colour_bar.shape[0]), colour, -1)
        top_x = bottom_x
    return colour_bar
    
    
def pretty_print_data(colour_info):
        for x in colour_info:
            print(pprint.pformat(x))
            print()
        
 
def main():

	activities = ["Detection","About Me"]
	choice = st.sidebar.selectbox("Menu",activities)
    
	if choice== 'About Me':
         st.subheader("Welcome")

         st.write("""Hello, my name is Adel Takawira and I am fifteen years old. I attend the international school of the hague, and for my personal project, I created a colour detecting software. Coming up with this idea required a lot of brainstorming on my part. But after combining my love for skincare and technology, I came up with this idea.""")
         st.write("""Many people struggle to find a foundation or concealer that is an exact match to their skin tone, especially dark skinned people. For a long time they would have to resort to mixing two different shades in order get a colour that was somewhat close to their natural tone. This is where I had the idea to create this software that detects, and extracts a person’s most dominant skin colour, then displays it in a colourbar. This information can be used by cosmetic companies, to create a customized foundation/concealer that is suitable for their customers unique tone of skin. 
        The great thing about this software is that it can be used, not only for cosmetic purposes, but also to customize ‘skin coloured’ tights.""")
        
         st.write("""The way my software works is, using a method called K-mean clustering. This lets me take an image and group its pixels according to their colour. The group with the most (similar) coloured pixels will be the most dominant colour of your skin. But unfortunately for me it was not that easy, before all of that I had to apply skin detection to make sure the k means didn’t count any background colours (that isn’t skin).
        """)

	if choice== 'Detection':
         st.subheader("Skin and Colour Detection")
         st.write("""  * Take the picture in front of a plain background (ideally a white background)""")
         ("""  * Try to fit only your face in the frame (exclude your clothes, hair and jewelry etc)""")
         ("""  * Use a fairly good quality photo (taken from your phone is fine)""")
         ("""  * Make sure there is enough lighting to see your face""")
         ("""  * When you select your image file, wait until the image is visible before clicking the process button""")

         image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

         if image_file is not None:
            image = Image.open(image_file)
            st.text("Original Image")
            st.image(image)
            #Converting Uploaded IOByte image into OpenCV Image
            image1 = cv2.cvtColour(np.array(image),cv2.COLOR_BGR2RGB)
                       
         task = ["Skin Colour Bar"]
         feature_choice = st.sidebar.selectbox("Choose Features",task)
         if st.button("Process"):

            if feature_choice== "Skin Colour Bar":
                   skin = detectSkin(image1)
                   dominantColours = extractDominantTone(skin, hasThresholding=True)
                   colour_bar = graphColourBar(dominantColours)
                   st.text("Colour Bar")
                   st.image(colour_bar)
                   st.text("Thresholded Image")
                   st.image(cv2.cvtColor(skin, cv2.COLOR_RGB2BGR))
                   image2   = cv2.resize(image1,(500,500))
                   #cv2.imshow("Converted OpenCV Image", image2)
                   skin1   = cv2.resize(skin,(500,500))
                  # cv2.imshow("StreamLIT Segmented Image", skin1)
                   #cv2.waitKey()
                   

if __name__ == '__main__':
	main() 
