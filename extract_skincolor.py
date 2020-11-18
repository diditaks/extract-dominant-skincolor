#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter
import pprint
from matplotlib import pyplot as plt
import streamlit as st
from PIL import Image

st.write("""
# Find Your Dominant Skincolor 

Have you struggled to find the right foundation and concealer because you don't know your exact skincolor? Or perhaps, you weren't able to find the right color tights to go with your unique skin tone?

""")

    
def extractSkin(image):
    
    img = image.copy()
    
    # Converting from BGR Colours Space to HSV
    img = np.array(img)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Defining HSV Threadholds
    lower = np.array([0, 48, 80], dtype=np.uint8)
    upper = np.array([20, 124, 124], dtype=np.uint8)

    # Single Channel mask,denoting presence of colours in the about threshold
    skinMask = cv2.inRange(img, lower, upper)

    # Cleaning up mask using Gaussian Filter
    
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    
        # Extracting skin from the threshold mask
    skin = cv2.bitwise_and(img, img, mask=skinMask)

    # Return the Skin image
    return cv2.cvtColor(skin, cv2.COLOR_HSV2BGR)


def removeBlack(estimator_labels, estimator_cluster):

    # Check for black
    hasBlack = False

    # Get the total number of occurance for each color
    occurance_counter = Counter(estimator_labels)

    # Quick lambda function to compare to lists
    def compare(x, y): return Counter(x) == Counter(y)

    # Loop through the most common occuring color
    for x in occurance_counter.most_common(len(estimator_cluster)):

        # Quick List comprehension to convert each of RBG Numbers to int
        color = [int(i) for i in estimator_cluster[x[0]].tolist()]

        # Check if the color is [0,0,0] that if it is black
        if compare(color, [0, 0, 0]) == True:
            # delete the occurance
            del occurance_counter[x[0]]
            # remove the cluster
            hasBlack = True
            estimator_cluster = np.delete(estimator_cluster, x[0], 0)
            break

    return (occurance_counter, estimator_cluster, hasBlack)


def getColorInformation(estimator_labels, estimator_cluster, hasThresholding=False):

    # Variable to keep count of the occurance of each color predicted
    occurance_counter = None

    # Output list variable to return
    colorInformation = []

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

    # Loop through all the predicted colors
    for x in occurance_counter.most_common(len(estimator_cluster)):

        index = (int(x[0]))

        # Quick fix for index out of bound when there is no threshold
        index = (index-1) if ((hasThresholding & hasBlack)
                              & (int(index) != 0)) else index

        # Get the color number into a list
        color = estimator_cluster[index].tolist()

        # Get the percentage of each color
        color_percentage = (x[1]/totalOccurance)

        # make the dictionay of the information
        colorInfo = {"cluster_index": index, "color": color,
                     "color_percentage": color_percentage}

        # Add the dictionary to the list
        colorInformation.append(colorInfo)

    return colorInformation


def extractDominantColor(image, number_of_colors=5, hasThresholding=False):

    # Quick Fix Increase cluster counter to neglect the black(Read Article)
    if hasThresholding == True:
        number_of_colors += 1

    # Taking Copy of the image
    img = image.copy()

    # Convert Image into RGB Colours Space
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Reshape Image
    img = img.reshape((img.shape[0]*img.shape[1]), 3)

    # Initiate KMeans Object
    estimator = KMeans(n_clusters=number_of_colors, random_state=0)

    # Fit the image
    estimator.fit(img)

    # Get Colour Information
    colorInformation = getColorInformation(
        estimator.labels_, estimator.cluster_centers_, hasThresholding)
    return colorInformation


def plotColorBar(colorInformation):
    # Create a 500x100 black image
    color_bar = np.zeros((100, 500, 3), dtype="uint8")

    top_x = 0
    for x in colorInformation:
        bottom_x = top_x + (x["color_percentage"] * color_bar.shape[1])

        color = tuple(map(int, (x['color'])))

        cv2.rectangle(color_bar, (int(top_x), 0),
                      (int(bottom_x), color_bar.shape[0]), color, -1)
        top_x = bottom_x
    return color_bar
    
    
def pretty_print_data(color_info):
        for x in color_info:
            print(pprint.pformat(x))
            print()
        
 
def main():

	activities = ["Detection","About Me"]
	choice = st.sidebar.selectbox("Menu",activities)
    
	if choice== 'About Me':
         st.subheader("Welcome")
         imge = Image.open('makeup.png')

         st.write("""Hello, my name is Adel Takawira and I am fifteen years old. I attend the international school of the hague, and for my personal project, I created a color detecting software. Coming up with this idea required a lot of brainstorming on my part. But after combining my love for skincare and technology, I came up with this idea.""")
         st.image(imge, width=None)
         st.write("""Many people struggle to find a foundation or concealer that is an exact match to their skin tone, especially dark skinned people. For a long time they would have to resort to mixing two different shades in order get a color that was somewhat close to their natural tone. This is where I had the idea to create this software that detects, and extracts a person’s most dominant skin color, then displays it in a colorbar. This information can be used by cosmetic companies, to create a customized foundation/concealer that is suitable for their customers unique tone of skin. 
        The great thing about this software is that it can be used, not only for cosmetic purposes, but also to customize ‘skin colored’ tights.""")
        
         st.write("""The way my software works is, using a method called K-mean clustering. This lets me take an image and group its pixels according to their color. The group with the most (similar) colored pixels will be the most dominant color of your skin. But unfortunately for me it was not that easy, before all of that I had to apply skin detection to make sure the k means didn’t count any background colors (that isn’t skin).
        """)

	if choice== 'Detection':
         imagg = Image.open('FSnm.png')
         st.image(imagg, width=None)
         st.subheader("Skin and Color Detection")
         st.write("""  * Take the picture in front of a plain background (ideally a white background)""")
         ("""  * Try to fit only your face in the frame (exclude your clothes and jewelry etc)""")
         ("""  * Use a fairly good quality photo (taken from your phone is fine)""")
         ("""  * Make sure there is enough lighting to see your face""")

         image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

         if image_file is not None:
            image = Image.open(image_file)
            st.text("Original Image")
            st.image(image)

         task = ["Skin Color Bar"]
         feature_choice = st.sidebar.selectbox("Choose Features",task)
         if st.button("Process"):

            if feature_choice== "Skin Color Bar":
                   skin = extractSkin(image)
                   dominantColors = extractDominantColor(skin, hasThresholding=True)
                   color_bar = plotColorBar(dominantColors)
                   st.image(color_bar)
                   

if __name__ == '__main__':
	main()
    
