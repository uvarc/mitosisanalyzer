#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:42:03 2021

@author: khs3z
"""

import os
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np

def get_centers(contour):
    M = cv2.moments(contour)
    return (int(M['m10']/M['m00']),int(M['m01']/M['m00']))

def get_rect_points(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)    

def euclidian(edge=None, p1=None, p2=None):
    if edge is not None:
        p1 = edge[0]
        p2 = edge[1]
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def get_edges(corners,length_sort=True):
    """creates list of edges sorted by length of edge (ascending)"""
    edges = [(corners[i],corners[i+1]) for i in range(len(corners)-1)]
    edges.append((corners[len(corners)-1], corners[0]))
    edges.sort(key=euclidian)
    return edges
    
def draw_lines(img, lines, color=(255,255,255),thickness=1):
    for l in lines:
        img = cv2.line(img,l[0],l[1],color,thickness)
    return img

def center(p1,p2):
    c = (int(0.5*(p1[0]+p2[0])),int(0.5*(p1[1]+p2[1])))
    return c

def process_spindle(img):
    img = cv2.GaussianBlur(img,(3,3),0)
    hist,bins = np.histogram(img.ravel(),256,[0,256])
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)
    ret1,binary = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
    
    sp_contours,hierarchy = cv2.findContours(binary, 1, 2)
    print (f'\tcontours: {len(sp_contours)}')
    poles = [get_centers(c) for c in sp_contours]
    if len(sp_contours) > 1:
        color = (255,255,255)
        thickness = 2
        binary = cv2.line(binary, poles[0], poles[1], color, thickness)
        sp_contours,hierarchy = cv2.findContours(binary, 1, 2)
    corners = get_rect_points(sp_contours[0])
    edges = get_edges(corners)
    spindle_poles = np.array([center(e[0], e[1]) for e in edges[:2]])
    print ('\tspindle poles:',','.join([str(p) for p in spindle_poles]))
    img = draw_lines(img,edges[:2],color=(255,0,0),thickness=1)
    img = cv2.line(img,spindle_poles[0], spindle_poles[1],(255,0,0),1)
    return img,binary,spindle_poles

    #ret2,binary2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

def process_dna(img):
    pass

def get_files(path, fpattern='*.tif'):
    files = glob.glob(os.path.join(path,fpattern))
    files.sort()
    return files
    
def main():
    path = os.path.join('data', 'wt-1')
    dna_files = get_files(path,fpattern='C1*.tif')
    sp_files = get_files(path,fpattern='C2*.tif')
    poles = []
    images = []
    for fname in sp_files:
        print (f'processing {fname}')
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        img,_,spindle_poles = process_spindle(img)
        poles.append(spindle_poles)
        images.append(cv2.cvtColor(img,cv2.COLOR_GRAY2RGB))
    poles = np.array(poles)
    sp_centers = [center(p[0], p[1]) for p in poles] 
    pole1_distance = [euclidian(p1=poles[i][0], p2=sp_centers[i]) for i in range(len(poles))]
    pole2_distance = [-euclidian(p1=poles[i][1], p2=sp_centers[i]) for i in range(len(poles))]
    plt.plot(pole1_distance, color='green')
    plt.plot(pole2_distance, color='magenta')
    plt.show()
    
    height, width = img.shape
    size = (width,height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vout = cv2.VideoWriter()
    success = vout.open('project.mp4',fourcc, 15, size, True)
    for img in images:
        vout.write(img)
    vout.release()
    
    cv2.imshow('image',img)
    cv2.waitKey()
    for name in dna_files:
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        process_dna(img)


if __name__ == '__main__':
    main()