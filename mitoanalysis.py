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
from nd2reader import ND2Reader

def get_files(path, fpattern='*.tif'):
    files = glob.glob(os.path.join(path,fpattern))
    files.sort()
    return files
    
def get_centers(contour):
    M = cv2.moments(contour)
    if M['m00'] == 0.0:
        return (0,0)
    else:    
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

def square_edges(edges):
    sedges = [e for e in edges]
    scorners = []
    for e in edges:
        p1 = e[0]
        p2 = e[1]
        dx = p1[0]-p2[0]
        dy = p1[1]-p2[1]
        p3 = (p2[0]+dy,p2[1]-dx)
        dx = p1[0]-p3[0]
        dy = p1[1]-p3[1]        
        p4 = (p2[0]+dy,p2[1]-dx)
        sedges.append((p2,p3))
        sedges.append((p3,p4))
        sedges.append((p4,p1))
        points = np.array([p1,p2,p3,p4])
        xmin = points[:,0].min()
        xmax = points[:,0].max()
        ymin = points[:,1].min()
        ymax = points[:,1].max()
        #region = img[xmin:xmax+1,ymin:ymax+1]
        #print (region.max())
        #centerx,centery = np.unravel_index(np.argmax(region, axis=None), region.shape)
        #centerx = int(0.5 * (xmin + xmax)) 
        #centery = int(0.5 * (ymin + ymax)) 
        #sedges.append((p1,(centerx,centery)))
        
        scorners.append([p1,p2,p3,p4])
    return sedges,scorners
    
def process_spindle(image):
    height,width = image.shape
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #image = clahe.apply(image)
    blurredimg = cv2.GaussianBlur(image,(3,3),0)
    hist,bins = np.histogram(blurredimg.ravel(),256,[0,256])
    
    ret1,binary = cv2.threshold(blurredimg,191,255,cv2.THRESH_BINARY)
    
    sp_contours,hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print (f'\tcontours: {len(sp_contours)}')
    if len(sp_contours) == 0:
        return image,binary,np.array([(0,0),(0,0)]) 
    poles = [get_centers(c) for c in sp_contours]
    if len(sp_contours) > 1:
        color = (255,255,255)
        thickness = 2
        binary = cv2.line(binary, poles[0], poles[1], color, thickness)
        sp_contours,hierarchy = cv2.findContours(binary, 1, 2)
    corners = get_rect_points(sp_contours[0])
    print (f'corners: {corners}', type(corners[0]))
    edges = get_edges(corners)
    print (f'edges: {edges}')
    edges,scorners = square_edges(edges[:2])
    print (f'square corners: {scorners}')
    poles = []
    for square in scorners[:2]:
        mask = np.zeros((height,width),dtype=np.uint8)
        square = np.array(square).flatten().reshape((len(square),2))
        print (square)
        cv2.fillPoly(mask, pts = [square], color =(255,255,255))
        img = cv2.bitwise_and(blurredimg,blurredimg,mask = mask)
        ret1,binary = cv2.threshold(img,191,255,cv2.THRESH_BINARY)    
        sp_contours,hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        poles.append(get_centers(sp_contours[0]))

    #image = draw_lines(image,edges,color=(255,0,0),thickness=1)
    print (f'spindle poles: {poles}')
    return image,binary,poles #spindle_poles

    #ret2,binary2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

def process_dna(img):
    return img,[],[]

def save_movie(images, fname, codec='mp4v'):
    height,width,_ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    vout = cv2.VideoWriter()
    success = vout.open(fname,fourcc, 15, (width,height), True)
    for img in images:
        vout.write(img)
    vout.release()
    return success

def process_file(fname):
    with ND2Reader(fname) as imagestack:
        print (imagestack.sizes)
        print (imagestack.axes)
        print (imagestack.frame_shape)
        poles = []
        spimages = []
        dnaimages = []
        imagestack.iter_axes = 'tc'
        width = imagestack.sizes['x']
        height = imagestack.sizes['y']
        print (imagestack.frame_shape)
        for i,img in enumerate(imagestack):
            print (f'i={i}, img.shape={img.shape}')
            # img =  cv2.normalize(img, None, 0, int(255.0*img.max()/65535.0), norm_type=cv2.NORM_MINMAX)
            img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX)
            img = np.uint8(img)

            if i % 2 == 0:
                dnaimg,binary,dnablobs = process_dna(img)
                dnaimages.append(dnaimg)
            else:     
                spimg,binary,spindle_poles = process_spindle(img)
                poles.append(spindle_poles)
                spimages.append(spimg)
        blue = np.zeros((width,height), np.uint8)
        images = [cv2.merge([blue,spimages[i], dnaimages[i]]) for i in range(len(spimages))]
        for i,frame_poles in enumerate(poles):
            for p in frame_poles:
                images[i] = cv2.circle(images[i], (p[0],p[1]), 4, (255,0,255), 1)
        #plt.subplot(121)
        #plt.imshow(spimg, cmap='gray')
        #plt.subplot(122)
        #plt.imshow(binary, cmap='gray')
        poles = np.array(poles)
        sp_centers = [center(p[0], p[1]) for p in poles] 
        pole1_distance = [euclidian(p1=poles[i][0], p2=sp_centers[i]) for i in range(len(poles))]
        pole2_distance = [-euclidian(p1=poles[i][1], p2=sp_centers[i]) for i in range(len(poles))]
        #plt.plot(pole1_distance, color='green')
        #plt.plot(pole2_distance, color='magenta')
        #plt.show()
    
        moviefile = os.path.splitext(fname)[0] + '.mp4'
        save_movie(images, moviefile)    
        cv2.imshow('image',spimg)

def main():
    files = get_files('data',fpattern='*.nd2')
    print (files)
    for fname in files:
        print (f'Processing: {fname}')
        process_file(fname)
    plt.show()
    cv2.waitKey()
  


if __name__ == '__main__':
    main()