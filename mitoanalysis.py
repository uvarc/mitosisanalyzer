#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:42:03 2021

@author: khs3z
"""

import os
import argparse
import glob
from typing import Tuple
from multiprocessing import Pool
from functools import partial
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import label
from skimage.morphology import skeletonize, medial_axis
from skimage.measure import profile_line
from nd2reader import ND2Reader
import pandas as pd
import math
from prefect import task, Flow, unmapped

RES_UNIT_DICT = {1:'<unknown>', 2:'inch', 3:'cm'}


def init_parser():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(
        description='Analyzes spindle pole and chromatid movements in .nd2 timelapse files')
    parser.add_argument('-i', '--input', required=True, help='.nd2 file or directory with .nd2 files to be processed')
    parser.add_argument('-o', '--output', default=None, help='output file or directory')
    parser.add_argument('-p', '--processes', default=1, type=int, help='number or parallel processes')
    parser.add_argument('-s', '--spindle', default=2, type=int, help='channel # for tracking spindle poles')
    parser.add_argument('-d', '--dna', default=1, type=int, help='channel # for tracking dna')
    parser.add_argument('-f', '--framerate', default=None, type=float, help='number of frames per second')
    return parser

def get_files(path, fpattern='*.tif') -> list:
    """Find files in a directory matching a customizable file name pattern"""
    files = []
    if os.path.isfile(path):
        files = [path]
    elif os.path.isdir(path):    
        for pattern in fpattern:
            files.extend(glob.glob(os.path.join(path,pattern)))
        # remove possible duplicates and sort
        files = list(set(files))
        files.sort() 
    return files
    
def get_centers(contour):
    """Extract center x/y coordinates from a contour object"""
    M = cv2.moments(contour)
    if M['m00'] == 0.0:
        return (0,0)
    else:    
        return (int(M['m10']/M['m00']),int(M['m01']/M['m00']))

def get_rect_points(contour):
    """Get corner point of smallest rectangle enclosing a contour"""
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return np.int0(box)    


def euclidian(edge=None, p1=None, p2=None):
    """Calculates the euclidian distance between two points"""
    if edge is not None:
        p1 = edge[0]
        p2 = edge[1]
    return np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)

def get_edges(corners,length_sort=True):
    """Creates list of edges defined by pairs of consecutive vertices. Optional: the edges 
    may be sorted by length (ascending)"""
    edges = [(corners[i],corners[i+1]) for i in range(len(corners)-1)]
    edges.append((corners[len(corners)-1], corners[0]))
    edges.sort(key=euclidian)
    return edges
    
def draw_lines(img, lines, color=(255,255,255),thickness=1):
    """Draws a group of lines to image using specified color and line thickness"""
    for l in lines:
        img = cv2.line(img,l[0],l[1],color,thickness)
    return img

def center(p1,p2):
    """Get the geometric center of two points"""
    cx = int(0.5*(p1[0]+p2[0]))
    cy = int(0.5*(p1[1]+p2[1]))
    return (cx,cy)

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

def register_stack(imagestack):
    return imagestack

def watershed(a, img, dilate=5, erode=5, relthr=0.7):
    """Separate joined objects in binary image via watershed"""
    border = cv2.dilate(img, None, iterations=dilate)
    border = border - cv2.erode(border, None)

    dt = cv2.distanceTransform(img, cv2.DIST_L2, 3)
    dt = ((dt - dt.min()) / (dt.max() - dt.min()) * 255).astype(np.uint8)
    _, dt = cv2.threshold(dt, relthr*255, 255, cv2.THRESH_BINARY)
    lbl, ncc = label(dt)
    lbl = lbl * (255 / (ncc + 1))
    # Completing the markers now. 
    lbl[border == 255] = 255

    lbl = lbl.astype(np.int32)
    odt = lbl.astype(np.uint8)
    markers = cv2.watershed(a, lbl)

    lbl[lbl == -1] = 0
    lbl = lbl.astype(np.uint8)
    masks = []
    #print (f'ncc={ncc}, unique={np.unique(lbl)}')
    for i in np.unique(lbl):
        if i == 0 or i == 255:
            continue
        mask = np.zeros(img.shape,np.uint8)
        mask[lbl==i] = 255
        mask = cv2.erode(mask, None, iterations=erode)
        masks.append(mask)
    return odt, 255 - lbl, masks
    
def find_embryos(channelstack, channel=0):
    med = np.median(channelstack, axis=0)
    med = cv2.normalize(med, None, 0, 255.0, norm_type=cv2.NORM_MINMAX)
    med = med.astype(np.uint8)
    kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
    #kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
    f2d = cv2.filter2D(med, -1, kernel)
    __,th = cv2.threshold(f2d,0.75*255,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #embryo = cv2.adaptiveThreshold(med,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    #embryo = cv2.morphologyEx(embryo, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50)))
    
    #cv2.imshow('med', med)
    #cv2.imshow('thresholded', th)
    #cv2.imshow('f2d', f2d)
    
    """
    ret, thresh = cv2.threshold(med,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    cv2.imshow('sure fg', sure_fg)
    cv2.imshow('uncertain', unknown)
    """
    
    kernel = np.ones((3,3),np.uint8)
    #embryo = cv2.morphologyEx(embryo,cv2.MORPH_OPEN, kernel, iterations = 2)
    dt, embryo, masks = watershed(cv2.merge([th,th,th]),th,relthr=0.8)
    #cv2.imshow('embryo overlay', cv2.merge([embryo,med,embryo]))
    #cv2.waitKey(0)
    return masks


def rotate_image(image, angle, center=None):
    """Rotate image around image reference point"""
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    if center is not None:
        dx = (image.shape[1]/2) - center[0]
        dy = (image.shape[0]/2) - center[1]
        #print (center, dx,dy, angle)
    
        transl_mat= np.float32([[1, 0, dx],[0, 1, dy]])
        image = cv2.warpAffine(image, transl_mat, (image.shape[1], image.shape[0]))
    else:
        center = (image.shape[1]/2, image.shape[0]/2)    
    return image
    
    
def process_spindle(image, polesize=20):
    height,width = image.shape
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #image = clahe.apply(image)
    blurredimg = cv2.GaussianBlur(image,(3,3),0)
    hist,bins = np.histogram(blurredimg.ravel(),256,[0,256])
    
    ret1,binary = cv2.threshold(blurredimg,191,255,cv2.THRESH_BINARY)
    
    sp_contours,hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(sp_contours) == 0:
        return image,binary,np.array([(0,0),(0,0)])
    #print (f'type(sp_contours)={type(sp_contours)}') 
    #print (f'type(sp_contours[0])={type(sp_contours[0])}') 
    sp_contours.sort(key=cv2.contourArea, reverse=True)
    if len(sp_contours) > 2:
        sp_contours = sp_contours[:2]
    #for c in sp_contours:
    #    print (f'\tarea={cv2.contourArea(c)}')
    #   print (f'\tcenter={get_centers(c)}')
    binary = np.zeros((height,width),np.uint8)
    cv2.fillPoly(binary, sp_contours, (255,255,255))
    poles = [get_centers(c) for c in sp_contours]
    if len(sp_contours) > 1:
        color = (255,255,255)
        thickness = 2
        binary = cv2.line(binary, poles[0], poles[1], color, thickness)
        sp_contours,hierarchy = cv2.findContours(binary, 1, 2)
    #print (f'\tlen(sp_contours): {len(sp_contours)}', flush=True)
    corners = get_rect_points(sp_contours[0])
    #print (f'\tcorners: {corners}', type(corners[0]))
    edges = get_edges(corners)
    #print (f'\tedges: {edges}')
    edges,scorners = square_edges(edges[:2])
    #print (f'\tsquare corners: {scorners}')
    poles = []
    binary = np.zeros((height,width),np.uint8)
    for square in scorners[:2]:
        mask = np.zeros((height,width),dtype=np.uint8)
        square = np.array(square).flatten().reshape((len(square),2))
        #print (f'\tsquare={square}')
        cv2.fillPoly(mask, pts=[square], color=(255,255,255))
        img = cv2.bitwise_and(blurredimg,blurredimg,mask = mask)
        ret1,tmpbinary = cv2.threshold(img,191,255,cv2.THRESH_BINARY)    
        sp_contours,hierarchy = cv2.findContours(tmpbinary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(sp_contours) == 1 :
           #print (f'scorners={scorners}, square={square}')
           #cv2.imshow('t',blurredimg)
           #cv2.waitKey(0)
           cv2.fillPoly(binary, pts=sp_contours, color=(255,255,255))
           poles.append(get_centers(sp_contours[0]))
        else:
           #cv2.fillPoly(binary, pts=sp_contours, color=(255,255,255))
           poles.append([np.nan,np.nan])

    #image = draw_lines(image,edges,color=(255,0,0),thickness=1)
    #print (f'\tspindle poles: {poles}')
    return image,binary,poles,corners #spindle_poles

    #ret2,binary2 = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

def process_dna(image):
    chromatids = []
    height,width = image.shape
    blurredimg = cv2.medianBlur(image,3,0)
    ret1,binary = cv2.threshold(blurredimg,127,255,cv2.THRESH_BINARY)
    #binary = cv2.adaptiveThreshold(blurredimg, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    
    dna_contours,hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    binary = np.zeros((width,height), np.uint8)
    cv2.fillPoly(binary, pts=dna_contours,color=(255,255,255))
    chromatids = [get_centers(c) for c in dna_contours]
    
    #dt, dna, binaries = watershed(cv2.merge([binary,binary,binary]),binary,dilate=1,erode=1,relthr=0.1)
    #chromatids = []
    #for b in binaries:
    #    dna_contours,hierarchy = cv2.findContours(b, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #    chromatids.extend([get_centers(c) for c in dna_contours])
    ##print (chromatids)
    return image,binary,chromatids

def save_movie(images, fname, codec='mp4v'):
    height,width,_ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*codec)
    vout = cv2.VideoWriter()
    success = vout.open(fname,fourcc, 15, (width,height), True)
    for img in images:
        vout.write(img)
    vout.release()
    return success

def profile_endpoints(p1,p2,center,length):
    dx = p1[0]-p2[0]
    dy = p1[1]-p2[1]
    l = np.sqrt(dx*dx + dy*dy)
    if l > 0.0:
        dx = dx/l
        dy = dy/l
        xoffset = int(dx*length*0.5)
        yoffset = int(dy*length*0.5)
        end1 = (center[0]+xoffset, center[1]+yoffset)
        end2 = (center[0]-xoffset, center[1]-yoffset)
        return (end1,end2)
    else:
        return (0,256),(512,256)


def get_angle(p1, p2):
    radians = math.atan2(p1[1]-p2[1], p1[0]-p2[0])
    angle = math.degrees(radians)
    return angle
    

def get_row_angle(r):
    p1 = (r[0], r[1])
    p2 = (r[2], r[3])
    a = get_angle(p1,p2)
    if a < 0:
        a = a + 360
    return a
    
def get_row_euclidian(r, pixel_res):
    p1 = (r[0], r[1])
    p2 = (r[2], r[3])
    dist = pixel_res * euclidian(p1=p1,p2=p2)
    return dist
    
def create_dataframe(allpoles, allchromatids, pixel_res=1.0, pixel_unit='um', rolling=3):
    # reshape and replace 0/0 coordinates with nan
    polearray = np.array(allpoles).reshape(len(allpoles), 4)
    polearray = np.where(polearray<=0, np.nan, polearray)

    df = pd.DataFrame(polearray, columns=['Pole 1,x (pixel)', 'Pole 1,y (pixel)', 'Pole 2,x (pixel)', 'Pole 2,y (pixel)'])
    #df['Pole 1,x (pixel)'] = df['Pole 1,x (pixel)'].rolling(rolling).mean()
    #df['Pole 1,y (pixel)'] = df['Pole 1,y (pixel)'].rolling(rolling).mean()
    #df['Pole 2,x (pixel)'] = df['Pole 2,x (pixel)'].rolling(rolling).mean()
    #df['Pole 2,y (pixel)'] = df['Pole 2,y (pixel)'].rolling(rolling).mean()
    df['Midzone,x (pixel)'] = 0.5 * (df['Pole 2,x (pixel)'] + df['Pole 1,x (pixel)'])
    df['Midzone,y (pixel)'] = 0.5 * (df['Pole 2,y (pixel)'] + df['Pole 1,y (pixel)'])
    # find nan values, forward and backfill
    outliers = df.isna().any(axis=1)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    df['angle'] = df.apply (lambda row: get_row_angle(row), axis=1)
    med_angle = df['angle'].median()
    swap = (df['angle']-med_angle).abs() > 90

    df.loc[swap, ['Pole 1,x (pixel)','Pole 2,x (pixel)']] = (df.loc[swap, ['Pole 2,x (pixel)','Pole 1,x (pixel)']].values)
    df.loc[swap, ['Pole 1,y (pixel)','Pole 2,y (pixel)']] = (df.loc[swap, ['Pole 2,y (pixel)','Pole 1,y (pixel)']].values)
    df['angle'] = df.apply (lambda row: get_row_angle(row), axis=1)

    # calculate pole distance
    df[f'Pole-Pole Distance [{pixel_unit}]'] = df.apply(lambda row: get_row_euclidian(row, pixel_res), axis=1)
    df['Pole 1 [pixel]'] = '(' + df['Pole 1,x (pixel)'].astype(str) + '/'+ df['Pole 1,y (pixel)'].astype(str) +')'
    df['Pole 2 [pixel]'] = '(' + df['Pole 2,x (pixel)'].astype(str) + '/'+ df['Pole 2,y (pixel)'].astype(str) +')'
    df['Midzone [pixel]'] = '(' + df['Midzone,x (pixel)'].astype(str) + '/'+ df['Midzone,y (pixel)'].astype(str) +')'
    df['Frame'] = np.arange(1, len(allpoles)+1)
    #df = df[['Frame', 'Pole 1 [pixel]', 'Pole 2 [pixel]', f'Pole-Pole Distance [{pixel_unit}]']]
    df = df.set_index('Frame')
    mean = df[f'Pole-Pole Distance [{pixel_unit}]'].mean()
    median = df[f'Pole-Pole Distance [{pixel_unit}]'].median()
    std = df[f'Pole-Pole Distance [{pixel_unit}]'].std()
    print (f'mean={mean}, median={median}, std={std}')
    valid = median != 0.0 and mean > 3. and mean/median > 0.8 and mean/median < 1.2 and std < 0.4*mean
    allpoles = df.iloc[:,0:4].values
    allpoles = allpoles.reshape((len(allpoles),2,2))

    return df,allpoles,True #valid

def nd2_opener(fname) -> Tuple[np.array, dict]:
    metadata = {}
    with ND2Reader(fname) as imagestack:
        pixelres = imagestack.metadata['pixel_microns']
        metadata['shape'] = imagestack.sizes
        metadata['axes'] = ['t', 'c', 'y', 'x'] #imagestack.axes
        metadata['pixel_unit'] = 'um'
        metadata['pixel_res'] = pixelres
        metadata['scale'] = f"{1/metadata['pixel_res']} pixels per {metadata['pixel_unit']}"
        #set order tcyx and convert to np array
        imagestack.bundle_axes = 'cyx'
        imagestack.iter_axes = 't'
        imagestack = np.array(imagestack)

    return imagestack, metadata

def tif_opener(fname) -> Tuple[np.array, dict]:
    from tifffile import imread, TiffFile
    imagestack = imread(fname)
    tif = TiffFile(fname)
    tags = tif.pages[0].tags 
    #for t in [t for t in tags if "ij" not in str(t).lower()]:
    #    print (type(t), t, t.value)
    metadata = {}
    unit = RES_UNIT_DICT[tags['ResolutionUnit'].value]
    axesorder = [o.lower() for o in tif.series[0].axes]
    metadata['axes'] = axesorder
    metadata['shape'] = {axesorder[i]:s for i,s in enumerate(imagestack.shape)}
    metadata['pixel_unit'] = 'um'
    pixelres = tags['XResolution'].value[1]/tags['XResolution'].value[0]
    if unit == 'cm':
        metadata['pixel_res'] = pixelres * 10000
    elif unit == 'inch':
        metadata['pixel_res'] = pixelres * 25400     
    else:
        metadata['pixel_res'] = pixelres   
    metadata['scale'] = f"{tags['XResolution'].value[0]/tags['XResolution'].value[1]} pixels per {unit}"
    return imagestack, metadata

def skip_opener(fname) -> Tuple[np.array, dict]:
    return None, None
    
def get_opener(fname): #-> Tuple[np.array, dict]:
    ext = os.path.splitext(fname)[1]
    if ext == '.nd2':
        return nd2_opener
    elif ext in ['.tif', '.tiff']:
        return tif_opener
    return skip_opener

def crop_stack(imagestack, width, height):
    #print (imagestack.shape)
    x1 = int((imagestack[0].shape[1] - width)/2)
    y1 = int((imagestack[0].shape[0] - height)/2)
    x2 = x1 + width
    y2 = y1 + height
    cropped = np.array([img[y1:y2,x1:x2] for img in imagestack])
    return cropped

def kymograph(images, allpoles, width=200, height=10, method='sum'):
    allangles = np.array([get_angle(pole1,pole2) for (pole1,pole2) in allpoles])
    allcenters = [center(pole1, pole2) for (pole1,pole2) in allpoles]
    #print (np.median(allangles), np.min(allangles), np.max(allangles))
    rotated_images = [rotate_image(images[i], allangles[i], center=allcenters[i]) for i in range(len(images))]
    cropped_rot = crop_stack(rotated_images, width, height)
    if method == 'sum':
	    kymo = np.array([np.sum(img, axis=0) for img in cropped_rot])
    else:
        kymo = np.array([np.max(img, axis=0) for img in cropped_rot])
    kymo = cv2.normalize(kymo, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    return kymo, cropped_rot

def enhanced_kymograph(kymo, allpoles, relthr=0.7, padding=10):
    poledistances = np.array([euclidian(p1=pole1,p2=pole2) for (pole1,pole2) in allpoles])
    dna_kymo = kymo[:,:,2]
    width = dna_kymo.shape[1]
    dna_kymo = cv2.medianBlur(dna_kymo,3,0)    
    thresh = cv2.adaptiveThreshold(dna_kymo, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 7) # need to rescale 0-1 for skeletoonize
    for row in range(len(kymo)): 
        thresh[row,0:int(0.5*(width+padding-poledistances[row]))] = 0
        thresh[row,int(0.5*(width-padding+poledistances[row])):-1] = 0	
    ekymo = cv2.distanceTransform(thresh, cv2.DIST_L2, 3).astype(np.uint8)
    ekymo = cv2.normalize(ekymo, None, 0, 255.0, norm_type=cv2.NORM_MINMAX)
    return ekymo

@task
def process_file(fname, spindle_ch, dna_ch, output):
    print (f'Processing: {fname}, spindle channel:{spindle_ch}, dna channel:{dna_ch}')
    opener = get_opener(fname)
    imagestack, metadata = opener(fname)
    print (f'\t{opener}')
    for k,v in metadata.items():
        print (f'\t{k}:{v}')
    if max(spindle_ch, dna_ch) > metadata['shape']['c']:
        print ("Skipping -- not enough channels.")
        return
    #pixel_microns = imagestack.metadata['pixel_microns']

    imagestack = register_stack(imagestack)

    width = metadata['shape']['x'] #imagestack.sizes['x']
    height = metadata['shape']['y'] #imagestack.sizes['y']
    
    #imagestack.bundle_axes = 'cxy'
    #imagestack.iter_axes = 't'
    
    
    #max_spindle_int = np.amax(np.array(imagestack)[:,1])
    #max_dna_int = np.amax(np.array(imagestack)[:,0])

    spindle_stack = np.array(imagestack)[:,spindle_ch-1]
    dna_stack = np.array(imagestack)[:,dna_ch-1]
    embryo_masks = find_embryos(spindle_stack)
    
    blank = np.zeros((height,width), np.uint8)

    for embryo_no,embryo in enumerate(embryo_masks):
        allpoles = []
        allcorners = []
        allchromatids = []
        spimages = []
        dnaimages = []
        dnabinimages = []
        for frame_no, spimg in enumerate(spindle_stack):
            #spimg = frame[spindle_ch-1]
            spimg = cv2.normalize(spimg, None, 0, 255.0, norm_type=cv2.NORM_MINMAX)
            spimg = np.uint8(spimg)
            spimages.append(spimg) # spimg
            #frame_int = np.amax(spimg)
            spimg = cv2.bitwise_and(spimg,spimg,mask = embryo)
            spimg = cv2.normalize(spimg, None, 0, 255.0, norm_type=cv2.NORM_MINMAX)
            #if (frame_no==0):
            #    cv2.imshow('spimg',spimg)
            #    cv2.waitKey(0)
            #print (f'embryo_no={embryo_no},frame_no={frame_no}')
            spimg,binary,spindle_poles,corners = process_spindle(spimg)
            #if len(spindle_poles) == 2:
            #    end1,end2 = profile_endpoints(spindle_poles[0], spindle_poles[1], center(spindle_poles[0], spindle_poles[1]), 100)
            #    profile = profile_line(spimg, end1, end2)
            #    print (f'profile={profile}')
            allpoles.append(spindle_poles)  #(end1,end2)
            allcorners.append(corners)
            
        for frame_no, dnaimg in enumerate(dna_stack):
            #dnaimg = frame[dna_ch-1]
            dnaimg = cv2.normalize(dnaimg, None, 0, 255, norm_type=cv2.NORM_MINMAX)
            dnaimg = np.uint8(dnaimg)
            dnaimages.append(dnaimg)
            
            dnaimg = cv2.bitwise_and(dnaimg,dnaimg,mask = embryo)
            dnaimg = cv2.normalize(dnaimg, None, 0, 255, norm_type=cv2.NORM_MINMAX)
            dnaimg,binary,chromatids = process_dna(dnaimg)
            dnabinimages.append(cv2.merge([blank,blank,binary]))
            chromatids = [c for c in chromatids if embryo[c[0]][c[1]]==255]
            allchromatids.append(chromatids)
            
        no_chromatids = np.array([len(c) for c in allchromatids])
        
        #if no_chromatids.mean() > 10:
        #    continue
        
        #blue = np.zeros((width,height), np.uint8)

        kymo_width = 200
        allpoles = np.array(allpoles)
        df, fixed_poles, valid = create_dataframe(allpoles, allchromatids, pixel_res=metadata['pixel_res'], pixel_unit=metadata['pixel_unit'])
        pole_dist = [euclidian(p1=p1,p2=p2) for (p1,p2)in fixed_poles]
        left_pole = [int(0.5*(kymo_width-d)) for d in pole_dist]
        right_pole = [int(0.5*(kymo_width+d)) for d in pole_dist]
        
        images = [cv2.merge([blank, spimages[i], dnaimages[i]]) for i in range(len(spimages))]
        kymo, cropped_images = kymograph(images, fixed_poles, width=kymo_width, height=25, method='max')
        dnakymo, cropped_dna = kymograph(dnabinimages, fixed_poles, width=kymo_width, height=25, method='max')
        for i,line in enumerate(dnakymo):
            dnakymo[i,left_pole[i],1] = 255
            dnakymo[i,right_pole[i],1] = 255
        gaussian = 3
        dnakymo = cv2.GaussianBlur(dnakymo,(gaussian,gaussian),0)
        dnakymo = cv2.normalize(dnakymo, None, 0, 255.0, norm_type=cv2.NORM_MINMAX)
        ekymo = enhanced_kymograph(kymo, fixed_poles, padding=15)
        
        print (f"kymo.shape={kymo.shape}")
        print (f"ekymo.shape={ekymo.shape}")
        for i,frame_poles in enumerate(fixed_poles):
            for p in frame_poles:
                images[i] = cv2.circle(images[i], (int(p[0]),int(p[1])), 4, (255,0,255), 1)
        for i,frame_chromatids in enumerate(allchromatids):
            for c in frame_chromatids:
                images[i] = cv2.circle(images[i], (c[0],c[1]), 4, (255,255,0), 1)
        df['left Pole (pixel)'] = left_pole
        df['right Pole (pixel)'] = right_pole
        df['left DNA edge (pixel)'] = [np.where(line[:,2] > 127)[0][0] for line in dnakymo]
        df['right DNA edge (pixel)'] = [np.where(line[:,2] > 127)[0][-1] for line in dnakymo]
        df[f'left DNA-Pole dist ({metadata["pixel_unit"]})'] = (df['left DNA edge (pixel)']-df['left Pole (pixel)']) * metadata['pixel_res']
        df[f'right DNA-Pole dist ({metadata["pixel_unit"]})'] = (df['right Pole (pixel)']-df['right DNA edge (pixel)']) * metadata['pixel_res']
        df[f'left DNA-Midzone dist ({metadata["pixel_unit"]})'] = (df['left DNA edge (pixel)']-0.5*kymo.shape[1]) * metadata['pixel_res']
        df[f'right DNA-Midzone dist ({metadata["pixel_unit"]})'] = (df['right DNA edge (pixel)']-0.5*kymo.shape[1]) * metadata['pixel_res']
        df[f'left DNA velocity ({metadata["pixel_unit"]}/frame)'] = df[f'left DNA-Midzone dist ({metadata["pixel_unit"]})'].diff(periods=1).rolling(7).mean()
        df[f'right DNA velocity ({metadata["pixel_unit"]}/frame)'] = df[f'right DNA-Midzone dist ({metadata["pixel_unit"]})'].diff(periods=1).rolling(7).mean()
        #df[f'left DNA velocity ({metadata["pixel_unit"]} ROLL /frame)'] = df[f'left DNA velocity ({metadata["pixel_unit"]}/frame)'].rolling(7).mean()
        #df[f'right DNA velocity ({metadata["pixel_unit"]} ROLL /frame)'] = df[f'right DNA velocity ({metadata["pixel_unit"]}/frame)'].rolling(7).mean()
        
        print (f'Processed embryo {embryo_no}')
        if valid:
            datafile = os.path.splitext(fname)[0] + f'-embryo-{(embryo_no+1):04d}.csv'
            moviefile = os.path.splitext(fname)[0] + f'-embryo-{(embryo_no+1):04d}.mp4'
            cropped_moviefile = os.path.splitext(fname)[0] + f'-embryo-{(embryo_no+1):04d}-cropped.mp4'
            dna_moviefile = os.path.splitext(fname)[0] + f'-embryo-{(embryo_no+1):04d}-dna.mp4'
            kymofile = os.path.splitext(fname)[0] + f'-embryo-{(embryo_no+1):04d}-kymo.png'
            dnakymofile = os.path.splitext(fname)[0] + f'-embryo-{(embryo_no+1):04d}-dnakymo.png'
            ekymofile = os.path.splitext(fname)[0] + f'-embryo-{(embryo_no+1):04d}-ekymo.png'
            df.to_csv(datafile)
            save_movie(images, moviefile)
            save_movie(cropped_images, cropped_moviefile)
            save_movie(dnabinimages, dna_moviefile)
            #save_movie(rotated_images, os.path.splitext(fname)[0] + '-rot.mp4')
            cv2.imwrite(kymofile, kymo)    
            cv2.imwrite(ekymofile, ekymo)    
            cv2.imwrite(dnakymofile, dnakymo)    

            print (f'Saved embryo {embryo_no}')
        
@task
def proc_file(fname, spindle_ch, dna_ch, output):
    return f"Processed {fname}"
         
def main():
    """Main code block"""
    parser = init_parser()
    args = parser.parse_args()
    with Flow('Analysis') as flow:        
        files = get_files(args.input,fpattern=['*.nd2','*.tiff', '*.tif'])
        processed = process_file.map(files, spindle_ch=unmapped(args.spindle), dna_ch=unmapped(args.dna), output=unmapped(args.output))
        #print (f'Processing: {fname}, spindle channel:{spindle_ch}, dna channel:{dna_ch}'
        #opener = get_opener.map(files)
        #imagestack, metadata = opener.map(files)
        """
        print (f'\t{opener}')
        for k,v in metadata.items():
            print (f'\t{k}:{v}')
        if max(spindle_ch, dna_ch) > metadata['shape']['c']:
            print ("Skipping -- not enough channels.")
            return
        #pixel_microns = imagestack.metadata['pixel_microns']

        imagestack = register_stack(imagestack)

        width = metadata['shape']['x'] #imagestack.sizes['x']
        height = metadata['shape']['y'] #imagestack.sizes['y']
    
        #imagestack.bundle_axes = 'cxy'
        #imagestack.iter_axes = 't'
    
    
        #max_spindle_int = np.amax(np.array(imagestack)[:,1])
        #max_dna_int = np.amax(np.array(imagestack)[:,0])

        spindle_stack = np.array(imagestack)[:,spindle_ch-1]
        dna_stack = np.array(imagestack)[:,dna_ch-1]
        embryo_masks = find_embryos(spindle_stack)
    
        blank = np.zeros((height,width), np.uint8)

		"""
    #flow.visualize()
    flow.run()
	  


if __name__ == '__main__':
    main()