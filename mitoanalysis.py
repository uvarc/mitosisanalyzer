#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 21:42:03 2021

@author: khs3z
"""

import os
import argparse
import glob
from multiprocessing import Pool
from functools import partial
import cv2
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import label
from skimage.measure import profile_line
from nd2reader import ND2Reader
import pandas as pd

RES_UNIT_DICT = {1:'<unknown>', 2:'inch', 3:'cm'}

def init_parser():
    parser = argparse.ArgumentParser(
        description='Analyzes spindle pole and chromatid movements in .nd2 timelapse files')
    parser.add_argument('-i', '--input', required=True, help='.nd2 file or directory with .nd2 files to be processed')
    parser.add_argument('-o', '--output', default=None, help='output file or directory')
    parser.add_argument('-p', '--processes', default=1, type=int, help='number or parallel processes')
    parser.add_argument('-s', '--spindle', default=2, type=int, help='channel # for tracking spindle poles')
    parser.add_argument('-d', '--dna', default=1, type=int, help='channel # for tracking dna')
    return parser
    
def get_files(path, fpattern='*.tif'):
    files = []
    for pattern in fpattern:
        files.extend(glob.glob(os.path.join(path,pattern)))
    # remove possible duplicates and sort
    files = list(set(files))
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

def register_stack(imagestack):
    return imagestack

def watershed(a, img, erode=5, relthr=0.7):
    border = cv2.dilate(img, None, iterations=5)
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
           poles.append([0,0])

    #image = draw_lines(image,edges,color=(255,0,0),thickness=1)
    #print (f'\tspindle poles: {poles}')
    return image,binary,poles #spindle_poles

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
    
def create_dataframe(allpoles, allchromatids, pixel_res=1.0, pixel_unit='um'):
    polearray = np.array(allpoles).reshape(len(allpoles), 4)
    df = pd.DataFrame(polearray, columns=['Pole 1,x (pixel)', 'Pole 1,y (pixel)', 'Pole 2,x (pixel)', 'Pole 2,y (pixel)'])
    df[f'Pole-Pole Distance [{pixel_unit}]'] = [pixel_res * euclidian(p1=allpoles[i][0],p2=allpoles[i][1]) for i in range(len(allpoles))]
    df['Pole 1 [pixel]'] = '(' + df['Pole 1,x (pixel)'].astype(str) + '/'+ df['Pole 1,y (pixel)'].astype(str) +')'
    df['Pole 2 [pixel]'] = '(' + df['Pole 2,x (pixel)'].astype(str) + '/'+ df['Pole 2,y (pixel)'].astype(str) +')'
    df['Frame'] = np.arange(1, len(allpoles)+1)
    df = df[['Frame', 'Pole 1 [pixel]', 'Pole 2 [pixel]', f'Pole-Pole Distance [{pixel_unit}]']]
    df = df.set_index('Frame')
    mean = df[f'Pole-Pole Distance [{pixel_unit}]'].mean()
    median = df[f'Pole-Pole Distance [{pixel_unit}]'].median()
    std = df[f'Pole-Pole Distance [{pixel_unit}]'].std()
    print (f'mean={mean}, median={median}, std={std}')
    valid = median != 0.0 and mean > 3. and mean/median > 0.8 and mean/median < 1.2 and std < 0.4*mean
    return df, True #valid

def nd2_opener(fname):
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

def tif_opener(fname):
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

def skip_opener(fname):
    return None, None
    
def get_opener(fname):
    ext = os.path.splitext(fname)[1]
    if ext == '.nd2':
        return nd2_opener
    elif ext in ['.tif', '.tiff']:
        return tif_opener
    return skip_opener

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
        allchromatids = []
        spimages = []
        dnaimages = []
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
            spimg,binary,spindle_poles = process_spindle(spimg)
            #if len(spindle_poles) == 2:
            #    end1,end2 = profile_endpoints(spindle_poles[0], spindle_poles[1], center(spindle_poles[0], spindle_poles[1]), 100)
            #    profile = profile_line(spimg, end1, end2)
            #    print (f'profile={profile}')
            allpoles.append(spindle_poles)  #(end1,end2)
            
        for frame_no, dnaimg in enumerate(dna_stack):
            #dnaimg = frame[dna_ch-1]
            dnaimg = cv2.normalize(dnaimg, None, 0, 255, norm_type=cv2.NORM_MINMAX)
            dnaimg = np.uint8(dnaimg)
            dnaimages.append(dnaimg)
            
            dnaimg = cv2.bitwise_and(dnaimg,dnaimg,mask = embryo)
            dnaimg = cv2.normalize(dnaimg, None, 0, 255, norm_type=cv2.NORM_MINMAX)
            dnaimg,binary,chromatids = process_dna(dnaimg)
            allchromatids.append(chromatids)
            
        no_chromatids = np.array([len(c) for c in allchromatids])
        
        #if no_chromatids.mean() > 10:
        #    continue
        
        #blue = np.zeros((width,height), np.uint8)
        images = [cv2.merge([blank, spimages[i], dnaimages[i]]) for i in range(len(spimages))]
        for i,frame_poles in enumerate(allpoles):
            for p in frame_poles:
                images[i] = cv2.circle(images[i], (p[0],p[1]), 4, (255,0,255), 1)
        for i,frame_chromatids in enumerate(allchromatids):
            for c in frame_chromatids:
                images[i] = cv2.circle(images[i], (c[0],c[1]), 4, (255,255,0), 1)
        #plt.subplot(121)
        #plt.imshow(spimg, cmap='gray')
        #plt.subplot(122)
        #plt.imshow(binary, cmap='gray')
        allpoles = np.array(allpoles)
        
        #sp_centers = [center(p[0], p[1]) for p in allpoles] 
        
        #pole1_distance = [euclidian(p1=allpoles[i][0], p2=sp_centers[i]) for i in range(len(allpoles))]
        #pole2_distance = [-euclidian(p1=allpoles[i][1], p2=sp_centers[i]) for i in range(len(allpoles))]
        #plt.plot(pole1_distance, color='green')
        #plt.plot(pole2_distance, color='magenta')
        #plt.show()
    
        print (f'Processed embryo {embryo_no}')
        df, valid = create_dataframe(allpoles, allchromatids, pixel_res=metadata['pixel_res'], pixel_unit=metadata['pixel_unit']) 
        if valid:
            datafile = os.path.splitext(fname)[0] + f'-embryo-{(embryo_no+1):04d}.csv'
            df.to_csv(datafile)

            moviefile = os.path.splitext(fname)[0] + f'-embryo-{(embryo_no+1):04d}.mp4'
            save_movie(images, moviefile)    
            print (f'Saved embryo {embryo_no}')
        
        """
        from PIL import Image, ImageSequence
        im = Image.open(fname)
        print (dir(im))
        print (im.ifd)
        dtype = {'F': np.float32, 'L': np.uint8}[im.mode]
        np_img = np.array(im.getdata(), dtype=dtype)
        for i, page in enumerate(ImageSequence.Iterator(im)):
            print (f'page {i}, {type(page)}')
        """
  
         
def main():
    parser = init_parser()
    args = parser.parse_args()
    if os.path.isfile(args.input):
        files = [args.input]
    elif os.path.isdir(args.input):    
        files = get_files(args.input,fpattern=['*.nd2','*.tiff', '*.tif'])
    else:
        print (f'File/directory {args.input} does not exists or has an invalid format. Only .nd or .tif files can be processed.')
        return
    with Pool(processes=args.processes) as pool:
        pool.map(partial(process_file, spindle_ch=args.spindle, dna_ch=args.dna, output=args.output), files)
    print ('Done.')
    #plt.show()
  


if __name__ == '__main__':
    main()