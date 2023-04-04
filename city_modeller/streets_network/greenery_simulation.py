import requests
import streamlit as st    
import time
from PIL import Image
import numpy as np
import io   
import pymeanshift as pms

def GSVpanoMetadataCollector(geom, api_key, allow_prints=False):
    '''
    This function is used to call the Google API url to collect the metadata of
    Google Street View Panoramas. 
    
    Parameters: 
        geom: shapely.geometry in EPSG4326
        
    '''
    
    lon = geom.y
    lat = geom.x     
    
    # get the meta data of panoramas 
    meta_base = 'https://maps.googleapis.com/maps/api/streetview/metadata?' #TODO: Maybe better inside config file?
                
    location = "{},{}".format(lon,lat) 
    # define the params for the metadata reques
    meta_params = {'key': api_key,
                   'location': location}
    
    # obtain the metadata of the request (this is free)
    meta_response = requests.get(meta_base, params=meta_params)
    
    data = meta_response.json()
    
    # in case there is not panorama in the site, therefore, continue
    if data['status']!='OK':
        if allow_prints:
            st.write("Reference Point not available")
    else:
        # get the meta data of the panorama
        panoDate = data['date']
        panoId = data['pano_id']
        panoLat = data['location']['lat']
        panoLon = data['location']['lng']
        
        if allow_prints:
            st.write('The coordinate (\{},{})\, panoId is: {}, panoDate is: {}'.format(panoLon,panoLat,panoId, panoDate))
        
        return panoDate, panoId, panoLat, panoLon
    
                    
def GreenViewComputing_3Horizon(headingArr,panoId,pitch,api_key,numGSVImg):
    
    """
    This function is used to download the GSV from the information provide
    by the gsv info txt, and save the result to a shapefile
    
    Required modules: StringIO, numpy, requests, and PIL
    
        greenmonth: a list of the green season, for example in Boston, greenmonth = ['05','06','07','08','09']
        
    last modified by Xiaojiang Li, MIT Senseable City Lab, March 25, 2018
    
    """
    # calculate the green view index
    greenPercent = 0.0
    images = []
    captions = []
    
    for heading in headingArr:
        #st.write("Heading is: {}".format(heading))
        
        # using different keys for different process, each key can only request 25,000 imgs every 24 hours
        URL = "https://maps.googleapis.com/maps/api/streetview?size=400x400&pano=%s&fov=80&heading=%d&pitch=%d&key=%s"%(panoId,heading,pitch,api_key)
        #st.write(URL)
        
        # let the code to pause by 1s, in order to not go over data limitation of Google quota
        time.sleep(1)
        #st.write(URL)
        #import pdb;pdb.set_trace()
        # classify the GSV images and calcuate the GVI
        try:
            response = requests.get(URL)
            image = Image.open(io.BytesIO(response.content))
            images.append(image)
            np_image = np.array(image)
            percent = VegetationClassification(np_image)
            captions.append(percent)
            greenPercent = greenPercent + percent
            

        # if the GSV images are not download successfully or failed to run, then return a null value
        except:
            greenPercent = -1000
            #percent = 0 #no greenery if error
            captions.append(0)
            #break
            continue

    # calculate the green view index by averaging six percents from six images
    greenViewVal = greenPercent/numGSVImg
    #st.write('The greenview: {}, pano: {}'.format(greenViewVal, panoId))
    return greenViewVal, images, captions
                 

def graythresh(array,level):
    '''array: is the numpy array waiting for processing
    return thresh: is the result got by OTSU algorithm
    if the threshold is less than level, then set the level as the threshold
    by Xiaojiang Li
    '''
    
    import numpy as np
    
    maxVal = np.max(array)
    minVal = np.min(array)
    
#   if the inputImage is a float of double dataset then we transform the data 
#   in to byte and range from [0 255]
    if maxVal <= 1:
        array = array*255
        # print "New max value is %s" %(np.max(array))
    elif maxVal >= 256:
        array = np.int((array - minVal)/(maxVal - minVal))
        # print "New min value is %s" %(np.min(array))
    
    # turn the negative to natural number
    negIdx = np.where(array < 0)
    array[negIdx] = 0
    
    # calculate the hist of 'array'
    dims = np.shape(array)
    hist = np.histogram(array,range(257))
    P_hist = hist[0]*1.0/np.sum(hist[0])
    
    omega = P_hist.cumsum()
    
    temp = np.arange(256)
    mu = P_hist*(temp+1)
    mu = mu.cumsum()
    
    n = len(mu)
    mu_t = mu[n-1]
    
    sigma_b_squared = (mu_t*omega - mu)**2/(omega*(1-omega))
    
    # try to found if all sigma_b squrered are NaN or Infinity
    indInf = np.where(sigma_b_squared == np.inf)
    
    CIN = 0
    if len(indInf[0])>0:
        CIN = len(indInf[0])
    
    maxval = np.max(sigma_b_squared)
    
    IsAllInf = CIN == 256
    if IsAllInf !=1:
        index = np.where(sigma_b_squared==maxval)
        idx = np.mean(index)
        threshold = (idx - 1)/255.0
    else:
        threshold = level
    
    if np.isnan(threshold):
        threshold = level
    
    return threshold



def VegetationClassification(Img):
    '''
    This function is used to classify the green vegetation from GSV image,
    This is based on object based and otsu automatically thresholding method
    The season of GSV images were also considered in this function
        Img: the numpy array image, eg. Img = np.array(Image.open(StringIO(response.content)))
        return the percentage of the green vegetation pixels in the GSV image
    
    By Xiaojiang Li
    '''
    
    # use the meanshift segmentation algorithm to segment the original GSV image
    (segmented_image, labels_image, number_regions) = pms.segment(Img,spatial_radius=6,
                                                     range_radius=7, min_density=40)
    
    I = segmented_image/255.0
    
    red = I[:,:,0]
    green = I[:,:,1]
    blue = I[:,:,2]
    
    # calculate the difference between green band with other two bands
    green_red_Diff = green - red
    green_blue_Diff = green - blue
    
    ExG = green_red_Diff + green_blue_Diff
    diffImg = green_red_Diff*green_blue_Diff
    
    redThreImgU = red < 0.6
    greenThreImgU = green < 0.9
    blueThreImgU = blue < 0.6
    
    shadowRedU = red < 0.3
    shadowGreenU = green < 0.3
    shadowBlueU = blue < 0.3
    del red, blue, green, I
    
    greenImg1 = redThreImgU * blueThreImgU*greenThreImgU
    greenImgShadow1 = shadowRedU*shadowGreenU*shadowBlueU
    del redThreImgU, greenThreImgU, blueThreImgU
    del shadowRedU, shadowGreenU, shadowBlueU
    
    greenImg3 = diffImg > 0.0
    greenImg4 = green_red_Diff > 0
    threshold = graythresh(ExG, 0.1)
    
    if threshold > 0.1:
        threshold = 0.1
    elif threshold < 0.05:
        threshold = 0.05
    
    greenImg2 = ExG > threshold
    greenImgShadow2 = ExG > 0.05
    greenImg = greenImg1*greenImg2 + greenImgShadow2*greenImgShadow1
    del ExG,green_blue_Diff,green_red_Diff
    del greenImgShadow1,greenImgShadow2
    
    # calculate the percentage of the green vegetation
    greenPxlNum = len(np.where(greenImg != 0)[0])
    greenPercent = greenPxlNum/(400.0*400)*100
    del greenImg1,greenImg2
    del greenImg3,greenImg4
    
    return greenPercent
