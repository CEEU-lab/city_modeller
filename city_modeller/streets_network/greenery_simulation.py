import numpy as np
import requests
import streamlit as st
import time
from PIL import Image
import io
import pymeanshift as pms


MIN_THRESHOLD, MAX_THRESHOLD = 0.05, 0.1
IMAGE_WIDTH, IMAGE_HEIGHT = 400, 400


def GSVpanoMetadataCollector(geom, api_key, allow_prints=False):
    """
    Calls the Google API url to collect the metadata of
    Google Street View Panoramas.

    Parameters:
    -----------
    geom : shapely.geometry in EPSG4326
        Resultant x/y coords after interpolation process
    api_key : str
        GSV client key
    allow_prints : bool
        allow informative print

    Returns
    -------
    panoDate : str
        Date of the panoramas
    panoId : str
        Pano idx associated to the interpolated Point
    panoLat : str
        Lat coord of the Pano idx associated to the interpolated Point
    panoLon : str
        Lon coord of the Pano idx associated to the interpolated Point
    """
    lon = geom.y
    lat = geom.x

    # get the meta data of panoramas
    # TODO: Maybe better inside config file?
    meta_base = "https://maps.googleapis.com/maps/api/streetview/metadata?"

    location = "{},{}".format(lon, lat)
    # define the params for the metadata reques
    meta_params = {"key": api_key, "location": location}

    # obtain the metadata of the request (this is free)
    meta_response = requests.get(meta_base, params=meta_params)

    data = meta_response.json()

    # in case there is not panorama in the site, therefore, continue
    if data["status"] != "OK":
        if allow_prints:  # TODO: Make widget with red text.
            st.write("Reference Point not available")
    else:
        # get the meta data of the panorama
        panoDate = data["date"]
        panoId = data["pano_id"]
        panoLat = data["location"]["lat"]
        panoLon = data["location"]["lng"]

        if allow_prints:  # TODO: Make widget with red text.
            st.write(
                "The coordinate ({},{}), panoId is: {}, panoDate is: {}".format(
                    panoLon, panoLat, panoId, panoDate
                )
            )

        return panoDate, panoId, panoLat, panoLon


def GreenViewComputing_3Horizon(headingArr, panoId, pitch, api_key, numGSVImg):

    """
    Calls the endpoint associated to the collected Panoramas and calculates
    Green View Index by calculating the green pixels average between all the
    images tied to each panorama.

    Parameters:
    -----------
    headingArr : np.array
        360° covering angles
    panoId : str
        panorama index
    pitch : str
        vertical covering angles. If implemented must be iterated like heading
    api_key
        GSV client key
    numGSVImg: int
        number of images associated to each Panorama. This depends
        on number of heading angles

    Returns
    -------
    greenViewVal : float
        Green View calculated for a given panorama
    images : list
        list of images associated to the panorama
    captions : list
        list of captions to describe each image
    """
    # calculate the green view index
    greenPercent = 0.0
    images = []
    captions = []

    for heading in headingArr:
        # TODO: moves endpoint to config file? Use request params.
        # each key can only request 25,000 imgs every 24 hours
        URL = (
            "https://maps.googleapis.com/maps/api/streetview?"
            + f"size={IMAGE_WIDTH}x{IMAGE_HEIGHT}&pano={panoId}&fov=80&"
            + f"heading={heading}&pitch={pitch}&key={api_key}"
        )

        # let the code to pause by 1s, in order to not go over data limitation of
        # Google quota
        time.sleep(1)  # FIXME: Try to reduce this to make faster.

        # classify the GSV images and calcuate the GVI
        try:
            response = requests.get(URL)
            image = Image.open(io.BytesIO(response.content))
            images.append(image)
            np_image = np.array(image)
            percent = VegetationClassification(np_image)
            captions.append(percent)
            greenPercent = greenPercent + percent

        # if the GSV images are not download successfully or failed to run, then return
        # a null value
        except:  # FIXME
            greenPercent = -1000
            captions.append(0)
            continue

    # calculate the green view index by averaging 3 percents from 3 images
    greenViewVal = greenPercent / numGSVImg
    # st.write('The greenview: {}, pano: {}'.format(greenViewVal, panoId))
    return greenViewVal, images, captions


def graythresh(array, level):
    """
    Applies OTSU thresholding method for greenery segmentation
    Parameters
    ----------
    array : np.array
        green to red + blue array of differences to use for thresholding
        pixels classification
    level : float
        if the threshold is less than level, then set the level as the threshold

    Returns
    -------
    threshold : float
        is the result got by OTSU algorithm

    by Xiaojiang Li
    """
    maxVal = np.max(array)
    minVal = np.min(array)

    #   if the inputImage is a float of double dataset then we transform the data
    #   in to byte and range from [0 255]
    if maxVal <= 1:
        array = array * 255  # NOTE: Might need an int.
        # print("New max value is %s" %(np.max(array)))
    elif maxVal >= 256:
        # FIXME: This is a MinMaxScaler, and is 0-1, and turned to int.
        array = np.int((array - minVal) / (maxVal - minVal))  # type: ignore
        # print("New min value is %s" %(np.min(array)))

    # turn the negative to natural number
    array = np.maximum(array, 0)

    # calculate the hist of 'array'
    hist = np.histogram(array, range(257))
    P_hist = hist[0] * 1.0 / np.sum(hist[0])

    omega = P_hist.cumsum()

    temp = np.arange(256)
    mu = P_hist * (temp + 1)
    mu = mu.cumsum()

    n = len(mu)
    mu_t = mu[n - 1]

    sigma_b_squared = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))

    # try to found if all sigma_b squrered are NaN or Infinity
    CIN = np.sum(sigma_b_squared == np.inf)

    maxval = np.max(sigma_b_squared)

    IsAllInf = CIN == 256
    if not IsAllInf:
        index = np.where(sigma_b_squared == maxval)
        idx = np.mean(index)
        threshold = (idx - 1) / 255.0
    else:
        threshold = level

    if np.isnan(threshold):
        threshold = level

    return threshold


def VegetationClassification(Img):
    """
    Classifies the green vegetation from GSV images.
    Parameters
    ----------
    Img: the numpy array image, eg.
        Img = np.array(Image.open(io.BytesIO(response.content)))

    Returns
    -------
    greenPercent : float

    By Xiaojiang Li
    """

    # TODO: Explore alternatives for img segmentation
    # use the meanshift segmentation algorithm to segment the original GSV image
    (segmented_image, labels_image, number_regions) = pms.segment(
        Img, spatial_radius=6, range_radius=7, min_density=40
    )

    I = segmented_image / 255.0

    red = I[:, :, 0]
    green = I[:, :, 1]
    blue = I[:, :, 2]

    # calculate the difference between green band with other two bands
    green_red_Diff = green - red
    green_blue_Diff = green - blue

    ExG = green_red_Diff + green_blue_Diff
    diffImg = green_red_Diff * green_blue_Diff

    redThreImgU = red < 0.6
    greenThreImgU = green < 0.9
    blueThreImgU = blue < 0.6

    shadowRedU = red < 0.3
    shadowGreenU = green < 0.3
    shadowBlueU = blue < 0.3
    del red, blue, green, I

    greenImg1 = redThreImgU * blueThreImgU * greenThreImgU
    greenImgShadow1 = shadowRedU * shadowGreenU * shadowBlueU
    del redThreImgU, greenThreImgU, blueThreImgU
    del shadowRedU, shadowGreenU, shadowBlueU

    greenImg3 = diffImg > 0.0
    greenImg4 = green_red_Diff > 0
    threshold = graythresh(ExG, 0.1)

    threshold = np.clip(threshold, MIN_THRESHOLD, MAX_THRESHOLD)

    greenImg2 = ExG > threshold
    greenImgShadow2 = ExG > 0.05
    greenImg = greenImg1 * greenImg2 + greenImgShadow2 * greenImgShadow1
    del ExG, green_blue_Diff, green_red_Diff
    del greenImgShadow1, greenImgShadow2

    # calculate the percentage of the green vegetation
    greenPxlNum = (greenImg != 0).sum()
    greenPercent = greenPxlNum / (IMAGE_WIDTH * IMAGE_HEIGHT) * 100
    del greenImg1, greenImg2
    del greenImg3, greenImg4

    return greenPercent
