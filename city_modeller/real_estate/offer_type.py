from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects import conversion, default_converter

import geopandas as gpd
import rasterio
from rasterio.features import shapes


# TODO: docstring + how terra predicts over the grid (wiki)
predict_offer_class = """real_estate_offer <- function(offer_area, prediction_method, dir) {
            # 1. data type checks
            stopifnot("Offer area must be tabular" = is(offer_area, "data.frame"))
            stopifnot("Prediction must be linear, orthogonal or spline" = is.character(
                prediction_method)
                )
            stopifnot("The path name must be a string of char with destination output" = is.character(
                dir))
            
            # 2. Loads Point coords
            require("dplyr")
            ppp_data <- offer_area %>% mutate(tipo = as.factor(tipo_agr)) %>%
                dplyr::select(lat, lon, tipo)
            # last factor level in alphabetical order is the target class
            target_label <- tail(levels(ppp_data$tipo),1)
            print(paste("Target class: ", target_label))

            # 3. logistic adjustment (intensity func)
            require(splines)
            if (prediction_method == "linear") 
            {
                #linear
                logistic_adj <- glm(tipo ~ lon + lat, data = ppp_data, family = "binomial")
            } 
            else if (prediction_method == "orthogonal") 
            {
                # Orthogonal Polynomials
                logistic_adj <- glm(tipo ~ poly(lon, 3) * poly(lat, 3),
                data = ppp_data, family = "binomial")
            } 
            else if (prediction_method == "splines") 
            {
                # Polynomial spline
                logistic_adj <- glm(tipo ~ bs(lon, 4) * bs(lat, 4),
                data = ppp_data, family = "binomial")
            }

            # 4. grid predictions
            cant <- 100
            grid_canvas <- expand.grid(
                lon = seq(
                min(ppp_data$lon),
                max(ppp_data$lon),
                length.out = cant),
                lat = seq(min(ppp_data$lat),
                max(ppp_data$lat),
                length.out = cant))
            
            pred <- predict(logistic_adj, newdata = grid_canvas, type = "response")
            summary(pred)

            # Empty raster
            require(raster)
            rowxcol <- cant
            raster_canvas <- raster(
                nrows = rowxcol, ncols = rowxcol,
                xmn = min(ppp_data$lon), xmx = max(ppp_data$lon),
                ymn = min(ppp_data$lat), ymx = max(ppp_data$lat)
            )
            # Rasterize prediction
            raster_pred <- raster::rasterize(grid_canvas,
            raster_canvas, field = pred, fun = median)
            
            writeRaster(raster_pred,dir,format="GTiff", overwrite=TRUE) }"""


def offer_type_predictor_wrapper(df, geom, path) -> None:
    """
    Python wrapper to run the R function predict_offer_class.

    Parameters:
    -----------
    df : pd.DataFrame
        Real Estate Offer with adjusted classes for prediction
    geom: 
    path : str
        Source route of the predicted output (html widget)

    Returns
    -------
    gpd_polygonized_raster : gpd.GeoDataFrame
        grid prediction in vector format
    """    
    with conversion.localconverter(default_converter):
        # loads pandas as data.frame r object
        with (ro.default_converter + pandas2ri.converter).context():
            r_from_pd_df = ro.conversion.get_conversion().py2rpy(df)

        # parameters
        # TODO: use histogram to chek the observed distribution of the target class?
        prediction_method = "splines"  # DENSITY FUNCTION.

        # predict offer type
        ro.r(predict_offer_class)
        predominant_offer = ro.globalenv["real_estate_offer"]
        # exports tif result
        predominant_offer(r_from_pd_df, prediction_method, path)

    mask = None
    with rasterio.Env():
        with rasterio.open(path) as src:
            image = src.read(1) # first band
            results = (
                {'properties': {'raster_val': v}, 'geometry': s}
                for i, (s, v) in enumerate(
                    shapes(image, mask=mask, transform=src.transform))
            )
            
    src.close()
    gpd_polygonized_raster = gpd.GeoDataFrame.from_features(list(results)) 
    gpd_polygonized_raster = gpd_polygonized_raster.set_crs("epsg:4326")
    market_area_mask = geom.to_crs(4326).unary_union.convex_hull   
    gpd_polygonized_raster = gpd_polygonized_raster.clip(market_area_mask)
    gpd_polygonized_raster.raster_val = round(gpd_polygonized_raster.raster_val, 2)
    return gpd_polygonized_raster
    