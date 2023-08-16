from rpy2 import robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects import conversion, default_converter


# TODO: docstring + how terra predicts over the grid (wiki)
predict_offer_class = '''real_estate_offer <- function(offer_area, prediction_method, intervals, colorsvec, dir) {
            # 1. data type checks
            stopifnot("Offer area must be tabular" = is(offer_area, "data.frame"))
            stopifnot("Prediction must be linear, orthogonal or spline" = is.character(
                prediction_method)
                )
            stopifnot("Specify the max number of intervals" = is.numeric(intervals))
            stopifnot("Colors must be a vector of char" = is.vector(colorsvec))
            stopifnot("The path name must be a string of char with destination output" = is.character(
                dir))
            
            # 2. Loads Point geoms
            require("sf")
            require("dplyr")
            # Convert csv to simple feat object
            ppp_caba <- offer_area %>%
            st_as_sf(coords = c("lon", "lat"), crs = 4326)
            ppp_coords <- st_coordinates(ppp_caba)
            ppp_data <- ppp_caba %>% mutate(lon = ppp_coords[, "X"],
                                            lat = ppp_coords[, "Y"],
                                            tipo = as.factor(tipo_agr)) %>%
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
            # require("terra")
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
            sf_collection <- st_union(ppp_data)
            study_area <- st_convex_hull(sf_collection)
            raster_mask <- raster::mask(raster_pred, as_Spatial(study_area))
            raster_vals <- raster::values(raster_mask)

            # 5. Plot prediction results
            require(tmap)
            require(leaflet)

            cat_breaks <- seq(min(raster_vals, na.rm = TRUE),
            max(raster_vals, na.rm = TRUE), length.out = intervals)

            col_bins <- colorBin(
                palette = colorsvec,
                bins = cat_breaks,
                domain = cat_breaks,
                na.color = "#00000000")
                    
            # export result
            outmap <- leaflet(width = 600, height = 400,
                options = leafletOptions(zoomControl = TRUE,
                                    minZoom = 10, maxZoom = 15)) %>%
                addTiles() %>%
                addProviderTiles(providers$CartoDB.Positron) %>%
                addRasterImage(raster_mask, opacity = 0.75, colors = colorsvec) %>%
                addLegend(pal = col_bins, values = cat_breaks)
            
            require(htmlwidgets)
            saveWidget(widget = outmap, file = dir)
            }'''


def offer_type_predictor_wrapper(df, path) -> None:
    """
    Python wrapper to run the R function predict_offer_class.

    Parameters:
    -----------
    df : pd.DataFrame
        Real Estate Offer with adjusted classes for prediction
    path : str
        Source route of the predicted output (html widget)

    Returns
    -------
    None
        Writes leaflet html widget 
    """
        
    with conversion.localconverter(default_converter):
        # loads pandas as data.frame r object
        with (ro.default_converter + pandas2ri.converter).context():
            r_from_pd_df = ro.conversion.get_conversion().py2rpy(df)
        
        # parameters
        # TODO: use histogram to chek the observed distribution of the target class?
        prediction_method = "splines" # DENSITY FUNCTION.  
        intervals = 10 
        colorsvec = ro.StrVector(['lightblue', 'yellow', 'purple'])

        # predict offer type
        ro.r(predict_offer_class)
        predominant_offer = ro.globalenv['real_estate_offer']
        # exports html result
        predominant_offer(r_from_pd_df, prediction_method, intervals, colorsvec, path)