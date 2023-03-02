import pandas as pd
import pandana as pnda
from datasources import get_bbox, get_pdna_network, get_census_data, get_public_space



# Mapa de distancia al EVP mas cercano por radio censal
tpois = get_public_space()

network.set_pois(category = 'tpois',
                 maxdist = 500,
                 maxitems = 1,
                 x_col = tpois.geometry.x, 
                 y_col = tpois.geometry.y)

results = network.nearest_pois(distance = 500,
                               category = 'tpois',
                               num_pois = 1,
                               include_poi_ids = True)

# Curva de población según minutos de caminata
# (...)

# Curva de poblacion segun area del espacio
# (...)