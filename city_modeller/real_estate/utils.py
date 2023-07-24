def density_agg_cat(x):
    btypes = {
        'high-density-types': ['Departamento'],
        'low-density-types': ['Casa', 'PH']
        }
    if x in btypes['high-density-types']:
        return 'high-density-types'
    elif x in btypes['low-density-types']:
        return 'low-density-types'
    else:
        return 'other'
                    
def residential_agg_cat(x):
    btypes = {
        'residential-types': ['Casa', 'PH', 'Departamento'],
        'non-residential-types':['Oficina', 'Depósito', 'Local' 'comercial']
        }
    if x in btypes['residential-types']:
        return 'residential-types'
    elif x in btypes['non-residential-types']:
        return 'non-residential-types'
    else:
        return 'other'