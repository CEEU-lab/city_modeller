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
                    
def land_use_agg_cat(x):
    btypes = {
        'residential-types': ['Casa', 'PH', 'Departamento'],
        'non-residential-types': ['Oficina', 'Depósito', 'Local comercial']
        }
    if x in btypes['residential-types']:
        return 'residential-types'
    elif x in btypes['non-residential-types']:
        return 'non-residential-types'
    else:
        return 'other'
    

def build_project_class(x, target_group, comparison_group):
    # TODO: documentation
    # last class name in alphabetical order is the target class 
    # groups = ['comparison_group','target_group']
    # target = groups[-1]
    btypes = {
        'target-group': target_group,
        'comparison-group': comparison_group
        }
    if x in btypes['target-group']:
        return 'target-group'
    elif x in btypes['comparison-group']:
        return 'comparison-group'
    else:
        return 'other'
    

    