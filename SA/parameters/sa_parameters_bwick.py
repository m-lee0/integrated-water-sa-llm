def get_sa_parameters():
    params = {
        '3607-land::surface_coefficient': {
            'minimum': 0,
            'maximum': 1,
            'default':0.05,
            'category': ''
        },
        '3607-land::percolation_coefficient': {
            'minimum': 0,
            'maximum': 1,
            'default':0.25,
            'category': ''
        }
    }
    return params

def params_to_select():
    return [
        '3607-land::surface_coefficient',
        '3607-land::percolation_coefficient',
    ]
