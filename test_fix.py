#!/usr/bin/env python3

import sys
sys.path.append('.')
from scenario02 import GNPY_AVAILABLE, calculate_scenario02_network

print('GNPY Available:', GNPY_AVAILABLE)

# Test with minimal data
test_params = {
    'topology_data': {
        'elements': [
            {
                'uid': 'Site_A', 
                'type': 'Transceiver', 
                'type_variety': 'Transceiver', 
                'latitude': 0.0, 
                'longitude': 0.0, 
                'location': 'Site_A', 
                'city': '', 
                'region': '', 
                'country': '', 
                'parameters': {
                    'P_tot_dbm_input': {'value': 1.0}, 
                    'tx_osnr': {'value': 40.0}
                }
            },
            {
                'uid': 'Site_B', 
                'type': 'Transceiver', 
                'type_variety': 'Transceiver', 
                'latitude': 0.0, 
                'longitude': 0.0, 
                'location': 'Site_B', 
                'city': '', 
                'region': '', 
                'country': '', 
                'parameters': {
                    'sens': {'value': -25.0}
                }
            }
        ],
        'connections': [
            {'from_node': 'Site_A', 'to_node': 'Site_B'}
        ]
    }
}

try:
    result = calculate_scenario02_network(test_params)
    print('Test result success:', result.get('success', False))
    if not result.get('success', False):
        print('Error:', result.get('error'))
    else:
        print('Function executed successfully!')
except Exception as e:
    print('Exception:', str(e))
    import traceback
    traceback.print_exc() 