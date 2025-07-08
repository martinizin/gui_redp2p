#!/usr/bin/env python3
"""
Verification script to reproduce the OSNR_bw results that match the notebook
Run this script to verify the scenario02 calculations work correctly.
"""

import json
from scenario02 import calculate_scenario02_network, enhance_elements_with_parameters

def main():
    print("🔬 OSNR_bw Verification Test")
    print("="*60)
    print("This script uses the exact same parameters as in the notebook")
    print("to verify that scenario02.py produces matching results.")
    print()
    
    # Test parameters (matching the notebook)
    test_params = {
        'topology_file': 'data/topologiaEdfa1.json',
        'edfa_nf': 6.0,           # Noise Figure for all EDFAs
        'tx_osnr': 40.0,          # Transmitter OSNR
        'total_power': 1.0,       # Total transmitter power (dBm)
        'sensitivity': 0.0        # Receiver sensitivity (dBm)
    }
    
    print("📋 Test Parameters:")
    print(f"   • Topology file: {test_params['topology_file']}")
    print(f"   • EDFA Noise Figure: {test_params['edfa_nf']} dB")
    print(f"   • Transmitter OSNR: {test_params['tx_osnr']} dB")
    print(f"   • Total Power: {test_params['total_power']} dBm")
    print(f"   • Receiver Sensitivity: {test_params['sensitivity']} dBm")
    print()
    
    try:
        # Load topology
        with open(test_params['topology_file'], 'r') as f:
            topology_data = json.load(f)
        
        # Enhance elements with parameters
        enhanced_elements = enhance_elements_with_parameters(topology_data['elements'])
        
        # Apply test parameters
        for element in enhanced_elements:
            if element.get('type') == 'Edfa':
                # Set NF for all EDFAs
                if 'parameters' not in element:
                    element['parameters'] = {}
                element['parameters']['nf0'] = {'value': test_params['edfa_nf']}
                
            elif element.get('type') == 'Transceiver':
                if 'parameters' not in element:
                    element['parameters'] = {}
                    
                if element.get('uid') == 'Site_A':
                    # Transmitter parameters
                    element['parameters']['tx_osnr'] = {'value': test_params['tx_osnr']}
                    element['parameters']['P_tot_dbm_input'] = {'value': test_params['total_power']}
                    
                elif element.get('uid') == 'Site_B':
                    # Receiver parameters
                    element['parameters']['sens'] = {'value': test_params['sensitivity']}
        
        # Prepare calculation
        enhanced_topology = {
            'elements': enhanced_elements,
            'connections': topology_data['connections']
        }
        
        calc_params = {
            'topology_data': enhanced_topology
        }
        
        print("🚀 Running calculation...")
        results = calculate_scenario02_network(calc_params)
        
        if results.get('success', False):
            print("✅ Calculation completed successfully!")
            print()
            
            # Display results table
            print("📊 OSNR_bw Results (should match notebook):")
            print("-" * 60)
            print(f"{'Stage':<12} {'Distance':<10} {'Power':<10} {'OSNR_bw':<10}")
            print("-" * 60)
            
            # Expected values from notebook
            expected_osnr = {
                'Site_A': 40.00,
                'Edfa1': 30.50,
                'Span1': 30.50,
                'Edfa2': 25.41,
                'Span2': 25.41,
                'Edfa3': 13.00,
                'Site_B': 13.00
            }
            
            for stage in results['stages']:
                stage_name = stage['name']
                calculated_osnr = float(stage['osnr_bw']) if stage['osnr_bw'] != '∞' else float('inf')
                expected_val = expected_osnr.get(stage_name, 'N/A')
                
                # Check if values match (within tolerance)
                if expected_val != 'N/A' and not np.isinf(calculated_osnr):
                    diff = abs(calculated_osnr - expected_val)
                    status = "✅" if diff < 0.1 else "❌"
                else:
                    status = "❓"
                
                print(f"{stage_name:<12} {stage['distance']:<10.1f} {stage['power_dbm']:<10.2f} {stage['osnr_bw']:<10} {status}")
            
            print("-" * 60)
            
            # Summary
            print("\n📋 Expected vs Calculated OSNR_bw:")
            print(f"{'Stage':<12} {'Expected':<10} {'Calculated':<12} {'Diff':<8} {'Status'}")
            print("-" * 55)
            
            all_match = True
            for stage in results['stages']:
                stage_name = stage['name']
                calculated_osnr = float(stage['osnr_bw']) if stage['osnr_bw'] != '∞' else float('inf')
                expected_val = expected_osnr.get(stage_name, 'N/A')
                
                if expected_val != 'N/A' and not np.isinf(calculated_osnr):
                    diff = abs(calculated_osnr - expected_val)
                    status = "✅ MATCH" if diff < 0.1 else "❌ DIFF"
                    if diff >= 0.1:
                        all_match = False
                    print(f"{stage_name:<12} {expected_val:<10.2f} {calculated_osnr:<12.2f} {diff:<8.2f} {status}")
                else:
                    print(f"{stage_name:<12} {expected_val:<10} {stage['osnr_bw']:<12} {'N/A':<8} {'❓ N/A'}")
            
            print("\n" + "="*60)
            if all_match:
                print("🎉 SUCCESS: All OSNR_bw values match the notebook results!")
            else:
                print("⚠️  WARNING: Some OSNR_bw values differ from notebook results")
            
            print("\n📈 Additional Results:")
            final = results['final_results']
            print(f"   • Final power: {final['final_power_dbm']:.2f} dBm")
            print(f"   • Link successful: {final['link_successful']}")
            print(f"   • Number of channels: {final['nch']}")
            print(f"   • Power per channel: {final['tx_power_per_channel_dbm']:.2f} dBm")
            
        else:
            print("❌ Calculation failed!")
            print(f"Error: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"❌ Exception occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import numpy as np
    main() 