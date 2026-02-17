#!/usr/bin/env python3
"""
Validación de extracción de datos de los logs de ciclos
"""
import re
import json

def extract_results_from_log(log_file):
    """Extrae resultados de un archivo de log"""
    results = {}
    
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    # Buscar bloques de resultados línea por línea
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Buscar línea de "Results for X.XM nodes:"
        match_header = re.search(r'📊 Results for ([0-9.]+)M nodes:', line)
        if match_header:
            size_m = float(match_header.group(1))
            size_nodes = int(size_m * 1_000_000)
            
            # Buscar las siguientes 3-4 líneas para standalone, burst, speedup
            standalone = None
            burst = None
            speedup = None
            
            for j in range(i+1, min(i+10, len(lines))):
                if 'Standalone:' in lines[j]:
                    match_s = re.search(r'Standalone:\s+([0-9.]+)\s+ms', lines[j])
                    if match_s:
                        standalone = float(match_s.group(1))
                
                if 'Burst:' in lines[j]:
                    match_b = re.search(r'Burst:\s+([0-9.]+)\s+ms', lines[j])
                    if match_b:
                        burst = float(match_b.group(1))
                
                if 'Speedup:' in lines[j]:
                    match_sp = re.search(r'Speedup:\s+([0-9.]+)x', lines[j])
                    if match_sp:
                        speedup = float(match_sp.group(1))
                
                # Si encontramos los 3, guardamos y salimos
                if standalone and burst and speedup:
                    results[size_nodes] = {
                        'standalone_ms': standalone,
                        'burst_ms': burst,
                        'speedup': speedup
                    }
                    break
        
        i += 1
    
    return results

def main():
    print("="*80)
    print("VALIDACIÓN DE DATOS EXTRAÍDOS DE LOGS")
    print("="*80)
    
    # Datos que puse en el análisis
    my_data = {
        '3.0M': {
            'ciclo1': {'standalone': None, 'burst': None, 'speedup': None},
            'ciclo2': {'standalone': 13024, 'burst': 4347, 'speedup': 3.00},
            'ciclo3': {'standalone': 13250, 'burst': 6461, 'speedup': 2.05}
        },
        '4.0M': {
            'ciclo1': {'standalone': 18460, 'burst': 5952, 'speedup': 3.10},
            'ciclo2': {'standalone': 19341, 'burst': 7655, 'speedup': 2.53},
            'ciclo3': {'standalone': 19079, 'burst': 7255, 'speedup': 2.63}
        },
        '4.5M': {
            'ciclo1': {'standalone': 20619, 'burst': 6521, 'speedup': 3.16},
            'ciclo2': {'standalone': 22599, 'burst': 6702, 'speedup': 3.37},
            'ciclo3': {'standalone': 20891, 'burst': 7684, 'speedup': 2.72}
        },
        '5.0M': {
            'ciclo1': {'standalone': 22809, 'burst': 6960, 'speedup': 3.28},
            'ciclo2': {'standalone': 23233, 'burst': 7845, 'speedup': 2.96},
            'ciclo3': {'standalone': None, 'burst': None, 'speedup': None}
        },
        '6.0M': {
            'ciclo1': {'standalone': 28128, 'burst': 7671, 'speedup': 3.67},
            'ciclo2': {'standalone': 28470, 'burst': 8757, 'speedup': 3.25},
            'ciclo3': {'standalone': None, 'burst': None, 'speedup': None}
        }
    }
    
    # Extraer de los logs
    print("\nExtrayendo datos de logs...")
    log_results = {}
    for cycle in [1, 2, 3]:
        log_file = f"validation_cycle_{cycle}.log"
        try:
            log_results[f'ciclo{cycle}'] = extract_results_from_log(log_file)
            print(f"✅ Ciclo {cycle}: {len(log_results[f'ciclo{cycle}'])} puntos extraídos")
        except FileNotFoundError:
            print(f"⚠️  Ciclo {cycle}: Archivo no encontrado")
        except Exception as e:
            print(f"❌ Ciclo {cycle}: Error - {e}")
    
    # Comparar
    print("\n" + "="*80)
    print("COMPARACIÓN: DATOS MANUALES vs LOGS")
    print("="*80)
    
    size_map = {
        3000000: '3.0M',
        4000000: '4.0M',
        4500000: '4.5M',
        5000000: '5.0M',
        6000000: '6.0M'
    }
    
    all_match = True
    
    for size_nodes, size_label in size_map.items():
        print(f"\n{size_label} ({size_nodes:,} nodes):")
        print(f"{'Ciclo':>8} {'Fuente':>10} {'Standalone':>12} {'Burst':>12} {'Speedup':>10} {'Match':>8}")
        print("-" * 70)
        
        for cycle_num in [1, 2, 3]:
            cycle_key = f'ciclo{cycle_num}'
            
            # Datos manuales
            manual = my_data[size_label][cycle_key]
            
            # Datos del log
            if cycle_key in log_results and size_nodes in log_results[cycle_key]:
                log_data = log_results[cycle_key][size_nodes]
                
                # Comparar
                standalone_match = abs(manual['standalone'] - log_data['standalone_ms']) < 0.1 if manual['standalone'] else False
                burst_match = abs(manual['burst'] - log_data['burst_ms']) < 0.1 if manual['burst'] else False
                speedup_match = abs(manual['speedup'] - log_data['speedup']) < 0.01 if manual['speedup'] else False
                
                match = standalone_match and burst_match and speedup_match
                match_symbol = "✅" if match else "❌"
                
                if not match:
                    all_match = False
                
                # Manual
                print(f"{cycle_num:>8} {'Manual':>10} {manual['standalone'] or 'N/A':>12} {manual['burst'] or 'N/A':>12} {manual['speedup'] or 'N/A':>10} {match_symbol:>8}")
                
                # Log
                print(f"{' ':>8} {'Log':>10} {log_data['standalone_ms']:>12.0f} {log_data['burst_ms']:>12.0f} {log_data['speedup']:>10.2f}")
            else:
                # No hay datos en el log (esperado para FAILED o INTERRUPTED)
                has_data = manual['standalone'] is not None
                match_symbol = "✅" if not has_data else "⚠️ "
                
                if has_data:
                    all_match = False
                    
                print(f"{cycle_num:>8} {'Manual':>10} {manual['standalone'] or 'N/A':>12} {manual['burst'] or 'N/A':>12} {manual['speedup'] or 'N/A':>10} {match_symbol:>8}")
                print(f"{' ':>8} {'Log':>10} {'NO DATA':>12} {'NO DATA':>12} {'NO DATA':>10}")
    
    # Resultado final
    print("\n" + "="*80)
    if all_match:
        print("✅ VALIDACIÓN EXITOSA: Todos los datos manuales coinciden con los logs")
    else:
        print("❌ VALIDACIÓN FALLIDA: Hay discrepancias entre datos manuales y logs")
    print("="*80)
    
    # Verificar cálculos estadísticos
    print("\n" + "="*80)
    print("VERIFICACIÓN DE CÁLCULOS ESTADÍSTICOS")
    print("="*80)
    
    import numpy as np
    
    for size_label in ['3.0M', '4.0M', '4.5M', '5.0M', '6.0M']:
        print(f"\n{size_label}:")
        
        # Recopilar valores no nulos
        standalone_vals = [my_data[size_label][f'ciclo{i}']['standalone'] 
                          for i in [1,2,3] if my_data[size_label][f'ciclo{i}']['standalone'] is not None]
        burst_vals = [my_data[size_label][f'ciclo{i}']['burst'] 
                     for i in [1,2,3] if my_data[size_label][f'ciclo{i}']['burst'] is not None]
        speedup_vals = [my_data[size_label][f'ciclo{i}']['speedup'] 
                       for i in [1,2,3] if my_data[size_label][f'ciclo{i}']['speedup'] is not None]
        
        if standalone_vals:
            mean_s = np.mean(standalone_vals)
            std_s = np.std(standalone_vals, ddof=1) if len(standalone_vals) > 1 else 0
            cv_s = (std_s / mean_s * 100) if mean_s > 0 else 0
            print(f"  Standalone: mean={mean_s:.0f}ms, std={std_s:.0f}ms, CV={cv_s:.1f}% (n={len(standalone_vals)})")
        
        if burst_vals:
            mean_b = np.mean(burst_vals)
            std_b = np.std(burst_vals, ddof=1) if len(burst_vals) > 1 else 0
            cv_b = (std_b / mean_b * 100) if mean_b > 0 else 0
            print(f"  Burst:      mean={mean_b:.0f}ms, std={std_b:.0f}ms, CV={cv_b:.1f}% (n={len(burst_vals)})")
        
        if speedup_vals:
            mean_sp = np.mean(speedup_vals)
            std_sp = np.std(speedup_vals, ddof=1) if len(speedup_vals) > 1 else 0
            cv_sp = (std_sp / mean_sp * 100) if mean_sp > 0 else 0
            print(f"  Speedup:    mean={mean_sp:.2f}x, std={std_sp:.2f}x, CV={cv_sp:.1f}% (n={len(speedup_vals)})")

if __name__ == "__main__":
    main()
