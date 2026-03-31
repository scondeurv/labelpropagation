# Label Propagation - Flujo Estandar de Benchmarks

Este repositorio queda normalizado con 3 grupos de scripts para el flujo completo:

1. Preparacion de datos:
   - `setup_large_lp_data.py`
   - `upload_to_minio.py`
2. Ejecucion:
   - `run_standalone.py`
   - `run_burst.py`
   - `validate_crossover.py`
3. Visualizacion:
   - `plot_new_results.py`

## Orden recomendado

1. Generar datos locales:

```bash
python3 setup_large_lp_data.py --nodes 1000000 --partitions 4 --no-s3 --density 10
1. Subir a MinIO/S3:

```bash
python3 upload_to_minio.py --nodes 1000000 --partitions 4 --bucket test-bucket --endpoint localhost:9000
```

1. Ejecutar standalone:

```bash
python3 run_standalone.py --nodes 1000000
```

1. Ejecutar burst:

```bash
python3 run_burst.py --nodes 1000000 --partitions 4 --memory 4096 --s3-endpoint http://minio-service.default.svc.cluster.local:9000 --bucket test-bucket
```

1. Ejecutar ronda comparativa:

```bash
python3 validate_crossover.py
```

1. Generar graficos estandar:

```bash
python3 plot_new_results.py
```

## Equivalencia de argumentos entre algoritmos

| Concepto | LP | BFS | SSSP | Louvain | GB | CF |
| --- | --- | --- | --- | --- | --- | --- |
| Tamano principal de entrada | `--nodes` | `--nodes` | `--nodes` | `--nodes` | `--rows` | `--users` |
| Particiones S3 | `--partitions` | `--partitions` | `--partitions` | `--partitions` | `--partitions` | `--partitions` |
| Endpoint MinIO local | `--endpoint` | `--endpoint` | `--endpoint` | `--endpoint` | `--endpoint` | `--endpoint` |
| Bucket S3 | `--bucket` | `--bucket` | `--bucket` | `--bucket` | `--bucket` | `--bucket` |
| Endpoint S3 en cluster (burst) | `--s3-endpoint` | `--s3-endpoint` | `--s3-endpoint` | `--s3-endpoint` | `--s3-endpoint` | `--s3-endpoint` |
| Memoria worker burst | `--memory` | `--memory` | `--memory` | `--memory` | `--memory` | `--memory` |
| Granularidad burst | `--granularity` | `--granularity` | `--granularity` | `--granularity` | `--granularity` | `--granularity` |
| Iteraciones/pases de algoritmo | `--iter` | `--max-levels` | `--max-iterations` | `--max-passes` | `--num-trees` | `--iterations` |
| Parametros extra de datos | `--density` | `--density` | `--density --max-weight` | `--mode --communities --p-in --p-out` | (dataset HIGGS preparado por filas) | `--items --avg-ratings --synthetic` |

Referencia rapida de este repo (LP): `nodes` + `partitions` + `memory`.

## Notas

- Los puntos de corte se calculan sobre mediciones reales, no sobre valores hardcodeados.
- `validate_crossover.py` escribe resultados en `crossover_validation_results.json`.
- El barrido nuevo de recursos esta en `run_resource_sweep_campaign.py` y deja sus artefactos en `resource_sweep/`.

## Troubleshooting rapido

- Error de modulo Python faltante: instala dependencias del repo (`pip install -r requirements.txt`) o usa el entorno virtual del proyecto.
- Error de binario standalone no encontrado: compila el binario Rust correspondiente (`cargo build --release`) dentro del subproyecto standalone.
- Error conectando a MinIO local: revisa `--endpoint` (normalmente `localhost:9000`) y credenciales de MinIO.
- Error en burst por endpoint S3: para workers en cluster usa `--s3-endpoint http://minio-service.default.svc.cluster.local:9000`.
- Error por datos faltantes en S3: ejecuta antes `upload_to_minio.py` con el mismo tamano y `--partitions` que usaras en benchmark.
