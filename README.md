# Label Propagation - Flujo Estandar de Benchmarks

Este repositorio queda normalizado con 6 scripts Python para el flujo completo:

1. `generate_test_data.py`: genera datos locales de prueba.
2. `upload_to_minio.py`: genera y sube datos a MinIO/S3.
3. `run_standalone.py`: ejecuta solo la version standalone.
4. `run_burst.py`: ejecuta solo la version burst.
5. `run_benchmark_round.py`: ejecuta una ronda comparativa completa.
6. `plot_benchmark_results.py`: genera los graficos estandar.

## Orden recomendado

1. Generar datos locales:

```bash
python3 generate_test_data.py --nodes 1000000 --partitions 4 --density 10
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
python3 run_benchmark_round.py
```

1. Generar graficos estandar:

```bash
python3 plot_benchmark_results.py
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
- `run_benchmark_round.py` escribe resultados en `crossover_validation_results.json`.

## Troubleshooting rapido

- Error de modulo Python faltante: instala dependencias del repo (`pip install -r requirements.txt`) o usa el entorno virtual del proyecto.
- Error de binario standalone no encontrado: compila el binario Rust correspondiente (`cargo build --release`) dentro del subproyecto standalone.
- Error conectando a MinIO local: revisa `--endpoint` (normalmente `localhost:9000`) y credenciales de MinIO.
- Error en burst por endpoint S3: para workers en cluster usa `--s3-endpoint http://minio-service.default.svc.cluster.local:9000`.
- Error por datos faltantes en S3: ejecuta antes `upload_to_minio.py` con el mismo tamano y `--partitions` que usaras en benchmark.
