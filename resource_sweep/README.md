# Resource Sweep

Este directorio contiene la nueva campana de barrido de recursos para `Burst/OpenWhisk` y `Spark`.

Fases disponibles:

```bash
python3 /home/sergio/src/labelpropagation/run_resource_sweep_campaign.py --phase plan
python3 /home/sergio/src/labelpropagation/run_resource_sweep_campaign.py --phase burst-feasibility --prepare-burst-cluster
python3 /home/sergio/src/labelpropagation/run_resource_sweep_campaign.py --phase spark-feasibility --prepare-spark-cluster --stop-spark-cluster
python3 /home/sergio/src/labelpropagation/run_resource_sweep_campaign.py --phase config-sweep --prepare-burst-cluster --prepare-spark-cluster --stop-spark-cluster
python3 /home/sergio/src/labelpropagation/run_resource_sweep_campaign.py --phase size-sweep --prepare-burst-cluster --prepare-spark-cluster --stop-spark-cluster
```

Presupuesto por defecto del host:

- reserva para SO: `8192 MB` y `2` hilos logicos
- reserva fija Burst para servicios: `8192 MB` y `6` CPU
- Spark master: `1 CPU`, `1g`

Resultados:

- `plans/`: plan materializado y calculos de factibilidad
- `feasibility/`: criba de configuraciones validas
- `config_sweep/`: barrido de particiones y granularidad a tamano fijo
- `size_sweep/`: escalado por tamano usando la mejor configuracion por presupuesto
- `reports/`: figuras, tablas y resumen Markdown generados desde los sweeps finales

Generacion de reportes:

```bash
MPLCONFIGDIR=/tmp/mpl-resource-sweep \
  /home/sergio/src/labelpropagation/.venv/bin/python \
  /home/sergio/src/labelpropagation/generate_resource_sweep_reports.py
```

Notas:

- `Burst` y `Spark` se ejecutan en fases separadas.
- `g = partitions` puede ejecutarse, pero no debe entrar en tablas de fases hasta corregir completamente esa instrumentacion.
- `WCC` se integra en los sweeps, pero la criba de memoria se hereda de la familia `SSSP` si no hay una criba propia previa.
