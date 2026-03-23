#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path("/home/sergio/src")
HERE = Path(__file__).resolve().parent
OUT_ROOT = HERE / "benchmark_reports"
RAW_DIR = OUT_ROOT / "raw"
DOC_DIR = OUT_ROOT / "docs"
FIG_DIR = OUT_ROOT / "figures"
MPL_DIR = Path("/tmp/mpl-benchmark-campaign")


@dataclass(frozen=True)
class ReportSpec:
    slug: str
    title: str
    result_file: Path
    x_key: str
    x_label: str
    standalone_key: str
    burst_span_key: str
    burst_total_key: str | None
    speedup_key: str
    speedup_total_key: str | None
    dataset_name: str
    theory: str
    standalone_desc: str
    burst_desc: str
    dataset_note: str
    validation_note: str
    crossover_key: str | None = None


def ensure_dirs() -> None:
    for directory in (OUT_ROOT, RAW_DIR, DOC_DIR, FIG_DIR, MPL_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def fmt_count(value: int, axis: str) -> str:
    if axis == "nodes" or axis == "users" or axis == "rows":
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if value >= 1_000:
            return f"{value / 1_000:.0f}k"
    return f"{value:,}"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_raw_copy(spec: ReportSpec, payload: dict[str, Any]) -> Path:
    out_path = RAW_DIR / f"{spec.slug}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def result_value(row: dict[str, Any], key: str | None) -> float | None:
    if not key:
        return None
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def make_figure(spec: ReportSpec, payload: dict[str, Any]) -> Path:
    results = payload.get("results", [])
    x_values = [int(row[spec.x_key]) for row in results]
    labels = [fmt_count(x, spec.x_key) for x in x_values]
    standalone = [result_value(row, spec.standalone_key) for row in results]
    burst_span = [result_value(row, spec.burst_span_key) for row in results]
    burst_total = [result_value(row, spec.burst_total_key) for row in results]
    speedup = [result_value(row, spec.speedup_key) for row in results]
    speedup_total = [result_value(row, spec.speedup_total_key) for row in results]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    ax.plot(labels, standalone, marker="o", linewidth=2, label="Standalone", color="#0f766e")
    ax.plot(labels, burst_span, marker="s", linewidth=2, label="Burst span", color="#1d4ed8")
    if any(v is not None for v in burst_total):
        ax.plot(labels, burst_total, marker="^", linewidth=2, label="Burst total", color="#b45309")
    ax.set_title(f"{spec.title}: tiempos")
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel("Tiempo (ms)")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(labels, speedup, marker="s", linewidth=2, label="Speedup algorítmico", color="#dc2626")
    if any(v is not None for v in speedup_total):
        ax2.plot(labels, speedup_total, marker="^", linewidth=2, label="Speedup total", color="#7c3aed")
    ax2.axhline(1.0, color="#374151", linestyle="--", linewidth=1)
    ax2.set_title(f"{spec.title}: speedup")
    ax2.set_xlabel(spec.x_label)
    ax2.set_ylabel("Speedup (x)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    figure_path = FIG_DIR / f"{spec.slug}.svg"
    fig.savefig(figure_path, format="svg")
    plt.close(fig)
    return figure_path


def detect_validation_summary(spec: ReportSpec, payload: dict[str, Any]) -> str:
    if spec.slug in {"bfs", "sssp"}:
        return spec.validation_note
    return spec.validation_note


def crossover_summary(spec: ReportSpec, payload: dict[str, Any]) -> str:
    key = spec.crossover_key
    value = payload.get(key) if key else None
    if value is None:
        return "No se detectó cruce dentro del rango probado."
    if spec.x_key == "rows":
        unit = "filas"
    elif spec.x_key == "users":
        unit = "usuarios"
    else:
        unit = "nodos"
    return f"Cruce estimado dentro del rango probado: aproximadamente {float(value):,.0f} {unit}."


def key_findings(spec: ReportSpec, payload: dict[str, Any]) -> list[str]:
    results = payload.get("results", [])
    if not results:
        return ["Sin resultados agregados."]

    first = results[0]
    last = results[-1]
    findings = [
        (
            f"En el punto menor ({fmt_count(int(first[spec.x_key]), spec.x_key)}), "
            f"standalone tarda {float(first[spec.standalone_key]):.1f} ms y burst span "
            f"{float(first[spec.burst_span_key]):.1f} ms."
        ),
        (
            f"En el punto mayor ({fmt_count(int(last[spec.x_key]), spec.x_key)}), "
            f"standalone tarda {float(last[spec.standalone_key]):.1f} ms y burst span "
            f"{float(last[spec.burst_span_key]):.1f} ms."
        ),
        crossover_summary(spec, payload),
    ]

    if spec.slug == "louvain":
        findings.append(
            "En campañas posteriores, 300k nodos validó correctamente y 400k/500k fallaron en runtime burst, así que el techo operativo quedó entre 300k y 400k."
        )
    if spec.slug == "sssp":
        findings.append("La campaña confirmó que el fallo previo estaba en el preflight S3 y no en el algoritmo burst.")
    return findings


def dataset_lines(spec: ReportSpec, payload: dict[str, Any]) -> list[str]:
    results = payload.get("results", [])
    tested = ", ".join(fmt_count(int(row[spec.x_key]), spec.x_key) for row in results)
    config = payload.get("configuration", {})
    lines = [
        f"- Dataset base: {spec.dataset_name}.",
        f"- Puntos probados: {tested}.",
        f"- Detalle: {spec.dataset_note}",
    ]
    if config:
        compact = ", ".join(f"{k}={v}" for k, v in config.items())
        lines.append(f"- Configuración de campaña: {compact}.")
    return lines


def result_table(spec: ReportSpec, payload: dict[str, Any]) -> str:
    rows = payload.get("results", [])
    header = (
        f"| {spec.x_label} | Standalone (ms) | Burst span (ms) | "
        f"Burst total (ms) | Speedup span | Speedup total | Ganador |\n"
        f"| --- | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    lines = [header]
    for row in rows:
        x_val = fmt_count(int(row[spec.x_key]), spec.x_key)
        sa = result_value(row, spec.standalone_key)
        bs = result_value(row, spec.burst_span_key)
        bt = result_value(row, spec.burst_total_key)
        sp = result_value(row, spec.speedup_key)
        st = result_value(row, spec.speedup_total_key)
        lines.append(
            f"| {x_val} | {sa:.2f} | {bs:.2f} | "
            f"{bt:.2f if bt is not None else 'n/d'} | "
            f"{sp:.2f}x | {st:.2f}x if False else '' |"
        )
    return "\n".join(lines)


def build_table_lines(spec: ReportSpec, payload: dict[str, Any]) -> list[str]:
    rows = payload.get("results", [])
    lines = [
        f"| {spec.x_label} | Standalone (ms) | Burst span (ms) | Burst total (ms) | Speedup span | Speedup total | Ganador |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        x_val = fmt_count(int(row[spec.x_key]), spec.x_key)
        sa = result_value(row, spec.standalone_key)
        bs = result_value(row, spec.burst_span_key)
        bt = result_value(row, spec.burst_total_key)
        sp = result_value(row, spec.speedup_key)
        st = result_value(row, spec.speedup_total_key)
        winner = row.get("winner", "n/d")
        lines.append(
            "| {x} | {sa:.2f} | {bs:.2f} | {bt} | {sp:.2f}x | {stxt} | {winner} |".format(
                x=x_val,
                sa=sa,
                bs=bs,
                bt=f"{bt:.2f}" if bt is not None else "n/d",
                sp=sp,
                stxt=f"{st:.2f}x" if st is not None else "n/d",
                winner=winner,
            )
        )
    return lines


def write_markdown(spec: ReportSpec, payload: dict[str, Any], figure_path: Path) -> Path:
    rel_figure = Path(os.path.relpath(figure_path, DOC_DIR))
    findings = key_findings(spec, payload)
    table_lines = build_table_lines(spec, payload)
    content = "\n".join(
        [
            f"# {spec.title}",
            "",
            "## Teoría",
            "",
            spec.theory,
            "",
            "## Implementaciones comparadas",
            "",
            f"- **Standalone**: {spec.standalone_desc}",
            f"- **Burst**: {spec.burst_desc}",
            "",
            "## Dataset y metodología",
            "",
            *dataset_lines(spec, payload),
            f"- Validación: {detect_validation_summary(spec, payload)}",
            "",
            "## Resultados",
            "",
            f"![{spec.title}]({rel_figure.as_posix()})",
            "",
            *table_lines,
            "",
            "## Hallazgos",
            "",
            *(f"- {line}" for line in findings),
            "",
        ]
    )
    md_path = DOC_DIR / f"{spec.slug}.md"
    md_path.write_text(content, encoding="utf-8")
    return md_path


def build_specs() -> list[ReportSpec]:
    return [
        ReportSpec(
            slug="bfs",
            title="Breadth-First Search",
            result_file=ROOT / "bfs/crossover_bfs_results.json",
            x_key="nodes",
            x_label="Nodos",
            standalone_key="standalone_ms",
            burst_span_key="burst_ms",
            burst_total_key=None,
            speedup_key="speedup",
            speedup_total_key=None,
            dataset_name="grafo dirigido sintético",
            theory="BFS recorre un grafo por niveles a partir de un nodo fuente y produce la distancia en aristas desde esa fuente al resto de nodos alcanzables.",
            standalone_desc="binario Rust monohilo que carga el grafo completo y expande una frontera FIFO nivel a nivel.",
            burst_desc="acción distribuida en OpenWhisk que reparte el grafo por particiones y sincroniza la frontera global entre workers.",
            dataset_note="Se reutilizó exactamente el mismo grafo por tamaño, subiendo a MinIO las particiones que consume burst y usando la copia local para standalone.",
            validation_note="Para tamaños grandes se reutilizó el mismo dataset en todas las repeticiones; la comparación funcional fuerte se apoya en la revisión estática y en las campañas pequeñas previas, porque el chequeo completo del vector de niveles no se ejecutó en memoria a estos tamaños.",
            crossover_key="crossover_estimate",
        ),
        ReportSpec(
            slug="sssp",
            title="Single-Source Shortest Paths",
            result_file=ROOT / "sssp/crossover_sssp_results.json",
            x_key="nodes",
            x_label="Nodos",
            standalone_key="standalone_ms",
            burst_span_key="burst_ms",
            burst_total_key=None,
            speedup_key="speedup",
            speedup_total_key=None,
            dataset_name="grafo ponderado sintético con pesos no negativos",
            theory="SSSP calcula las distancias mínimas desde una fuente al resto de nodos. En este workspace se trabaja con pesos no negativos y la versión burst propaga relajaciones entre particiones.",
            standalone_desc="binario Rust local que ejecuta el algoritmo de caminos mínimos sobre el grafo completo.",
            burst_desc="versión distribuida que reparte aristas por partición y coordina mejoras de distancia a través del middleware burst.",
            dataset_note="Los grafos se generaron una vez por tamaño con la misma semilla y los mismos pesos para ambas implementaciones.",
            validation_note="Las repeticiones reutilizan el mismo dataset; el benchmark grande omite la validación in-memory completa por coste, pero sí compara métricas de salida consistentes y se ejecutó sobre exactamente la misma entrada.",
            crossover_key="crossover_estimate",
        ),
        ReportSpec(
            slug="labelpropagation",
            title="Label Propagation",
            result_file=ROOT / "labelpropagation/crossover_validation_results.json",
            x_key="nodes",
            x_label="Nodos",
            standalone_key="standalone_ms",
            burst_span_key="burst_ms",
            burst_total_key=None,
            speedup_key="speedup",
            speedup_total_key=None,
            dataset_name="grafo sintético con propagación de etiquetas",
            theory="Label Propagation difunde etiquetas por vecindad hasta estabilizarse y suele usarse para clasificación semisupervisada o detección de comunidades ligeras.",
            standalone_desc="implementación Rust monohilo con acceso completo al grafo y a las etiquetas.",
            burst_desc="acción distribuida que reparte el grafo por particiones y agrega los votos de etiquetas entre workers.",
            dataset_note="La campaña seria reutilizó el mismo dataset por tamaño y la misma semilla en standalone y burst.",
            validation_note="Las rondas de la campaña reutilizan el mismo dataset y la equivalencia semántica se corrigió antes de lanzar esta tanda.",
            crossover_key="crossover_estimate",
        ),
        ReportSpec(
            slug="louvain",
            title="Louvain Community Detection",
            result_file=ROOT / "louvain/crossover_louvain_results.partial.json",
            x_key="nodes",
            x_label="Nodos",
            standalone_key="standalone_ms",
            burst_span_key="burst_span_ms",
            burst_total_key="burst_total_ms",
            speedup_key="speedup",
            speedup_total_key="speedup_total",
            dataset_name="grafo planted-partition ponderado",
            theory="Louvain maximiza modularidad moviendo nodos entre comunidades y, en su formulación habitual, reagrupa la solución para refinarla jerárquicamente.",
            standalone_desc="binario Rust que optimiza modularidad sobre el grafo completo.",
            burst_desc="versión distribuida que reparte el grafo y sincroniza cambios de comunidad y métricas globales entre workers.",
            dataset_note="Se usaron grafos planted-partition con semillas fijas y el mismo dataset por tamaño para ambas implementaciones.",
            validation_note="Las campañas cerradas hasta 200k validaron estrictamente la partición; además, una corrida manual posterior validó 300k y mostró el techo operativo burst antes de 400k.",
            crossover_key="crossover_estimate_nodes",
        ),
        ReportSpec(
            slug="gradientboosting",
            title="Gradient Boosting",
            result_file=ROOT / "gradientboosting/crossover_gb_results.json",
            x_key="rows",
            x_label="Filas",
            standalone_key="standalone_ms",
            burst_span_key="burst_ms",
            burst_total_key=None,
            speedup_key="speedup",
            speedup_total_key=None,
            dataset_name="subconjuntos crecientes de HIGGS",
            theory="Gradient Boosting construye un ensamble aditivo de árboles donde cada árbol intenta corregir el residuo del ensamble acumulado.",
            standalone_desc="binario Rust monohilo que entrena el ensamble completo localmente.",
            burst_desc="acción distribuida que reparte filas entre workers y agrega histogramas parciales para construir los árboles.",
            dataset_note="Cada tamaño reutiliza exactamente el mismo subconjunto de HIGGS en local y sus particiones equivalentes en MinIO.",
            validation_note="Cada repetición valida accuracy y logloss contra la salida burst estructurada en S3.",
            crossover_key="crossover_estimate",
        ),
        ReportSpec(
            slug="collaborativefiltering",
            title="Collaborative Filtering",
            result_file=ROOT / "collaborativefiltering/crossover_cf_results.json",
            x_key="users",
            x_label="Usuarios",
            standalone_key="standalone_ms",
            burst_span_key="burst_span_ms",
            burst_total_key="burst_total_ms",
            speedup_key="speedup",
            speedup_total_key="speedup_total",
            dataset_name="datasets sintéticos de ratings",
            theory="Collaborative Filtering con ALS factoriza la matriz usuario-item en vectores latentes alternando pasos de optimización para usuarios e ítems.",
            standalone_desc="binario Rust local que entrena ALS sobre el dataset completo.",
            burst_desc="versión burst que reparte ratings entre workers y coordina las actualizaciones de factores latentes.",
            dataset_note="Cada punto usa un dataset sintético fijo por tamaño, reutilizado entre repeticiones y replicado a MinIO para burst.",
            validation_note="La validación compara RMSE y parámetros estructurados de la ejecución burst frente al standalone.",
            crossover_key="crossover_estimate_users",
        ),
        ReportSpec(
            slug="unionfind",
            title="Union-Find",
            result_file=ROOT / "unionfind/uf_crossover_validation_results.json",
            x_key="nodes",
            x_label="Nodos",
            standalone_key="standalone_ms",
            burst_span_key="burst_ms",
            burst_total_key=None,
            speedup_key="speedup",
            speedup_total_key=None,
            dataset_name="grafo sintético por componentes disjuntas",
            theory="Union-Find mantiene componentes conexas mediante operaciones de unión y búsqueda de representante, normalmente optimizadas con union by rank y path compression.",
            standalone_desc="binario Rust local que procesa el grafo completo y devuelve la partición final.",
            burst_desc="acción distribuida que ejecuta uniones locales por partición y luego coordina la fusión global entre workers.",
            dataset_note="Los grafos sintéticos se generan una vez por tamaño con el mismo número de componentes y aristas por nodo para ambas implementaciones.",
            validation_note="La validación fuerte usa hash canónico de la partición además del número de componentes.",
            crossover_key="crossover_estimate",
        ),
    ]


def generate_all() -> None:
    ensure_dirs()
    index: dict[str, Any] = {}
    for spec in build_specs():
        if not spec.result_file.exists():
            continue
        payload = load_json(spec.result_file)
        raw_copy = write_raw_copy(spec, payload)
        figure = make_figure(spec, payload)
        doc = write_markdown(spec, payload, figure)
        index[spec.slug] = {
            "result_file": str(spec.result_file),
            "raw_copy": str(raw_copy),
            "figure": str(figure),
            "doc": str(doc),
            "points": len(payload.get("results", [])),
        }
    (RAW_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Generated {len(index)} reports in {OUT_ROOT}")


if __name__ == "__main__":
    generate_all()
