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


WARM_BURST_KEY = "burst_warm_ms"
WARM_SPEEDUP_KEY = "speedup_warm"


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


def validate_payload(spec: ReportSpec, payload: dict[str, Any]) -> None:
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError(f"{spec.slug}: payload does not contain a valid 'results' list")
    if not results:
        raise ValueError(f"{spec.slug}: payload contains an empty 'results' list")

    required_keys = [spec.x_key, spec.standalone_key, spec.burst_span_key, spec.speedup_key]
    for index, row in enumerate(results):
        missing = [key for key in required_keys if key not in row]
        if missing:
            raise ValueError(
                f"{spec.slug}: row {index} is missing required keys: {', '.join(missing)}"
            )


def write_raw_copy(spec: ReportSpec, payload: dict[str, Any]) -> Path:
    out_path = RAW_DIR / f"{spec.slug}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def load_preferred_payload(spec: ReportSpec) -> tuple[dict[str, Any], Path]:
    candidate_paths = [spec.result_file, RAW_DIR / f"{spec.slug}.json"]
    errors: list[str] = []

    for path in candidate_paths:
        if not path.exists():
            continue
        try:
            payload = load_json(path)
            validate_payload(spec, payload)
            return payload, path
        except Exception as exc:
            errors.append(f"{path}: {exc}")

    joined = "; ".join(errors) if errors else "no candidate files found"
    raise ValueError(f"{spec.slug}: could not load a valid non-empty payload ({joined})")


def result_value(row: dict[str, Any], key: str | None) -> float | None:
    if not key:
        return None
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def metric_series(results: list[dict[str, Any]], key: str | None) -> list[float | None]:
    return [result_value(row, key) for row in results]


def has_metric(results: list[dict[str, Any]], key: str | None) -> bool:
    if not key:
        return False
    return any(result_value(row, key) is not None for row in results)


def has_total_metrics(spec: ReportSpec, payload: dict[str, Any]) -> bool:
    results = payload.get("results", [])
    return has_metric(results, "standalone_total_ms") and has_metric(results, spec.burst_total_key)


def has_warm_metrics(payload: dict[str, Any]) -> bool:
    results = payload.get("results", [])
    return has_metric(results, WARM_BURST_KEY) and has_metric(results, WARM_SPEEDUP_KEY)


def primary_mode(spec: ReportSpec, payload: dict[str, Any]) -> str:
    results = payload.get("results", [])
    if has_metric(results, spec.burst_total_key) and has_metric(results, spec.speedup_total_key):
        return "total"
    return "span"


def primary_metric_label(spec: ReportSpec, payload: dict[str, Any]) -> str:
    return "tiempo total extremo a extremo cold" if primary_mode(spec, payload) == "total" else "span algorítmico"


def primary_standalone_key(spec: ReportSpec, payload: dict[str, Any]) -> str:
    if primary_mode(spec, payload) == "total" and has_metric(payload.get("results", []), "standalone_total_ms"):
        return "standalone_total_ms"
    return spec.standalone_key


def primary_burst_key(spec: ReportSpec, payload: dict[str, Any]) -> str:
    if primary_mode(spec, payload) == "total" and spec.burst_total_key:
        return spec.burst_total_key
    return spec.burst_span_key


def primary_speedup_key(spec: ReportSpec, payload: dict[str, Any]) -> str:
    if primary_mode(spec, payload) == "total" and spec.speedup_total_key:
        return spec.speedup_total_key
    return spec.speedup_key


def metric_display_name(kind: str) -> str:
    return {
        "cold": "tiempo total cold",
        "warm": "tiempo total warm",
        "span": "span algorítmico",
    }[kind]


def metric_speedup_key(spec: ReportSpec, kind: str) -> str | None:
    if kind == "cold":
        return spec.speedup_total_key
    if kind == "warm":
        return WARM_SPEEDUP_KEY
    return spec.speedup_key


def metric_standalone_key(spec: ReportSpec, kind: str) -> str:
    if kind in {"cold", "warm"}:
        return "standalone_total_ms"
    return spec.standalone_key


def metric_burst_key(spec: ReportSpec, kind: str) -> str | None:
    if kind == "cold":
        return spec.burst_total_key
    if kind == "warm":
        return WARM_BURST_KEY
    return spec.burst_span_key


def available_metric_kinds(spec: ReportSpec, payload: dict[str, Any]) -> list[str]:
    kinds: list[str] = []
    if has_total_metrics(spec, payload) and spec.speedup_total_key:
        kinds.append("cold")
    if has_warm_metrics(payload):
        kinds.append("warm")
    kinds.append("span")
    return kinds


def metric_summary(spec: ReportSpec, payload: dict[str, Any], kind: str) -> str:
    results = payload.get("results", [])
    speedup_key = metric_speedup_key(spec, kind)
    x_key = spec.x_key
    label = metric_display_name(kind)
    if not speedup_key:
        return f"No hay datos suficientes para estimar el cruce según {label}."
    series = [result_value(row, speedup_key) for row in results]
    series = [value for value in series if value is not None]
    if not series:
        return f"No hay datos suficientes para estimar el cruce según {label}."
    if all(value > 1.0 for value in series):
        return f"Burst ya supera a standalone en todo el rango probado según {label}; el cruce queda por debajo del mínimo medido."
    if all(value < 1.0 for value in series):
        return f"Standalone sigue por delante en todo el rango probado según {label}; el cruce queda por encima del máximo medido."
    value, _intervals = estimate_crossing(results, x_key, speedup_key)
    if value is None:
        return f"No se pudo estimar un cruce único según {label}."
    unit = "filas" if x_key == "rows" else "usuarios" if x_key == "users" else "nodos"
    return f"Cruce estimado dentro del rango probado según {label}: aproximadamente {float(value):,.0f} {unit}."


def crossing_intervals_for_key(
    results: list[dict[str, Any]], x_key: str, speedup_key: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    intervals = []
    for idx in range(1, len(results)):
        prev = results[idx - 1]
        curr = results[idx]
        prev_speedup = result_value(prev, speedup_key)
        curr_speedup = result_value(curr, speedup_key)
        if prev_speedup is None or curr_speedup is None:
            continue
        if (prev_speedup - 1.0) * (curr_speedup - 1.0) <= 0 and prev_speedup != curr_speedup:
            intervals.append((prev, curr))
    return intervals


def estimate_crossing(
    results: list[dict[str, Any]], x_key: str, speedup_key: str
) -> tuple[float | None, list[tuple[dict[str, Any], dict[str, Any]]]]:
    intervals = crossing_intervals_for_key(results, x_key, speedup_key)
    upward = []
    for prev, curr in intervals:
        prev_speedup = result_value(prev, speedup_key)
        curr_speedup = result_value(curr, speedup_key)
        if prev_speedup is not None and curr_speedup is not None and prev_speedup < 1.0 <= curr_speedup:
            upward.append((prev, curr))
    if len(upward) != 1:
        return None, intervals
    prev, curr = upward[0]
    prev_speedup = result_value(prev, speedup_key)
    curr_speedup = result_value(curr, speedup_key)
    if prev_speedup is None or curr_speedup is None:
        return None, intervals
    slope = (curr_speedup - prev_speedup) / (curr[x_key] - prev[x_key])
    intercept = prev_speedup - slope * prev[x_key]
    return (1.0 - intercept) / slope, intervals


def make_figure(spec: ReportSpec, payload: dict[str, Any]) -> Path:
    results = payload.get("results", [])
    x_values = [int(row[spec.x_key]) for row in results]
    labels = [fmt_count(x, spec.x_key) for x in x_values]
    standalone_exec = metric_series(results, spec.standalone_key)
    standalone_total = metric_series(results, "standalone_total_ms")
    burst_span = metric_series(results, spec.burst_span_key)
    burst_total = metric_series(results, spec.burst_total_key)
    burst_warm = metric_series(results, WARM_BURST_KEY)
    speedup_span = metric_series(results, spec.speedup_key)
    speedup_total = metric_series(results, spec.speedup_total_key)
    speedup_warm = metric_series(results, WARM_SPEEDUP_KEY)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ax = axes[0]
    if any(v is not None for v in standalone_total) and any(v is not None for v in burst_total):
        ax.plot(labels, standalone_total, marker="o", linewidth=2, label="Standalone total", color="#0f766e")
        ax.plot(labels, burst_total, marker="^", linewidth=2, label="Burst cold total", color="#b45309")
        if any(v is not None for v in burst_warm):
            ax.plot(labels, burst_warm, marker="D", linewidth=2, label="Burst warm total", color="#7c3aed")
        ax.set_title(f"{spec.title}: tiempos totales")
    else:
        ax.plot(labels, standalone_exec, marker="o", linewidth=2, label="Standalone exec", color="#0f766e")
        ax.plot(labels, burst_span, marker="s", linewidth=2, label="Burst span", color="#1d4ed8")
        ax.set_title(f"{spec.title}: tiempos disponibles")
    ax.set_xlabel(spec.x_label)
    ax.set_ylabel("Tiempo (ms)")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    ax2.plot(labels, speedup_span, marker="s", linewidth=2, label="Speedup span", color="#dc2626")
    if any(v is not None for v in speedup_total):
        ax2.plot(labels, speedup_total, marker="^", linewidth=2, label="Speedup cold total", color="#2563eb")
    if any(v is not None for v in speedup_warm):
        ax2.plot(labels, speedup_warm, marker="D", linewidth=2, label="Speedup warm total", color="#7c3aed")
    ax2.axhline(1.0, color="#374151", linestyle="--", linewidth=1)
    ax2.set_title(f"{spec.title}: speedups")
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
    return metric_summary(spec, payload, "cold" if has_total_metrics(spec, payload) else "span")


def crossover_context(spec: ReportSpec, payload: dict[str, Any]) -> list[str]:
    results = payload.get("results", [])
    speedup_key = primary_speedup_key(spec, payload)
    metric_label = primary_metric_label(spec, payload)
    intervals = crossing_intervals_for_key(results, spec.x_key, speedup_key)
    if not intervals:
        return []

    lines: list[str] = []
    rendered = []
    for prev, curr in intervals:
        rendered.append(
            f"{fmt_count(int(prev[spec.x_key]), spec.x_key)} a {fmt_count(int(curr[spec.x_key]), spec.x_key)}"
        )
    if rendered:
        lines.append(
            f"Intervalos con cambio de ganador observados según {metric_label}: {', '.join(rendered)}."
        )
    if len(intervals) > 1:
        lines.append(
            f"La curva de speedup no fue monotónica según {metric_label}, así que cualquier cruce estimado debe leerse como aproximación local."
        )
    return lines


def key_findings(spec: ReportSpec, payload: dict[str, Any]) -> list[str]:
    results = payload.get("results", [])
    if not results:
        return ["Sin resultados agregados."]

    first = results[0]
    last = results[-1]
    findings: list[str] = []
    if has_total_metrics(spec, payload):
        findings.append(
            f"En el punto menor ({fmt_count(int(first[spec.x_key]), spec.x_key)}), standalone total tarda "
            f"{float(first['standalone_total_ms']):.1f} ms y burst cold total {float(first[spec.burst_total_key]):.1f} ms."
        )
        findings.append(
            f"En el punto mayor ({fmt_count(int(last[spec.x_key]), spec.x_key)}), standalone total tarda "
            f"{float(last['standalone_total_ms']):.1f} ms y burst cold total {float(last[spec.burst_total_key]):.1f} ms."
        )
        findings.append(metric_summary(spec, payload, "cold"))
    else:
        findings.append(
            f"En el punto menor ({fmt_count(int(first[spec.x_key]), spec.x_key)}), standalone exec tarda "
            f"{float(first[spec.standalone_key]):.1f} ms y burst span {float(first[spec.burst_span_key]):.1f} ms."
        )
        findings.append(
            f"En el punto mayor ({fmt_count(int(last[spec.x_key]), spec.x_key)}), standalone exec tarda "
            f"{float(last[spec.standalone_key]):.1f} ms y burst span {float(last[spec.burst_span_key]):.1f} ms."
        )
    if has_warm_metrics(payload):
        findings.append(metric_summary(spec, payload, "warm"))
    else:
        findings.append("La campaña actual no publica todavía una métrica warm end-to-end separada; solo pueden compararse explícitamente cold total y span.")
    findings.append(metric_summary(spec, payload, "span"))
    if has_total_metrics(spec, payload) and has_metric(results, spec.speedup_key):
        cold_values = [result_value(row, spec.speedup_total_key) for row in results]
        cold_values = [value for value in cold_values if value is not None]
        span_values = [result_value(row, spec.speedup_key) for row in results]
        span_values = [value for value in span_values if value is not None]
        if cold_values and span_values and any(c < 1.0 for c in cold_values) and all(s > 1.0 for s in span_values):
            findings.append(
                "Aquí se ve bien la diferencia entre sistema y algoritmo: burst gana en span, pero el cold end-to-end todavía no amortiza el arranque y la coordinación."
            )
    findings.extend(crossover_context(spec, payload))

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
        (
            "- Marco de lectura: siguiendo COST, la comparación principal se hace sobre tiempo end-to-end real; "
            "siguiendo el artículo de burst computing, se separa ese coste del span algorítmico para entender cuánto aporta el paralelismo útil."
        ),
    ]
    if has_total_metrics(spec, payload):
        lines.append("- Métricas reportadas: cold end-to-end, span algorítmico, y warm end-to-end solo cuando el benchmark lo publique explícitamente.")
    else:
        lines.append(
            "- Limitación: este informe no dispone todavía de una métrica cold total comparable para todo el rango, así que las conclusiones quedan apoyadas sobre todo en span algorítmico."
        )
    if not has_warm_metrics(payload):
        lines.append("- En esta campaña no hay una columna warm separada; no se ha imputado artificialmente a partir de otras marcas temporales.")
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
        f"| {spec.x_label} | SA total (ms) | Burst cold (ms) | Burst warm (ms) | SA exec (ms) | Burst span (ms) | Speedup cold | Speedup warm | Speedup span |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        x_val = fmt_count(int(row[spec.x_key]), spec.x_key)
        sa_total = result_value(row, "standalone_total_ms")
        burst_cold = result_value(row, spec.burst_total_key)
        burst_warm = result_value(row, WARM_BURST_KEY)
        sa_exec = result_value(row, spec.standalone_key)
        burst_span = result_value(row, spec.burst_span_key)
        sp_cold = result_value(row, spec.speedup_total_key)
        sp_warm = result_value(row, WARM_SPEEDUP_KEY)
        sp_span = result_value(row, spec.speedup_key)
        lines.append(
            "| {x} | {sa_total} | {burst_cold} | {burst_warm} | {sa_exec:.2f} | {burst_span:.2f} | {sp_cold} | {sp_warm} | {sp_span:.2f}x |".format(
                x=x_val,
                sa_total=f"{sa_total:.2f}" if sa_total is not None else "n/d",
                burst_cold=f"{burst_cold:.2f}" if burst_cold is not None else "n/d",
                burst_warm=f"{burst_warm:.2f}" if burst_warm is not None else "n/d",
                sa_exec=sa_exec,
                burst_span=burst_span,
                sp_cold=f"{sp_cold:.2f}x" if sp_cold is not None else "n/d",
                sp_warm=f"{sp_warm:.2f}x" if sp_warm is not None else "n/d",
                sp_span=sp_span,
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
            "## Lectura de Métricas",
            "",
            "- `Cold end-to-end`: mide la latencia real observada si la campaña dispara workers fríos.",
            "- `Warm end-to-end`: modela workers precalentados; solo se reporta cuando el benchmark la publica explícitamente.",
            "- `Span algorítmico`: aísla el tramo de cómputo distribuido y sirve para explicar la escalabilidad del algoritmo, no para sustituir al tiempo real del sistema.",
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
            burst_total_key="burst_total_ms",
            speedup_key="speedup",
            speedup_total_key="speedup_total",
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
            burst_total_key="burst_total_ms",
            speedup_key="speedup",
            speedup_total_key="speedup_total",
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
            burst_total_key="burst_total_ms",
            speedup_key="speedup",
            speedup_total_key="speedup_total",
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
            burst_total_key="burst_total_ms",
            speedup_key="speedup",
            speedup_total_key="speedup_total",
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
            burst_total_key="burst_total_ms",
            speedup_key="speedup",
            speedup_total_key="speedup_total",
            dataset_name="grafo sintético por componentes disjuntas",
            theory="Union-Find mantiene componentes conexas mediante operaciones de unión y búsqueda de representante, normalmente optimizadas con union by rank y path compression.",
            standalone_desc="binario Rust local que procesa el grafo completo y devuelve la partición final.",
            burst_desc="acción distribuida que ejecuta uniones locales por partición y luego coordina la fusión global entre workers.",
            dataset_note="Los grafos sintéticos se generan una vez por tamaño con el mismo número de componentes y aristas por nodo para ambas implementaciones.",
            validation_note="La validación fuerte usa hash canónico de la partición además del número de componentes. El benchmark ya quedó preparado para separar tiempo total y span, pero la campaña agregada actual todavía debe reejecutarse si se quiere una lectura estrictamente alineada con cold starts.",
            crossover_key="crossover_estimate",
        ),
    ]


def generate_all() -> None:
    ensure_dirs()
    index: dict[str, Any] = {}
    for spec in build_specs():
        try:
            payload, source_path = load_preferred_payload(spec)
        except ValueError as exc:
            print(f"Skipping {spec.slug}: {exc}")
            continue
        raw_copy = write_raw_copy(spec, payload)
        figure = make_figure(spec, payload)
        doc = write_markdown(spec, payload, figure)
        index[spec.slug] = {
            "result_file": str(source_path),
            "raw_copy": str(raw_copy),
            "figure": str(figure),
            "doc": str(doc),
            "points": len(payload.get("results", [])),
        }
    (RAW_DIR / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Generated {len(index)} reports in {OUT_ROOT}")


if __name__ == "__main__":
    generate_all()
