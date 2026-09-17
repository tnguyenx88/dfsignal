"""Public Kaggle GPU specifications ingestion, normalization, and signals.

The Kaggle export is a wide, source-native table.  The adapter intentionally
keeps source values alongside canonical fields: product names, release dates,
and generation labels are not inferred silently when the source is incomplete.
"""

from io import BytesIO
import hashlib
from collections.abc import Mapping
from pathlib import Path
import re
import zipfile
from typing import Any

import httpx
import pandas as pd

from ..config import load_yaml


KAGGLE_GPU_URL = "https://www.kaggle.com/api/v1/datasets/download/ellimaaac/gpus-specs-from-1986-to-2026"

_MISSING_TEXT = frozenset({"", "-", "--", "—", "na", "n/a", "nan", "none", "null", "unknown"})
_NUMBER_PATTERN = r"([-+]?\d*\.?\d+)"
_UNIT_PATTERN = r"(?i)(kib|mib|gib|tib|kb|mb|gb|tb|w|mw|nm|bit|bits)"
_MONTH_PATTERN = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)"
)

def _gpu_config(config_dir: str | Path = "config", filename: str = "gpu_generations.yaml") -> dict[str, Any]:
    path = Path(config_dir) / filename
    if not path.exists():
        path = Path("config") / filename
    return load_yaml(path) if path.exists() else {}


def _canonical_generation(
    name: object,
    vendor: object,
    architecture: object,
    raw_generation: object,
    *,
    config: Mapping[str, Any] | None = None,
) -> tuple[str, str, str]:
    """Map observed GPU names to a configured taxonomy without guessing."""
    config = config or {}
    combined = " ".join(
        str(value) for value in (name, architecture, raw_generation)
        if value is not None and not pd.isna(value)
    )
    vendor_text = str(vendor or "").casefold()
    for rule in config.get("rules", []) if isinstance(config.get("rules"), list) else []:
        if not isinstance(rule, Mapping):
            continue
        expected_vendor = str(rule.get("vendor", "")).casefold()
        if expected_vendor and expected_vendor not in vendor_text:
            continue
        try:
            matched = re.search(str(rule.get("pattern", "")), combined, flags=re.IGNORECASE)
        except re.error:
            matched = None
        if matched:
            return str(rule.get("canonical_generation", "UNKNOWN")), "config_rule", "PASS"
    if not combined.strip():
        return "UNKNOWN", "unknown", "REVIEW"
    return "UNKNOWN", "unmapped", "REVIEW"


def _apply_gpu_override(
    row: Mapping[str, Any],
    *,
    config: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str | None]:
    """Apply an explicit reviewed override after automatic rules."""
    result = dict(row)
    for override in config.get("overrides", []) if isinstance(config, Mapping) else []:
        if not isinstance(override, Mapping):
            continue
        match = override.get("match", {})
        if not isinstance(match, Mapping):
            continue
        if any(str(result.get(key, "")).casefold() != str(value).casefold() for key, value in match.items()):
            continue
        for key in ("generation", "market_segment", "performance_tier"):
            if key in override:
                result[key] = override[key]
        return result, str(override.get("override_reason", "reviewed manual override"))
    return result, None

def _canonical_segment(value: object) -> str:
    aliases = {
        "consumer": "DESKTOP_CONSUMER",
        "desktop": "DESKTOP_CONSUMER",
        "mobile": "MOBILE",
        "workstation": "WORKSTATION",
        "datacenter": "DATACENTER",
        "integrated": "INTEGRATED",
        "embedded": "EMBEDDED",
        "console": "CONSOLE",
        "legacy": "LEGACY",
    }
    return aliases.get(str(value).strip().casefold(), "UNKNOWN")


def _canonical_tier(value: object) -> str:
    aliases = {
        "enthusiast": "ENTHUSIAST",
        "high": "HIGH",
        "mid": "MID",
        "entry": "ENTRY",
        "integrated": "INTEGRATED",
    }
    return aliases.get(str(value).strip().casefold(), "UNKNOWN")


def download_gpu_dataset(destination: str | Path, force: bool = False) -> Path:
    """Download the public Kaggle archive once and return its local path."""

    destination = Path(destination)
    if destination.exists() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = httpx.get(KAGGLE_GPU_URL, timeout=60.0, follow_redirects=True)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def read_gpu_specs(path: str | Path) -> pd.DataFrame:
    """Read a direct CSV/Parquet file or the first CSV in a Kaggle archive."""

    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source, dtype=str)
    with zipfile.ZipFile(source) as archive:
        names = sorted(name for name in archive.namelist() if name.lower().endswith(".csv"))
        if not names:
            raise ValueError("GPU archive contains no CSV file")
        return pd.read_csv(BytesIO(archive.read(names[0])), dtype=str)


def _normalized_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _column(raw: pd.DataFrame, *names: str) -> pd.Series | None:
    """Return the first matching source column using punctuation-insensitive names."""

    normalized: dict[str, object] = {}
    for column in raw.columns:
        normalized.setdefault(_normalized_name(column), column)
    for name in names:
        match = normalized.get(_normalized_name(name))
        if match is not None:
            return raw[match]
    return None


def _empty_text(index: pd.Index) -> pd.Series:
    return pd.Series(pd.NA, index=index, dtype="string")


def _text(values: pd.Series | None, index: pd.Index) -> pd.Series:
    if values is None:
        return _empty_text(index)
    return values.reindex(index).astype("string").str.strip()


def _meaningful(values: pd.Series) -> pd.Series:
    """Return source text with known sentinels treated as missing for parsing."""

    result = values.astype("string").str.strip()
    lowered = result.str.casefold()
    return result.mask(lowered.isin(_MISSING_TEXT))


def _coalesce_text(index: pd.Index, *values: pd.Series | None) -> pd.Series:
    result = _empty_text(index)
    for value in values:
        candidate = _text(value, index)
        missing = result.isna() | result.eq("").fillna(False)
        result = result.mask(missing, candidate)
    return result


def _numeric(values: pd.Series | None) -> pd.Series:
    """Extract the first numeric token, retaining a nullable float result."""

    if values is None:
        return pd.Series(dtype="float64")
    text = _meaningful(values)
    number = text.str.extract(_NUMBER_PATTERN, expand=False)
    return pd.to_numeric(number, errors="coerce")


def _unit_numeric(
    values: pd.Series | None,
    unit_factors: dict[str, float],
    assumed_unit: str,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Parse a number with an explicit unit and return value, unit, status."""

    if values is None:
        empty = pd.Series(dtype="float64")
        return empty, pd.Series(dtype="string"), pd.Series(dtype="string")
    text = _meaningful(values)
    number = pd.to_numeric(text.str.extract(_NUMBER_PATTERN, expand=False), errors="coerce")
    unit = text.str.extract(_UNIT_PATTERN, expand=False).astype("string").str.lower()
    factors = unit.map(unit_factors).astype("float64")
    has_number = number.notna()
    explicit_unit = factors.notna()
    factors = factors.fillna(1.0)
    parsed = (number * factors).astype("Float64")
    parsed = parsed.where(has_number)
    unit_label = unit.where(explicit_unit, assumed_unit)
    unit_label = unit_label.where(has_number, "unknown").astype("string")
    status = pd.Series("REVIEW", index=text.index, dtype="string")
    status = status.mask(has_number & explicit_unit, "PASS")
    return parsed, unit_label, status


def _memory_size(values: pd.Series | None) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Parse memory size into MB, including optional ``xN`` module counts."""

    parsed, unit, status = _unit_numeric(
        values,
        {
            "kib": 1 / 1024,
            "kb": 1 / 1024,
            "mib": 1,
            "mb": 1,
            "gib": 1024,
            "gb": 1024,
            "tib": 1024 * 1024,
            "tb": 1024 * 1024,
        },
        "MB_assumed",
    )
    if values is None or parsed.empty:
        return parsed, unit, status
    text = _meaningful(values)
    multiplier = pd.to_numeric(text.str.extract(r"(?i)x\s*(\d+)", expand=False), errors="coerce").fillna(1)
    parsed = (parsed * multiplier).astype("Float64")
    return parsed, unit, status


def _parse_date(values: pd.Series) -> pd.Series:
    """Parse Kaggle date strings without discarding ordinal suffixes or precision."""

    cleaned = _meaningful(values)
    cleaned = cleaned.str.replace(r"(\d+)(st|nd|rd|th)", r"\1", regex=True, flags=re.IGNORECASE)
    return pd.to_datetime(cleaned, errors="coerce", format="mixed")


def _date_precision(values: pd.Series) -> pd.Series:
    text = _meaningful(values)
    day = (
        text.str.match(rf"(?i)^\s*{_MONTH_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s*|\s+)\d{{4}}\s*$")
        | text.str.match(r"^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}\s*$")
    ).fillna(False)
    month = (
        text.str.match(rf"(?i)^\s*{_MONTH_PATTERN}(?:\s+|[-/])\d{{4}}\s*$")
        | text.str.match(r"^\s*\d{4}[-/]\d{1,2}\s*$")
    ).fillna(False)
    year = text.str.match(r"^\s*\d{4}\s*$").fillna(False)
    result = pd.Series("unknown", index=text.index, dtype="string")
    result = result.mask(year, "year")
    result = result.mask(month, "month")
    result = result.mask(day, "day")
    result = result.mask(text.isna(), "missing")
    return result


def _classify_vendor(value: object) -> tuple[str, str, str]:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    lowered = text.casefold()
    mappings = (
        (("nvidia",), "NVIDIA", "NVIDIA"),
        (("advanced micro devices", "amd"), "AMD", "AMD"),
        (("ati",), "ATI", "AMD"),
        (("intel",), "Intel", "Intel"),
        (("matrox",), "Matrox", "Matrox"),
        (("3dfx",), "3dfx", "3dfx"),
        (("xgi",), "XGI", "XGI"),
        (("sis",), "SiS", "SiS"),
        (("sony",), "Sony", "Sony"),
        (("microsoft",), "Microsoft", "Microsoft"),
        (("s3",), "S3", "S3"),
        (("via",), "VIA", "VIA"),
    )
    for tokens, canonical, family in mappings:
        if any(token in lowered for token in tokens):
            return canonical, family, f"matched source brand '{text}'"
    if not text or lowered in _MISSING_TEXT:
        return "Unknown", "Unknown", "source brand is missing or a sentinel"
    return "Unknown", "Unknown", f"unmapped source brand '{text}'"


def _known_generation(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.casefold() in _MISSING_TEXT:
        return None
    return text


def _generation_heuristic(name: object, architecture: object, vendor: object) -> tuple[str | None, str]:
    name_text = "" if name is None or pd.isna(name) else str(name).strip()
    architecture_text = "" if architecture is None or pd.isna(architecture) else str(architecture).strip()
    lowered = f"{name_text} {architecture_text}".casefold()
    architecture_labels = (
        "Blackwell", "Hopper", "Ada Lovelace", "Ampere", "Turing", "Volta", "Pascal",
        "Maxwell", "Kepler", "Fermi", "Tesla", "RDNA 4", "RDNA 3", "RDNA 2", "RDNA",
        "Vega", "Navi", "Polaris", "GCN", "TeraScale", "Arc", "Xe",
    )
    for label in architecture_labels:
        if label.casefold() in lowered:
            return label, f"inferred from architecture/name cue '{label}'"
    vendor_text = "" if vendor is None or pd.isna(vendor) else str(vendor).casefold()
    if "nvidia" in vendor_text:
        match = re.search(r"\b(?:geforce|rtx|gtx|quadro)\s*(\d{2,4})\b", lowered)
        if match:
            number = match.group(1)
            if len(number) == 4:
                return f"GeForce {number[:2]}00", f"inferred from NVIDIA series token '{number}'"
            return f"GeForce {number}", f"inferred from NVIDIA series token '{number}'"
    if "amd" in vendor_text or "ati" in vendor_text:
        match = re.search(r"\b(?:rx|r|radeon)\s*(\d{3,4})\b", lowered)
        if match:
            number = match.group(1)
            return f"Radeon {number[:2]}00", f"inferred from AMD/ATI series token '{number}'"
    if name_text:
        return name_text, "fallback to product name because no generation cue was available"
    return None, "no generation value or inferable cue"


def _generation_classification(
    name: object,
    architecture: object,
    vendor: object,
    generation_values: list[tuple[str, object]],
) -> tuple[str, str, str, str, bool, str]:
    meaningful = [(source, value) for source, value in generation_values if _known_generation(value) is not None]
    unique: list[str] = []
    for _, value in meaningful:
        text = _known_generation(value)
        assert text is not None
        if text.casefold() not in {candidate.casefold() for candidate in unique}:
            unique.append(text)
    conflict = len(unique) > 1
    if meaningful:
        source, value = meaningful[0]
        generation = _known_generation(value)
        assert generation is not None
        status = "REVIEW" if conflict else "PASS"
        reason = f"used source generation field '{source}'"
        if conflict:
            reason += f"; conflicting values: {', '.join(unique)}"
        return generation, source, status, reason, conflict, " | ".join(unique)
    inferred, reason = _generation_heuristic(name, architecture, vendor)
    if inferred is None:
        return "Unknown", "unknown", "REVIEW", reason, False, ""
    return inferred, "heuristic", "REVIEW", reason, False, inferred


def _direct_segment(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.casefold() in _MISSING_TEXT:
        return None
    lowered = text.casefold().replace("-", " ").replace("_", " ")
    direct = {
        "consumer": "consumer", "desktop": "consumer", "discrete": "consumer",
        "mobile": "mobile", "laptop": "mobile", "notebook": "mobile",
        "workstation": "workstation", "professional": "workstation",
        "server": "datacenter", "datacenter": "datacenter", "data center": "datacenter",
        "console": "console", "embedded": "embedded", "integrated": "integrated", "igp": "integrated",
    }
    return direct.get(lowered)


def _market_segment_classification(
    name: object,
    vendor: object,
    generation: object,
    architecture: object,
    direct_value: object,
    memory_type: object,
) -> tuple[str, str, str, str]:
    direct = _direct_segment(direct_value)
    name_text = "" if name is None or pd.isna(name) else str(name).strip()
    vendor_text = "" if vendor is None or pd.isna(vendor) else str(vendor).strip()
    generation_text = "" if generation is None or pd.isna(generation) else str(generation).strip()
    architecture_text = "" if architecture is None or pd.isna(architecture) else str(architecture).strip()
    memory_text = "" if memory_type is None or pd.isna(memory_type) else str(memory_type).strip()
    combined = f"{name_text} {vendor_text} {generation_text} {architecture_text} {memory_text}".casefold()
    cues: list[str] = []
    patterns = (
        ("console", r"\b(console|playstation|xbox|nintendo)\b|sony|microsoft"),
        ("datacenter", r"\b(server|datacenter|data center|tesla|grid|instinct)\b|\b(?:a100|h100|v100|p100|t4)\b|\bmi\d"),
        ("workstation", r"\b(quadro|firepro|firegl|workstation|rtx pro|radeon pro|pro w|wx\d)\b"),
        ("embedded", r"\bembedded\b|\bindustrial\b"),
        ("integrated", r"\b(igp|integrated|system shared|uhd graphics|iris|hd graphics|gma)\b"),
        ("mobile", r"\b(mobile|mobility|laptop|notebook|max[\s-]?q|go)\b"),
        ("consumer", r"\b(geforce|radeon|r(?:tx|x)|gtx|arc)\b"),
    )
    for segment, pattern in patterns:
        if re.search(pattern, combined, flags=re.IGNORECASE):
            cues.append(segment)
    if direct is not None:
        if cues and direct not in cues:
            return direct, "REVIEW", f"used direct market segment '{direct}' despite cue(s): {', '.join(cues)}", "|".join(cues)
        return direct, "PASS", f"used direct market segment '{direct}'", "|".join(cues or [direct])
    unique = list(dict.fromkeys(cues))
    if not unique:
        return "Unknown", "REVIEW", "no direct segment or recognized name/generation cue", ""
    if len(unique) > 1:
        priority = ("console", "datacenter", "workstation", "embedded", "integrated", "mobile", "consumer")
        selected = next(segment for segment in priority if segment in unique)
        return selected, "REVIEW", f"overlapping segment cues: {', '.join(unique)}", "|".join(unique)
    return unique[0], "PASS", f"inferred from cue '{unique[0]}'", unique[0]


def _performance_classification(
    name: object,
    generation: object,
    market_segment: object,
    tdp_w: object,
    memory_size_mb: object,
    direct_value: object,
) -> tuple[str, float | None, str, str]:
    if direct_value is not None and not pd.isna(direct_value):
        text = str(direct_value).strip().casefold()
        aliases = {"flagship": "enthusiast", "enthusiast": "enthusiast", "high": "high", "upper": "high", "mid": "mid", "middle": "mid", "entry": "entry", "low": "entry", "integrated": "integrated"}
        if text in aliases:
            tier = aliases[text]
            return tier, {"integrated": 0.0, "entry": 1.0, "mid": 2.0, "high": 3.0, "enthusiast": 4.0}[tier], "PASS", f"used direct performance tier '{tier}'"
    name_text = "" if name is None or pd.isna(name) else str(name).strip()
    generation_text = "" if generation is None or pd.isna(generation) else str(generation).strip()
    segment_text = "" if market_segment is None or pd.isna(market_segment) else str(market_segment).strip().casefold()
    combined = f"{name_text} {generation_text}".casefold()
    if segment_text == "integrated" or re.search(r"\b(igp|integrated|uhd graphics|iris|hd graphics|gma)\b", combined):
        return "integrated", 0.0, "PASS", "integrated segment/name cue"
    if re.search(r"\b(flagship|titan|xtx|rtx\s*(?:5090|4090|3090)|(?:rx\s*)?79(?:00|50))\b", combined):
        return "enthusiast", 4.0, "PASS", "flagship model-family cue"
    if re.search(r"\b(?:rtx|gtx|geforce)\s*(?:5080|4080|3080|2080|1080|980|780|680)\b", combined):
        return "high", 3.0, "PASS", "high-tier NVIDIA model cue"
    if re.search(r"\b(?:rx\s*)?(?:7800|6800|7700|6700|5700|5700\s*xt|a7[567]0)\b", combined):
        return "high", 3.0, "PASS", "high-tier AMD/Intel model cue"
    if re.search(r"\b(?:rtx|gtx|geforce)\s*(?:5070|4070|3070|2070|1070|970|770|670)\b", combined):
        return "mid", 2.0, "PASS", "mid-tier NVIDIA model cue"
    if re.search(r"\b(?:rx\s*)?(?:7600|6600|5600|5500|6500|6400|a[357]50|a380)\b", combined):
        return "mid", 2.0, "PASS", "mid-tier AMD/Intel model cue"
    if re.search(r"\b(?:rtx|gtx|geforce)\s*(?:5060|4060|3060|2060|1660|1650|1060|960|760|660|560|460|360|260)\b", combined):
        return "entry", 1.0, "PASS", "entry-tier NVIDIA model cue"
    if re.search(r"\b(?:rx\s*)?(?:6500|6400|5500|5400|5300|560|550|540|530|460|340|240)\b", combined):
        return "entry", 1.0, "PASS", "entry-tier AMD model cue"
    tdp = pd.to_numeric(pd.Series([tdp_w]), errors="coerce").iloc[0]
    memory = pd.to_numeric(pd.Series([memory_size_mb]), errors="coerce").iloc[0]
    if (pd.notna(tdp) and float(tdp) >= 300) or (pd.notna(memory) and float(memory) >= 16384):
        return "high", 3.0, "REVIEW", "numeric high-capability proxy without a recognized model tier"
    if (pd.notna(tdp) and float(tdp) >= 200) or (pd.notna(memory) and float(memory) >= 8192):
        return "mid", 2.0, "REVIEW", "numeric capability proxy without a recognized model tier"
    return "Unknown", None, "REVIEW", "no direct tier or recognized model/numeric cue"


def _launch_importance(tier: object, generation_source: object, tier_score: object) -> tuple[str, float | None, str, str]:
    value = "" if tier is None or pd.isna(tier) else str(tier)
    score = pd.to_numeric(pd.Series([tier_score]), errors="coerce").iloc[0]
    if value == "enthusiast":
        return "high", 3.0, "PASS", "enthusiast/flagship performance tier"
    if value == "high":
        return "high", 2.0, "REVIEW" if generation_source == "heuristic" else "PASS", "high performance tier"
    if value == "mid":
        return "medium", 1.0, "REVIEW" if generation_source == "heuristic" else "PASS", "mid performance tier"
    if value in {"entry", "integrated"}:
        return "low", 0.0, "PASS", f"{value} performance tier"
    if pd.notna(score):
        return "medium", 1.0, "REVIEW", "numeric tier score without a named tier"
    return "Unknown", None, "REVIEW", "launch importance cannot be inferred from available fields"


def _status_rank(value: object) -> int:
    return {"PASS": 0, "REVIEW": 1, "FAIL": 2}.get(str(value), 1)


def normalize_gpu_specs(
    raw: pd.DataFrame,
    source_id: str = "KAGGLE_GPU_SPECS",
    *,
    config_dir: str | Path = "config",
) -> pd.DataFrame:
    """Normalize the observed Kaggle export while retaining uncertain rows.

    The export has separate Graphics Card, Mobile Graphics, and Integrated
    Graphics release/generation fields.  Release dates are coalesced per row
    in that order, with every source-native field retained.  Numeric memory is
    canonicalized to MB and TDP to watts.  Rows with missing dates, metrics, or
    heuristic classifications remain in the output and carry ``REVIEW`` status
    rather than being dropped.
    """

    generation_config = _gpu_config(config_dir, "gpu_generations.yaml")
    override_config = _gpu_config(config_dir, "gpu_mapping_overrides.yaml")
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("GPU normalization requires a pandas DataFrame")
    if raw.empty:
        raise ValueError("GPU dataset is empty")
    source_frame = raw.reset_index(drop=True)
    index = source_frame.index
    brand_source = _column(source_frame, "Brand", "vendor", "manufacturer")
    name_source = _column(source_frame, "Name", "product_name", "GPU Name", "gpu_name")
    if name_source is None:
        raise ValueError("GPU dataset missing identity field: Name/product_name")
    release_columns = (
        ("graphics_card", _column(source_frame, "Graphics Card__Release Date", "graphics_release_date")),
        ("mobile", _column(source_frame, "Mobile Graphics__Release Date", "mobile_release_date")),
        ("integrated", _column(source_frame, "Integrated Graphics__Release Date", "integrated_release_date")),
        ("generic", _column(source_frame, "release_date", "Release Date")),
    )
    if all(values is None for _, values in release_columns):
        raise ValueError("GPU dataset missing release-date fields")

    brand_raw = _text(brand_source, index)
    product_name = _meaningful(_text(name_source, index))
    graphics_release_raw = _text(release_columns[0][1], index)
    mobile_release_raw = _text(release_columns[1][1], index)
    integrated_release_raw = _text(release_columns[2][1], index)
    generic_release_raw = _text(release_columns[3][1], index)
    release_raw = _coalesce_text(index, graphics_release_raw, mobile_release_raw, integrated_release_raw, generic_release_raw)
    release_date = _parse_date(release_raw)
    release_source = pd.Series("missing", index=index, dtype="string")
    for source_name, values in release_columns:
        candidate = _text(values, index)
        meaningful = _meaningful(candidate).notna()
        release_source = release_source.mask((release_source == "missing") & meaningful, source_name)
    release_precision = _date_precision(release_raw)
    release_status = pd.Series("REVIEW", index=index, dtype="string")
    release_status = release_status.mask(release_raw.notna() & release_date.notna(), "PASS")

    graphics_generation = _text(_column(source_frame, "Graphics Card__Generation", "graphics_generation"), index)
    mobile_generation = _text(_column(source_frame, "Mobile Graphics__Generation", "mobile_generation"), index)
    integrated_generation = _text(_column(source_frame, "Integrated Graphics__Generation", "integrated_generation"), index)
    generic_generation = _text(_column(source_frame, "generation", "Generation"), index)
    processor_generation = _text(_column(source_frame, "Graphics Processor__Generation"), index)
    architecture = _meaningful(_text(_column(source_frame, "Graphics Processor__Architecture", "Architecture"), index))
    memory_raw = _coalesce_text(index, _column(source_frame, "Memory__Memory Size"), _column(source_frame, "Top__MEMORY SIZE"), _column(source_frame, "Graphics Card__Memory Size"), _column(source_frame, "Memory Size", "vram", "VRAM"))
    memory_size_mb, memory_unit, memory_parse_status = _memory_size(memory_raw)
    tdp_raw = _coalesce_text(index, _column(source_frame, "Board Design__TDP"), _column(source_frame, "Graphics Card__TDP"), _column(source_frame, "TDP"))
    tdp_w, tdp_unit, tdp_parse_status = _unit_numeric(tdp_raw, {"w": 1.0, "mw": 0.001}, "W_assumed")
    memory_type = _meaningful(_coalesce_text(index, _column(source_frame, "Memory__Memory Type"), _column(source_frame, "Top__MEMORY TYPE"), _column(source_frame, "Graphics Card__Memory Type"), _column(source_frame, "Memory Type")))
    memory_bus_raw = _coalesce_text(index, _column(source_frame, "Memory__Memory Bus"), _column(source_frame, "Top__BUS WIDTH"), _column(source_frame, "Memory Bus"))
    memory_bus = _numeric(memory_bus_raw)
    process_nm = _numeric(_coalesce_text(index, _column(source_frame, "Graphics Processor__Process Size"), _column(source_frame, "Process Size")))
    market_raw = _coalesce_text(index, _column(source_frame, "market_segment", "Market Segment"))
    direct_tier = _coalesce_text(index, _column(source_frame, "performance_tier", "Performance Tier"))

    vendor_values: list[str] = []
    vendor_families: list[str] = []
    vendor_statuses: list[str] = []
    vendor_reasons: list[str] = []
    for value in brand_raw:
        vendor, family, reason = _classify_vendor(value)
        vendor_values.append(vendor)
        vendor_families.append(family)
        vendor_statuses.append("PASS" if vendor != "Unknown" else "REVIEW")
        vendor_reasons.append(reason)
    generations: list[str] = []
    generation_sources: list[str] = []
    generation_statuses: list[str] = []
    generation_reasons: list[str] = []
    generation_conflicts: list[bool] = []
    generation_candidates: list[str] = []
    market_segments: list[str] = []
    market_statuses: list[str] = []
    market_reasons: list[str] = []
    market_candidates: list[str] = []
    market_segment_categories: list[str] = []
    performance_tier_canonicals: list[str] = []
    override_reasons: list[str | None] = []
    performance_tiers: list[str] = []
    performance_scores: list[float | None] = []
    performance_statuses: list[str] = []
    performance_reasons: list[str] = []
    importance_values: list[str] = []
    importance_scores: list[float | None] = []
    importance_statuses: list[str] = []
    importance_reasons: list[str] = []
    high_power: list[bool | None] = []
    high_power_status: list[str] = []
    high_vram: list[bool | None] = []
    high_vram_status: list[str] = []
    for row_number in range(len(source_frame)):
        generation, generation_source, generation_status, generation_reason, conflict, candidates = _generation_classification(
            product_name.iloc[row_number],
            architecture.iloc[row_number],
            vendor_values[row_number],
            [
                ("graphics_card", graphics_generation.iloc[row_number]),
                ("mobile", mobile_generation.iloc[row_number]),
                ("integrated", integrated_generation.iloc[row_number]),
                ("generic", generic_generation.iloc[row_number]),
            ],
        )
        canonical_generation, mapping_source, mapping_status = _canonical_generation(
            product_name.iloc[row_number],
            vendor_values[row_number],
            architecture.iloc[row_number],
            generation,
            config=generation_config,
        )
        segment, segment_status, segment_reason, segment_candidate = _market_segment_classification(
            product_name.iloc[row_number],
            vendor_values[row_number],
            canonical_generation,
            architecture.iloc[row_number],
            market_raw.iloc[row_number],
            memory_type.iloc[row_number],
        )
        tier, score, tier_status, tier_reason = _performance_classification(
            product_name.iloc[row_number],
            canonical_generation,
            segment,
            tdp_w.iloc[row_number],
            memory_size_mb.iloc[row_number],
            direct_tier.iloc[row_number],
        )
        override_row, override_reason = _apply_gpu_override(
            {
                "vendor": vendor_values[row_number],
                "product_name": product_name.iloc[row_number],
                "generation": canonical_generation,
                "market_segment": _canonical_segment(segment),
                "performance_tier": _canonical_tier(tier),
            },
            config=override_config,
        )
        if override_reason is not None:
            canonical_generation = str(override_row["generation"])
            segment = str(override_row["market_segment"])
            tier = str(override_row["performance_tier"]).casefold()
            score = {"enthusiast": 4.0, "high": 3.0, "mid": 2.0, "entry": 1.0, "integrated": 0.0}.get(tier)
            generation_status = segment_status = tier_status = "REVIEW"
            generation_reason = segment_reason = tier_reason = override_reason
            mapping_source = "manual_override"
            mapping_status = "REVIEW"
        importance, importance_score, importance_status, importance_reason = _launch_importance(tier, generation_source, score)
        tdp_value = pd.to_numeric(pd.Series([tdp_w.iloc[row_number]]), errors="coerce").iloc[0]
        memory_value = pd.to_numeric(pd.Series([memory_size_mb.iloc[row_number]]), errors="coerce").iloc[0]
        high_power.append(None if pd.isna(tdp_value) else bool(float(tdp_value) >= 200))
        high_power_status.append("PASS" if pd.notna(tdp_value) else "REVIEW")
        high_vram.append(None if pd.isna(memory_value) else bool(float(memory_value) >= 8192))
        high_vram_status.append("PASS" if pd.notna(memory_value) else "REVIEW")
        generations.append(canonical_generation)
        generation_sources.append(mapping_source)
        generation_statuses.append("REVIEW" if mapping_status == "REVIEW" or conflict else generation_status)
        generation_reasons.append(generation_reason if mapping_status == "PASS" else f"{generation_reason}; mapping={mapping_source}")
        generation_conflicts.append(conflict)
        generation_candidates.append(candidates)
        market_segments.append(segment)
        market_segment_categories.append(_canonical_segment(segment))
        market_statuses.append(segment_status)
        market_reasons.append(segment_reason)
        market_candidates.append(segment_candidate)
        performance_tiers.append(tier)
        performance_tier_canonicals.append(_canonical_tier(tier))
        performance_scores.append(score)
        performance_statuses.append(tier_status)
        performance_reasons.append(tier_reason)
        importance_values.append(importance)
        importance_scores.append(importance_score)
        importance_statuses.append(importance_status)
        importance_reasons.append(importance_reason)
        override_reasons.append(override_reason)

    vendor_series = pd.Series(vendor_values, index=index, dtype="string")
    product_for_key = product_name.fillna("__MISSING_PRODUCT__").str.casefold()
    date_for_key = release_date.dt.strftime("%Y-%m-%d").fillna("__MISSING_DATE__")
    duplicate_key = vendor_series.str.casefold() + "|" + product_for_key + "|" + date_for_key
    duplicate_count = duplicate_key.groupby(duplicate_key, sort=False).transform("size").astype("Int64")
    source_duplicate = source_frame.duplicated(keep=False).astype("boolean")
    source_record_ids = pd.Series(
        [
            hashlib.sha256(
                f"{source_id}|{row_number}|{repr(tuple(source_frame.iloc[row_number].tolist()))}".encode("utf-8")
            ).hexdigest()
            for row_number in range(len(source_frame))
        ],
        index=index,
        dtype="string",
    )
    classification_statuses: list[str] = []
    classification_reasons: list[str] = []
    for row_number in range(len(source_frame)):
        component_statuses = [release_status.iloc[row_number], vendor_statuses[row_number], generation_statuses[row_number], market_statuses[row_number], performance_statuses[row_number], importance_statuses[row_number], high_power_status[row_number], high_vram_status[row_number]]
        status = "FAIL" if pd.isna(product_name.iloc[row_number]) else ("REVIEW" if any(_status_rank(item) > 0 for item in component_statuses) else "PASS")
        reasons: list[str] = []
        if status != "PASS":
            for name, component in (("release_date", release_status.iloc[row_number]), ("vendor", vendor_statuses[row_number]), ("generation", generation_statuses[row_number]), ("market_segment", market_statuses[row_number]), ("performance_tier", performance_statuses[row_number]), ("launch_importance", importance_statuses[row_number]), ("high_power", high_power_status[row_number]), ("high_vram", high_vram_status[row_number])):
                if _status_rank(component) > 0:
                    reasons.append(f"{name}:{component}")
            if pd.isna(product_name.iloc[row_number]):
                reasons.append("product_name:FAIL")
        classification_statuses.append(status); classification_reasons.append("; ".join(reasons))

    result = pd.DataFrame({
        "market_segment_raw": market_raw, "market_segment": pd.Series(market_segments, index=index, dtype="string"), "market_segment_category": pd.Series(market_segment_categories, index=index, dtype="string"), "market_segment_candidates": pd.Series(market_candidates, index=index, dtype="string"), "market_segment_status": pd.Series(market_statuses, index=index, dtype="string"), "market_segment_reason": pd.Series(market_reasons, index=index, dtype="string"),
        "source_record_id": source_record_ids, "gpu_id": (vendor_series + ":" + product_name.fillna("__MISSING_PRODUCT__")).astype("string"),
        "vendor_raw": brand_raw, "vendor": vendor_series, "vendor_family": pd.Series(vendor_families, index=index, dtype="string"),
        "vendor_classification_status": pd.Series(vendor_statuses, index=index, dtype="string"), "vendor_classification_reason": pd.Series(vendor_reasons, index=index, dtype="string"),
        "product_name": product_name, "graphics_release_date_raw": graphics_release_raw, "mobile_release_date_raw": mobile_release_raw, "integrated_release_date_raw": integrated_release_raw, "release_date_raw": release_raw, "release_date": release_date, "release_date_source": release_source, "release_date_precision": release_precision, "release_date_status": release_status,
        "architecture": architecture, "memory_size_raw": memory_raw, "memory_size_mb": memory_size_mb, "memory_size_gb": (memory_size_mb / 1024).astype("Float64"), "memory_size": memory_size_mb, "memory_unit": memory_unit, "memory_parse_status": memory_parse_status, "memory_type": memory_type, "memory_bus_raw": memory_bus_raw, "memory_bus": memory_bus, "memory_bus_bits": memory_bus, "tdp_raw": tdp_raw, "tdp_w": tdp_w, "tdp_unit": tdp_unit, "tdp_parse_status": tdp_parse_status, "process_nm": process_nm,
        "graphics_generation_raw": graphics_generation, "mobile_generation_raw": mobile_generation, "integrated_generation_raw": integrated_generation, "processor_generation_raw": processor_generation, "generation_raw": _coalesce_text(index, graphics_generation, mobile_generation, integrated_generation, generic_generation), "generation": pd.Series(generations, index=index, dtype="string"), "generation_source": pd.Series(generation_sources, index=index, dtype="string"), "generation_candidates": pd.Series(generation_candidates, index=index, dtype="string"), "generation_status": pd.Series(generation_statuses, index=index, dtype="string"), "generation_reason": pd.Series(generation_reasons, index=index, dtype="string"), "generation_conflict_flag": pd.Series(generation_conflicts, index=index, dtype="boolean"),
        "performance_tier": pd.Series(performance_tiers, index=index, dtype="string"), "performance_tier_canonical": pd.Series(performance_tier_canonicals, index=index, dtype="string"), "performance_tier_score": pd.Series(performance_scores, index=index, dtype="Float64"), "performance_tier_status": pd.Series(performance_statuses, index=index, dtype="string"), "performance_tier_reason": pd.Series(performance_reasons, index=index, dtype="string"), "launch_importance": pd.Series(importance_values, index=index, dtype="string"), "launch_importance_score": pd.Series(importance_scores, index=index, dtype="Float64"), "launch_importance_status": pd.Series(importance_statuses, index=index, dtype="string"), "launch_importance_reason": pd.Series(importance_reasons, index=index, dtype="string"),
        "high_power_flag": pd.Series(high_power, index=index, dtype="boolean"), "is_high_power": pd.Series(high_power, index=index, dtype="boolean"), "high_power_status": pd.Series(high_power_status, index=index, dtype="string"), "high_vram_flag": pd.Series(high_vram, index=index, dtype="boolean"), "is_high_vram": pd.Series(high_vram, index=index, dtype="boolean"), "high_vram_status": pd.Series(high_vram_status, index=index, dtype="string"),
        "duplicate_product_key": duplicate_key.astype("string"), "duplicate_product_count": duplicate_count, "duplicate_product_flag": (duplicate_count > 1).astype("boolean"), "exact_duplicate_flag": source_duplicate, "override_reason": pd.Series(override_reasons, index=index, dtype="string"), "event_type": "GPU_LAUNCH", "event_status": pd.Series(["AUTO_DERIVED" if status == "PASS" and precision == "day" else "REVIEW" for status, precision in zip(release_status, release_precision)], index=index, dtype="string"), "classification_status": pd.Series(classification_statuses, index=index, dtype="string"), "classification_reasons": pd.Series(classification_reasons, index=index, dtype="string"), "source": source_id,
    })
    return result


def _empty_gpu_weekly_signals() -> pd.DataFrame:
    return pd.DataFrame(columns=["week", "gpu_launch_count", "gpu_record_count", "gpu_duplicate_record_count", "nvidia_gpu_launch_count", "amd_gpu_launch_count", "enthusiast_gpu_launch_count", "high_power_gpu_launch_count", "high_vram_gpu_launch_count", "avg_new_gpu_tdp", "min_new_gpu_tdp", "max_new_gpu_tdp", "avg_new_gpu_vram", "min_new_gpu_vram", "max_new_gpu_vram", "major_gpu_launch_flag", "gpu_upgrade_intensity", "gpu_power_requirement_index"])


def _vendor_column_name(vendor: object) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(vendor).casefold()).strip("_")
    return f"{slug or 'unknown'}_gpu_launch_count"

def filter_gpu_products(
    products: pd.DataFrame,
    *,
    config_path: str | Path = "config/gpu_signal.yaml",
) -> pd.DataFrame:
    """Apply configured market-segment scope while retaining excluded rows upstream."""
    config = load_yaml(config_path) if Path(config_path).exists() else {}
    settings = config.get("filter", {}) if isinstance(config, Mapping) else {}
    if "market_segment_category" not in products.columns and "market_segment" not in products.columns:
        return products.copy().reset_index(drop=True)
    segments = products.get("market_segment_category", products.get("market_segment"))
    allowed = {str(value).upper() for value in settings.get("allowed_market_segments", [])}
    include_unknown = bool(settings.get("include_unknown", False))
    mask = segments.astype("string").str.upper().isin(allowed)
    if include_unknown:
        mask |= segments.astype("string").str.upper().eq("UNKNOWN")
    return products.loc[mask].copy().reset_index(drop=True)
def derive_gpu_launch_events(
    products: pd.DataFrame,
    *,
    config_path: str | Path = "config/gpu_signal.yaml",
) -> pd.DataFrame:
    """Create canonical GPU launch events from observed dated product records."""
    if products.empty:
        return pd.DataFrame(
            columns=[
                "event_id", "event_date", "week", "vendor", "product_name",
                "generation", "architecture", "market_segment", "performance_tier",
                "memory_size_gb", "tdp_w", "event_type", "event_status",
                "source_id", "source_record_id",
            ]
        )
    scoped = filter_gpu_products(products, config_path=config_path)
    dates = pd.to_datetime(scoped.get("release_date"), errors="coerce")
    names = scoped.get("product_name", pd.Series(pd.NA, index=scoped.index)).astype("string")
    valid = dates.notna() & names.notna() & names.str.strip().ne("")
    scoped = scoped.loc[valid].copy()
    dates = dates.loc[valid]
    if scoped.empty:
        return derive_gpu_launch_events(pd.DataFrame(), config_path=config_path)
    record_ids = scoped.get(
        "source_record_id",
        pd.Series([f"row-{index}" for index in scoped.index], index=scoped.index, dtype="string"),
    ).astype("string")
    vendor = scoped.get("vendor", pd.Series("Unknown", index=scoped.index)).astype("string").fillna("Unknown")
    names = scoped["product_name"].astype("string").str.strip()
    event_ids = [
        hashlib.sha256(f"{vendor.iloc[i]}|{names.iloc[i]}|{dates.iloc[i].date()}".encode("utf-8")).hexdigest()[:20]
        for i in range(len(scoped))
    ]
    result = pd.DataFrame(
        {
            "event_id": event_ids,
            "event_date": dates.dt.normalize().to_numpy(),
            "week": (dates + pd.to_timedelta((5 - dates.dt.dayofweek) % 7, unit="D")).dt.normalize().to_numpy(),
            "vendor": vendor.to_numpy(),
            "product_name": names.to_numpy(),
            "generation": scoped.get("generation", "UNKNOWN").astype("string").fillna("UNKNOWN").to_numpy(),
            "architecture": scoped.get("architecture", pd.NA).to_numpy(),
            "market_segment": scoped.get("market_segment_category", "UNKNOWN").astype("string").fillna("UNKNOWN").to_numpy(),
            "performance_tier": scoped.get("performance_tier_canonical", scoped.get("performance_tier", "UNKNOWN")).astype("string").fillna("UNKNOWN").to_numpy(),
            "memory_size_gb": pd.to_numeric(scoped.get("memory_size_gb"), errors="coerce").to_numpy(),
            "tdp_w": pd.to_numeric(scoped.get("tdp_w"), errors="coerce").to_numpy(),
            "event_type": "GPU_LAUNCH",
            "event_status": [
                "AUTO_DERIVED" if str(precision).casefold() == "day" else "REVIEW"
                for precision in scoped.get("release_date_precision", pd.Series("unknown", index=scoped.index))
            ],
            "source_id": scoped.get("source", "KAGGLE_GPU_SPECS").astype("string").to_numpy(),
            "source_record_id": record_ids.to_numpy(),
        }
    )
    return result.sort_values(["event_date", "vendor", "product_name"]).reset_index(drop=True)




def derive_gpu_weekly_signals(products: pd.DataFrame) -> pd.DataFrame:
    """Create deduplicated, transparent weekly launch signals.

    Rows without a valid release date or product name are excluded explicitly
    from weekly aggregation and remain available to validation diagnostics.
    Exact and identity duplicates are not counted twice, while duplicate counts
    are exposed in the weekly output.
    """

    required = {"product_name", "release_date"}
    missing = required.difference(products.columns)
    if missing:
        raise ValueError(f"GPU products missing required columns: {', '.join(sorted(missing))}")
    frame = filter_gpu_products(products)
    frame["__release_date"] = pd.to_datetime(frame["release_date"], errors="coerce")
    frame["__product_name"] = frame["product_name"].astype("string").str.strip()
    valid = frame.loc[frame["__release_date"].notna() & frame["__product_name"].notna() & frame["__product_name"].ne("")].copy()
    if valid.empty:
        return _empty_gpu_weekly_signals()
    valid["week"] = valid["__release_date"] + pd.to_timedelta((5 - valid["__release_date"].dt.dayofweek) % 7, unit="D")
    valid["week"] = valid["week"].dt.normalize()
    valid["__vendor"] = valid.get("vendor", pd.Series("Unknown", index=valid.index)).astype("string").fillna("Unknown")
    valid["__vendor_family"] = valid.get("vendor_family", valid["__vendor"]).astype("string").fillna("Unknown")
    valid["__launch_key"] = valid["__vendor"].str.casefold() + "|" + valid["__product_name"].str.casefold() + "|" + valid["__release_date"].dt.strftime("%Y-%m-%d")
    record_counts = valid.groupby("week", as_index=False).size().rename(columns={"size": "gpu_record_count"})
    unique = valid.sort_values(["week", "__launch_key", "__product_name"], kind="mergesort").drop_duplicates("__launch_key", keep="first").copy()
    unique_counts = unique.groupby("week", as_index=False).size().rename(columns={"size": "gpu_launch_count"})
    tdp_column = ("tdp_w", "mean") if "tdp_w" in unique else ("__release_date", "size")
    tdp_min_column = ("tdp_w", "min") if "tdp_w" in unique else ("__release_date", "size")
    tdp_max_column = ("tdp_w", "max") if "tdp_w" in unique else ("__release_date", "size")
    vram_name = "memory_size_mb" if "memory_size_mb" in unique else "memory_size" if "memory_size" in unique else "__release_date"
    grouped = unique.groupby("week", as_index=False).agg(avg_new_gpu_tdp=tdp_column, min_new_gpu_tdp=tdp_min_column, max_new_gpu_tdp=tdp_max_column, avg_new_gpu_vram=(vram_name, "mean"), min_new_gpu_vram=(vram_name, "min"), max_new_gpu_vram=(vram_name, "max"))
    grouped = unique_counts.merge(record_counts, on="week", how="left").merge(grouped, on="week", how="left")
    grouped["gpu_duplicate_record_count"] = grouped["gpu_record_count"] - grouped["gpu_launch_count"]
    family_lower = unique["__vendor_family"].str.casefold()
    unique["__is_nvidia"] = family_lower.eq("nvidia")
    unique["__is_amd"] = family_lower.eq("amd")
    tier = unique.get("performance_tier", pd.Series("Unknown", index=unique.index)).astype("string").fillna("Unknown").str.casefold()
    unique["__is_enthusiast"] = tier.eq("enthusiast")
    unique["__is_high_power"] = unique.get("high_power_flag", pd.Series(pd.NA, index=unique.index, dtype="boolean")).astype("boolean").fillna(False)
    unique["__is_high_vram"] = unique.get("high_vram_flag", pd.Series(pd.NA, index=unique.index, dtype="boolean")).astype("boolean").fillna(False)
    counters = unique.groupby("week", as_index=False).agg(nvidia_gpu_launch_count=("__is_nvidia", "sum"), amd_gpu_launch_count=("__is_amd", "sum"), enthusiast_gpu_launch_count=("__is_enthusiast", "sum"), high_power_gpu_launch_count=("__is_high_power", "sum"), high_vram_gpu_launch_count=("__is_high_vram", "sum"))
    grouped = grouped.merge(counters, on="week", how="left")
    vendor_counts = unique.groupby(["week", "__vendor"], as_index=False).size()
    for vendor_name in sorted(unique["__vendor"].dropna().unique(), key=lambda value: str(value).casefold()):
        column = _vendor_column_name(vendor_name)
        counts = vendor_counts.loc[vendor_counts["__vendor"] == vendor_name, ["week", "size"]].rename(columns={"size": column})
        grouped = grouped.merge(counts, on="week", how="left")
    scoring_config = load_yaml("config/gpu_signal.yaml").get("scoring", {})
    weights = scoring_config.get("weights", {}) if isinstance(scoring_config, Mapping) else {}
    threshold = float(scoring_config.get("major_launch_threshold", 3)) if isinstance(scoring_config, Mapping) else 3.0
    grouped["major_gpu_launch_flag"] = (grouped["gpu_launch_count"] >= threshold).astype("int64")
    grouped["gpu_upgrade_intensity"] = (
        grouped["gpu_launch_count"] * float(weights.get("launch_count", 1.0))
        + grouped["enthusiast_gpu_launch_count"] * float(weights.get("enthusiast_count", 2.0))
        + grouped["high_power_gpu_launch_count"] * float(weights.get("high_power_count", 1.0))
    )
    grouped["gpu_power_requirement_index"] = grouped["avg_new_gpu_tdp"].replace(0, pd.NA) / 100
    grouped["gpu_power_requirement_index"] = grouped["gpu_power_requirement_index"].fillna(0)
    return grouped.sort_values("week").reset_index(drop=True)
