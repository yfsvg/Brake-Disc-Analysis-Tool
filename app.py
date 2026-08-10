from __future__ import annotations

import csv
import io
import json
import logging
import math
import struct
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from numba import njit

# Set headless backend so matplotlib doesn't expect a GUI event loop
matplotlib.use("Agg")
from matplotlib.figure import Figure
import webview

webview.settings["ALLOW_FILE_URLS"] = True

# Configure logging format and level
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s",
)
logger = logging.getLogger("HeatmapApp")

# Constants
MICRONS_PER_RANGE_UNIT = 0.167569681
MICRONS_PER_MM = 1000.0


# --- Numba JIT Compiled Grid Accumulation ---

@njit
def _accumulate_grid_numba(
    values: np.ndarray,
    angle_index: np.ndarray,
    pixel_index: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
):
    """
    Compiled C-speed loop to replace Python's row-by-row iteration and np.add.at.
    """
    num_rows, num_cols = values.shape
    for r in range(num_rows):
        ai = angle_index[r]
        for c in range(num_cols):
            val = values[r, c]
            if np.isfinite(val):
                pi = pixel_index[c]
                sums[ai, pi] += val
                counts[ai, pi] += 1


@dataclass
class HeatmapData:
    grid: np.ndarray
    theta_edges: np.ndarray
    radius_edges: np.ndarray
    label: str
    source: Path


# --- Data Processing Helper Functions ---


def _safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def _read_local_zip_members(path: Path) -> dict[str, bytes]:
    logger.debug("Attempting manual zip member extraction for path: %s", path)
    data = path.read_bytes()
    out: dict[str, bytes] = {}
    pos = 0
    sig = b"PK\x03\x04"
    count = 0
    while True:
        found = data.find(sig, pos)
        if found < 0 or found + 30 > len(data):
            break
        try:
            (
                signature,
                _version,
                flag,
                method,
                _mtime,
                _mdate,
                _crc,
                comp_size,
                _uncomp_size,
                name_len,
                extra_len,
            ) = struct.unpack_from("<IHHHHHIIIHH", data, found)
        except struct.error:
            logger.warning("Struct unpack failed at position %d", found)
            break
        if signature != 0x04034B50:
            pos = found + 4
            continue
        name_start = found + 30
        name_end = name_start + name_len
        body_start = name_end + extra_len
        if flag & 0x08 or name_end > len(data) or body_start > len(data):
            break
        body_end = body_start + comp_size
        if body_end > len(data):
            break
        name = data[name_start:name_end].decode("utf-8", errors="replace")
        payload = data[body_start:body_end]
        try:
            if method == zipfile.ZIP_STORED:
                raw = payload
            elif method == zipfile.ZIP_DEFLATED:
                raw = zlib.decompress(payload, -15)
            else:
                raw = b""
            if raw:
                out[name] = raw
                count += 1
        except zlib.error as err:
            logger.warning("Decompression failed for entry %s: %s", name, err)
            pass
        pos = body_end
    logger.debug("Extracted %d zip members manually.", count)
    return out


def _load_npz_members(path: Path) -> tuple[dict[str, bytes], bool]:
    logger.debug("Opening NPZ archive via standard zipfile: %s", path)
    try:
        with zipfile.ZipFile(path, "r") as zf:
            members = {name: zf.read(name) for name in zf.namelist()}
            logger.debug("Successfully read %d members via ZipFile.", len(members))
            return members, False
    except zipfile.BadZipFile:
        logger.warning(
            "BadZipFile exception encountered. Falling back to manual zip parsing."
        )
        members = _read_local_zip_members(path)
        if not members:
            logger.error("Failed to recover any zip members manually.")
            raise
        logger.info("Recovered %d zip members manually.", len(members))
        return members, True


def _read_npy(data: bytes) -> np.ndarray:
    with io.BytesIO(data) as bio:
        arr = np.lib.format.read_array(bio, allow_pickle=False)
        logger.debug(
            "Parsed NPY array with shape %s and dtype %s.", arr.shape, arr.dtype
        )
        return arr


def load_npz_heatmap(
    path: Path,
    *,
    angle_bins: int,
    radial_bins: int,
    inner_diameter_mm: float,
    outer_diameter_mm: float,
    scan_angle_deg: float,
    remap_lines: bool,
    q_low: float,
    q_high: float,
    progress_callback=None,
) -> HeatmapData:
    logger.info("Starting NPZ heatmap loading process for: %s", path)
    members, recovered = _load_npz_members(path)
    names = set(members)
    chunks = (
        json.loads(members["chunks.json"].decode("utf-8"))
        if "chunks.json" in names
        else []
    )
    logger.debug("Found %d chunk definitions in metadata.", len(chunks))

    value_names = sorted(name for name in names if name.endswith("_values.npy"))
    angle_names = sorted(
        name for name in names if name.endswith("_angles_deg.npy")
    )
    logger.debug(
        "Identified %d value files and %d angle files.",
        len(value_names),
        len(angle_names),
    )

    if not value_names or not angle_names:
        raise ValueError("NPZ 数据块缺少 *_values.npy / *_angles_deg.npy 文件。")

    first = _read_npy(members[value_names[0]])
    width = int(first.shape[1])
    angle_bins = max(1, int(angle_bins))
    radial_bins = max(1, int(radial_bins or width))
    logger.debug(
        "Configured bins -> angle_bins: %d, radial_bins: %d, frame width: %d",
        angle_bins,
        radial_bins,
        width,
    )

    total_lines = sum(int(_read_npy(members[name]).size) for name in angle_names)
    logger.debug("Calculated total scan lines across all chunks: %d", total_lines)

    line_offset = 0
    sums = np.zeros((angle_bins, radial_bins), dtype=np.float64)
    counts = np.zeros((angle_bins, radial_bins), dtype=np.uint32)
    pixel_index = np.floor(
        np.arange(width, dtype=np.float64) / max(width, 1) * radial_bins
    ).astype(np.int32)
    pixel_index = np.clip(pixel_index, 0, radial_bins - 1)
    pixel_index = (radial_bins - 1) - pixel_index

    value_kind = chunks[0].get("value_kind") if chunks else "value"
    label = "height (um)"
    logger.debug("Detected value_kind: %s", value_kind)

    total_chunks = len(value_names)
    for idx, (value_name, angle_name) in enumerate(
        zip(value_names, angle_names, strict=False)
    ):
        logger.debug(
            "Processing chunk block %d/%d: %s | %s",
            idx + 1,
            len(value_names),
            value_name,
            angle_name,
        )
        values = _read_npy(members[value_name]).astype(np.float32, copy=False)
        stored_angles = (
            _read_npy(members[angle_name]).astype(np.float64, copy=False) % 360.0
        )

        if values.ndim != 2 or stored_angles.size != values.shape[0]:
            raise ValueError(
                f"NPZ 数据块尺寸不匹配: {value_name} / {angle_name}"
            )

        if value_kind == "height_mm":
            values = values * MICRONS_PER_MM
        else:
            values = values * MICRONS_PER_RANGE_UNIT

        if remap_lines:
            rows = np.arange(
                line_offset, line_offset + values.shape[0], dtype=np.float64
            )
            angles = (rows / max(total_lines, 1) * float(scan_angle_deg)) % 360.0
        else:
            angles = stored_angles

        line_offset += values.shape[0]
        angle_index = np.floor(angles / 360.0 * angle_bins).astype(np.int32)
        angle_index = np.clip(angle_index, 0, angle_bins - 1)

        # 🚀 ACCELERATED WITH NUMBA
        _accumulate_grid_numba(values, angle_index, pixel_index, sums, counts)

        # Update Progress Bar dynamically (0%-80% range for chunk parsing)
        if progress_callback:
            percent = int(((idx + 1) / total_chunks) * 80)
            progress_callback(percent)

    grid = np.full((angle_bins, radial_bins), np.nan, dtype=np.float32)
    valid_grid = counts > 0
    grid[valid_grid] = (sums[valid_grid] / counts[valid_grid]).astype(np.float32)
    logger.debug(
        "Grid accumulation complete. Valid grid cells: %d / %d",
        np.count_nonzero(valid_grid),
        grid.size,
    )

    inner_r = float(inner_diameter_mm) / 2.0
    outer_r = float(outer_diameter_mm) / 2.0
    if not (outer_r > inner_r > 0):
        raise ValueError("内外径必须有效，例如内径230、外径330。")

    theta_edges = np.linspace(0.0, 2.0 * np.pi, angle_bins + 1)
    radius_edges = np.linspace(inner_r, outer_r, radial_bins + 1)
    if recovered:
        label += " (recovered NPZ)"

    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        raise ValueError("热力图没有有效值。")

    q_low_val, q_high_val = np.nanpercentile(finite, [q_low, q_high])
    logger.info(
        "Percentile bounds calculated: q_low(%.1f%%) = %.3f, q_high(%.1f%%) = %.3f",
        q_low,
        q_low_val,
        q_high,
        q_high_val,
    )

    return HeatmapData(
        grid=grid,
        theta_edges=theta_edges,
        radius_edges=radius_edges,
        label=label,
        source=path,
    )


def load_csv_heatmap(
    path: Path,
    *,
    angle_bins: int,
    radial_bins: int,
    inner_diameter_mm: float,
    outer_diameter_mm: float,
    q_low: float,
    q_high: float,
    progress_callback=None,
) -> HeatmapData:
    logger.info("Starting CSV heatmap loading process for: %s", path)
    
    file_size = path.stat().st_size

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        logger.debug("CSV Field names discovered: %s", fields)

        height_field = "height_mm" if "height_mm" in fields else "range_units"
        if not {"angle_deg", "pixel", height_field}.issubset(fields):
            raise ValueError(
                "CSV 缺少 angle_deg/pixel/height_mm 或 range_units 字段。"
            )

        angles: list[float] = []
        pixels: list[int] = []
        values: list[float] = []
        max_pixel = 0
        row_count = 0

        for row in reader:
            row_count += 1
            angle = _safe_float(row.get("angle_deg"))
            pixel = _safe_float(row.get("pixel"))
            value = _safe_float(row.get(height_field))

            if row_count % 10000 == 0 and progress_callback and file_size > 0:
                read_bytes = f.tell()
                percent = min(80, int((read_bytes / file_size) * 80))
                progress_callback(percent)

            if angle is None or pixel is None or value is None:
                continue
            pix = int(round(pixel))
            if pix < 0:
                continue
            angles.append(angle % 360.0)
            pixels.append(pix)
            if height_field == "height_mm":
                values.append(value * MICRONS_PER_MM)
            else:
                values.append(value * MICRONS_PER_RANGE_UNIT)
            max_pixel = max(max_pixel, pix)

    logger.debug(
        "Parsed %d CSV rows (%d valid entries). Max pixel index: %d",
        row_count,
        len(values),
        max_pixel,
    )
    if not values:
        raise ValueError("CSV 没有可用数据。")

    if progress_callback:
        progress_callback(80)

    angle_bins = max(1, int(angle_bins))
    radial_bins = max(1, int(radial_bins or (max_pixel + 1)))

    angle_index = np.floor(np.asarray(angles) / 360.0 * angle_bins).astype(np.int32)
    angle_index = np.clip(angle_index, 0, angle_bins - 1)
    pixel_index = np.floor(
        np.asarray(pixels) / max(max_pixel + 1, 1) * radial_bins
    ).astype(np.int32)
    pixel_index = np.clip(pixel_index, 0, radial_bins - 1)
    pixel_index = (radial_bins - 1) - pixel_index

    sums = np.zeros((angle_bins, radial_bins), dtype=np.float64)
    counts = np.zeros((angle_bins, radial_bins), dtype=np.uint32)
    np.add.at(sums, (angle_index, pixel_index), np.asarray(values, dtype=np.float64))
    np.add.at(counts, (angle_index, pixel_index), 1)

    grid = np.full((angle_bins, radial_bins), np.nan, dtype=np.float32)
    valid = counts > 0
    grid[valid] = (sums[valid] / counts[valid]).astype(np.float32)

    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        raise ValueError("热力图没有有效值。")

    q_low_val, q_high_val = np.nanpercentile(finite, [q_low, q_high])

    inner_r = float(inner_diameter_mm) / 2.0
    outer_r = float(outer_diameter_mm) / 2.0
    theta_edges = np.linspace(0.0, 2.0 * np.pi, angle_bins + 1)
    radius_edges = np.linspace(inner_r, outer_r, radial_bins + 1)
    label = "height (um)"

    return HeatmapData(
        grid=grid,
        theta_edges=theta_edges,
        radius_edges=radius_edges,
        label=label,
        source=path,
    )


# --- PyWebView API Interface ---


class Api:
    """Methods in this class are exposed to JavaScript via pywebview.api"""

    def __init__(self):
        self.window = None

    def _update_js_progress(self, percent: int):
        """Dispatches progress percentage updates to JavaScript"""
        if self.window:
            self.window.evaluate_js(f"updateProgressBar({percent})")

    def say_hello(self, name):
        logger.debug("say_hello called with argument: %s", name)
        return f"Hello {name} from the Python backend!"

    def openFile(self, **kwargs):
        logger.info("Triggered openFile dialog from JS interface with options: %s", kwargs)
        file_types = (
            "Profile Files (*.npz;*.csv)",
            "NPZ Files (*.npz)",
            "CSV Files (*.csv)",
            "All files (*.*)",
        )

        result = self.window.create_file_dialog(
            dialog_type=webview.FileDialog.OPEN,
            allow_multiple=False,
            file_types=file_types,
        )

        if not result:
            logger.info("File dialog was cancelled by user.")
            return {"status": "cancelled", "message": "File selection cancelled."}

        file_path = str(result[0])
        logger.info("Selected file: %s", file_path)
        return {"status": "success", "file_path": file_path}

    def processAndVisualize(self, kwargs):
        file_path_str = kwargs.get("file_path")
        if not file_path_str:
            return {"status": "error", "message": "No file path provided."}

        file_path = Path(file_path_str)
        params = {
            "angle_bins": kwargs.get("angle_bins", 720),
            "radial_bins": kwargs.get("radial_bins", 900),
            "inner_diameter_mm": kwargs.get("inner_diameter_mm", 230.0),
            "outer_diameter_mm": kwargs.get("outer_diameter_mm", 330.0),
            "q_low": kwargs.get("q_low", 1.0),
            "q_high": kwargs.get("q_high", 99.0),
            "progress_callback": self._update_js_progress,
        }

        # Step 1: Execute calculation algorithm pipeline
        try:
            self._update_js_progress(5)
            if file_path.suffix.lower() == ".npz":
                logger.info("Routing process to NPZ loader.")
                data = load_npz_heatmap(
                    file_path,
                    scan_angle_deg=kwargs.get("scan_angle_deg", 360.0),
                    remap_lines=kwargs.get("remap_lines", True),
                    **params,
                )
            elif file_path.suffix.lower() == ".csv":
                logger.info("Routing process to CSV loader.")
                data = load_csv_heatmap(file_path, **params)
            else:
                logger.error(
                    "Unsupported file extension provided: %s", file_path.suffix
                )
                return {
                    "status": "error",
                    "message": f"Unsupported format: {file_path.suffix}",
                }
        except Exception as e:
            logger.exception("An exception occurred while processing dataset:")
            return {"status": "error", "message": f"Processing Error: {str(e)}"}

        # Step 2: Render & save visualization to PNG file
        self._update_js_progress(85)
        output_png_path = file_path.with_suffix(".png")
        logger.info("Rendering heatmap visualization to PNG: %s", output_png_path)

        fig = Figure(figsize=(9, 8), dpi=100)
        ax = fig.add_subplot(111, projection="polar")

        theta, radius = np.meshgrid(data.theta_edges, data.radius_edges)
        finite = data.grid[np.isfinite(data.grid)]
        vmin, vmax = np.nanpercentile(finite, [params["q_low"], params["q_high"]])

        mesh = ax.pcolormesh(
            theta,
            radius,
            data.grid.T,
            shading="auto",
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0.0, float(data.radius_edges[-1]))
        ax.set_ylabel("Radius (mm)")
        ax.set_title(data.source.name)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.08, shrink=0.78)
        cbar.set_label(data.label)

        fig.savefig(output_png_path, bbox_inches="tight")
        logger.info("PNG export complete: %s", output_png_path)

        self._update_js_progress(100)

        # Step 3: Compute basic dataset statistics
        z_min = float(np.nanmin(finite))
        z_max = float(np.nanmax(finite))
        z_mean = float(np.nanmean(finite))

        return {
            "status": "success",
            "file_path": str(file_path),
            "saved_image": str(output_png_path),
            "stats": {
                "min": z_min,
                "max": z_max,
                "range": z_max - z_min,
                "mean": z_mean,
            },
        }


if __name__ == "__main__":
    logger.info("Starting PyWebView Desktop Application backend...")
    api = Api()

    window = webview.create_window(
        title="Breakdisk Analysis Tool",
        url="assets/index.html",
        js_api=api,
        width=800,
        height=600,
    )

    api.window = window
    webview.start(debug=True)