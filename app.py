from __future__ import annotations

# removed CSV functionality.
# instead, just rely on NPZ. if time allows, just convert the csv to npz easy.

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

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


matplotlib.use("Agg")
from matplotlib.figure import Figure
import webview


webview.settings["ALLOW_FILE_URLS"] = True

# Global variables for making passing to js easer
innerDim = 0;
outerDim = 0;



logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s",

)
logger = logging.getLogger("Breakdisk Analysis Tool")

# Constants (Borrowed from zekai's original version)
MICRONS_PER_RANGE_UNIT = 0.167569681
MICRONS_PER_MM = 1000.0



# numba compiled grid accumulation
# High level overview would just be that previous iteration did use c level libraries which is good, but entire binning process was
# was wrapped with a python iteration. very quick replacement to use numba instead and actually fully get to feel the new speed using
# it to compile to machine code
@njit
def _accumulate_grid_numba(
    breakdisk_Around: np.ndarray,
    angle_index: np.ndarray,
    pixel_index: np.ndarray,
    sums: np.ndarray,
    counts: np.ndarray,
):

    num_rows, num_cols = breakdisk_Around.shape
    for row in range(num_rows):
        ai = angle_index[row]

        for c in range(num_cols):
            
            val = breakdisk_Around[row, c]
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




def _read_npz(path: Path) -> dict[str, bytes]:
    logger.debug("Opening npz: %s", path)
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
        members = _read_npz(path)
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





# User customization features






# Identifies plane tilt in polar space using linear least squares =and levels the tilt (A*X + B*Y) while preserving the absolute height scale.
def apply_flatness_adjust(
    grid: np.ndarray, theta_edges: np.ndarray, radius_edges: np.ndarray
) -> np.ndarray:

    # Compute bin centers for polar coordinates
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    radius_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])

    # Create 2D coordinate grid (angle_bins, radial_bins)
    theta_grid, radius_grid = np.meshgrid(theta_centers, radius_centers, indexing="ij")

    X = radius_grid * np.cos(theta_grid)
    Y = radius_grid * np.sin(theta_grid)

    mask = np.isfinite(grid)
    if not np.any(mask):
        return grid

    x_valid = X[mask]
    y_valid = Y[mask]
    z_valid = grid[mask]

    # use linear regression approachj
    M = np.column_stack([x_valid, y_valid, np.ones_like(x_valid)])

    # Solve linear least squares fit
    (A, B, C), _, _, _ = np.linalg.lstsq(M, z_valid, rcond=None)

    # Subtract ONLY the slope components (A*X + B*Y) to level tilt without shifting height level
    tilt = A * X + B * Y
    adjusted_grid = grid - tilt

    logger.debug("Flatness tilt removed: A=%.4e, B=%.4e (Intercept C=%.4e retained)", A, B, C)
    return adjusted_grid.astype(np.float32)



# In all of the code, outlier filling is still referred to as "null filling" although technically that is not a real thing anymore
def apply_null_filling(
    grid: np.ndarray, 
    kernel_size: int = 25, 
    threshold_std: float = 2.0, 
    max_abs_diff: float | None = None
) -> np.ndarray:
    
    cleaned_grid = np.asarray(grid, dtype=np.float32).copy()
    pad = kernel_size // 2

    padded = np.pad(cleaned_grid, ((pad, pad), (pad, pad)), mode="constant", constant_values=np.nan)

    padded[0:pad, pad:-pad] = cleaned_grid[-pad:, :]
    padded[-pad:, pad:-pad] = cleaned_grid[0:pad, :]

    angle_bins, radial_bins = cleaned_grid.shape

    neighbors = []
    for di in range(kernel_size):
        for dj in range(kernel_size):
            if di == pad and dj == pad:
                continue  # Skip the center cell itself
            neighbors.append(padded[di : di + angle_bins, dj : dj + radial_bins])

    neighbors_stack = np.stack(neighbors, axis=0)  # Shape: (kernel_size^2 - 1, angle_bins, radial_bins)

    # 3. Calculate local neighborhood mean and std (ignoring NaNs)
    local_mean = np.nanmean(neighbors_stack, axis=0)
    local_std = np.nanstd(neighbors_stack, axis=0)

    # Prevent division by zero or extreme sensitivity on perfectly flat regions
    local_std = np.maximum(local_std, 1e-5)

    # 4. Identify outliers (super high points / extreme noise)
    diff = np.abs(cleaned_grid - local_mean)
    
    outlier_mask = diff > (threshold_std * local_std)

    if max_abs_diff is not None:
        outlier_mask |= diff > max_abs_diff

    # Ensure we don't process pre-existing NaNs as outliers
    outlier_mask &= np.isfinite(cleaned_grid)

    # 5. Replace outliers with local neighborhood average
    cleaned_grid[outlier_mask] = local_mean[outlier_mask]

    logger.debug(
        "Outlier removal complete: replaced %d spike points.", 
        np.count_nonzero(outlier_mask)
    )

    return cleaned_grid





# 
def apply_radial_flattening(
    grid: np.ndarray, radius_edges: np.ndarray, method: str = "profile"
) -> np.ndarray:
    flattened_grid = np.asarray(grid, dtype=np.float32).copy()
    angle_bins, radial_bins = flattened_grid.shape

    if method == "profile":
        # Find mean for compare
        radial_profile = np.nanmean(flattened_grid, axis=0)

        # Center the radial profile around zero so absolute baseline is maintained
        if np.any(np.isfinite(radial_profile)):
            radial_profile -= np.nanmean(radial_profile)
            # Subtract average radial shape from every angular ray
            flattened_grid = flattened_grid - radial_profile

    elif method == "per_ray":
        radius_centers = 0.5 * (radius_edges[:-1] + radius_edges[1:])

        for a in range(angle_bins):
            ray = flattened_grid[a, :]
            valid = np.isfinite(ray)

            # Creates a way if valid
            if np.count_nonzero(valid) >= 2:
                r_valid = radius_centers[valid]
                z_valid = ray[valid]

                # Use a regular 
                slope, intercept = np.polyfit(r_valid, z_valid, 1)

                # Remove trend centered on the ray's mean radius
                r_mean = np.mean(r_valid)
                trend = slope * (radius_centers - r_mean)
                flattened_grid[a, :] -= trend

    logger.debug("Applied radial flattening using method: %s", method)
    return flattened_grid.astype(np.float32)


"""
def apply_polar_blur(grid: np.ndarray, blur_size: int) -> np.ndarray:

    if blur_size <= 1:
        return grid.copy()

    pad = blur_size // 2
    angle_bins, radial_bins = grid.shape

    padded = np.pad(grid, ((pad, pad), (0, 0)), mode="wrap")
    padded = np.pad(padded, ((0, 0), (pad, pad)), mode="edge")

    blurred_grid = np.empty_like(grid, dtype=np.float32)

    for a in range(angle_bins):
        for r in range(radial_bins):
            window = padded[a : a + blur_size, r : r + blur_size]
            
            valid_mask = np.isfinite(window)
            if np.any(valid_mask):
                blurred_grid[a, r] = np.mean(window[valid_mask])
            else:
                blurred_grid[a, r] = np.nan

    logger.debug("Applied polar mean blur with window size: %dx%d", blur_size, blur_size)
    return blurred_grid
"""






















def load_npz_heatmap(
    path: Path,
    *,
    angle_bins: int = 720,
    radial_bins: int = 900,
    inner_diameter_mm: float = 230.0,
    outer_diameter_mm: float = 320.0,
    scan_angle_deg: float = 360.0,
    remap_lines: bool = True,
    q_low: float = 1.0,
    q_high: float = 99.0,
    start_scan_range: float = 0.0, # the start of the scan
    end_scan_range: float = 360.0, # the end of the scan
    ignore_minimum: int | None = 5600, # any value lower than this gets turned into null. Don't directly write to NPZ file, but rather when copying it down just replace it with null
    ignore_maximum: int | None = 6650, # same thing with the minimum thing
    blur: int = 0,

    reference_zeroing: bool = False,
    flatness_adjust: bool = False,
    null_filling: bool = False,
    radial_flattening: bool = False,


    cancel_check=None,
    progress_callback=None
) -> HeatmapData:

    global innerDim 
    innerDim = inner_diameter_mm
    global outerDim 
    outerDim = outer_diameter_mm
    
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
    angle_bins = max(180, int(angle_bins))
    radial_bins = max(150, int(radial_bins or width))
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

    # Scan range bounds, agnostic to order even though the user should know better lol
    low_scan_deg = min(float(start_scan_range), float(end_scan_range))
    high_scan_deg = max(float(start_scan_range), float(end_scan_range))
    range_span_deg = high_scan_deg - low_scan_deg

    total_chunks = len(value_names)
    for idx, (value_name, angle_name) in enumerate(
        zip(value_names, angle_names, strict=False)
    ):
        if cancel_check:
                cancel_check()
        
        logger.debug(
            "Processing chunk block %d/%d: %s | %s",
            idx + 1,
            len(value_names),
            value_name,
            angle_name,
        )
        values = _read_npy(members[value_name]).astype(np.float32, copy=False)
        stored_angles = _read_npy(members[angle_name]).astype(np.float64, copy=False)

        if values.ndim != 2 or stored_angles.size != values.shape[0]:
            raise ValueError(
                f"NPZ 数据块尺寸不匹配: {value_name} / {angle_name}"
            )

        if value_kind == "height_mm":
            values = values * MICRONS_PER_MM
        else:
            values = values * MICRONS_PER_RANGE_UNIT


        if ignore_minimum is not None: values[values < ignore_minimum] = np.nan
        if ignore_maximum is not None: values[values > ignore_maximum] = np.nan

        num_chunk_lines = values.shape[0]

        if remap_lines:
            rows = np.arange( 
                line_offset, line_offset + num_chunk_lines, dtype=np.float64
            )
            angles = rows / max(total_lines, 1) * float(scan_angle_deg)
        else:
            angles = stored_angles

        line_offset += num_chunk_lines

        angles_grid = angles % 360.0
        angle_index = np.floor(angles_grid / 360.0 * angle_bins).astype(np.int32)
        angle_index = np.clip(angle_index, 0, angle_bins - 1)

        # ACCELERATED WITH NUMBA
        _accumulate_grid_numba(values, angle_index, pixel_index, sums, counts)

        # Update Progress Bar dynamically in the JS callback
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

    # apply user customizatin features

    theta_edges = np.linspace(0.0, 2.0 * np.pi, angle_bins + 1)
    radius_edges = np.linspace(inner_r, outer_r, radial_bins + 1)

    if flatness_adjust:
        grid = apply_flatness_adjust(grid, theta_edges, radius_edges)

    if null_filling:
        grid = apply_null_filling(grid)

    if radial_flattening:
        grid = apply_radial_flattening(grid, radius_edges, method="profile")


    # if blur and int(blur) > 1:
    #     grid = apply_polar_blur(grid, int(blur))

    if reference_zeroing:
        finite = grid[np.isfinite(grid)]
        grid_mean = np.nanmean(finite)
        grid = grid - grid_mean
        label = "height - mean (um)"

    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        raise ValueError("热力图没有有效值。")




    
    # Create full angular grid centers (in degrees) to match grid rows
    angles_deg = np.linspace(0.0, 360.0, angle_bins, endpoint=False)

    # Mask out indices that fall outside [start_scan_range, end_scan_range]
    if range_span_deg < 360.0:
        valid_angular_mask = ((angles_deg - low_scan_deg) % 360.0) <= range_span_deg
        grid[~valid_angular_mask, :] = np.nan

    # Ensure there is valid data remaining inside the requested range
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        raise ValueError("Heatmap has no valid values within the selected scan range.")

    return HeatmapData(
        grid=grid,
        theta_edges=theta_edges,
        radius_edges=radius_edges,
        label=label,
        source=path,
    )


# API mehtods


class Api:
    """Methods in this class are exposed to JavaScript via pywebview.api to allow for data transfer to frontend and user int. and such"""

    def __init__(self):
        self.window = None
        self._is_cancelled = False

    def cancelProcessing(self):
        """Called by JavaScript to signal cancellation."""
        logger.info("Cancellation requested via API.")
        self._is_cancelled = True
        return {"status": "cancelled"}

    def _check_cancellation(self):
        """Helper to raise an exception if cancel was requested."""
        if self._is_cancelled:
            logger.info("Processing safely halted due to cancellation signal.")
            raise InterruptedError("Processing was cancelled by the user.")



    def _update_js_progress(self, percent: int):
        """Dispatches progress percentage updates to JavaScript"""
        if self.window:
            self.window.evaluate_js(f"updateProgressBar({percent})")


    def openFile(self, **kwargs):
        logger.info("Triggered openFile dialog from JS interface with options: %s", kwargs)
        file_types = (
            "NPZ Files (*.npz)",
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
        logger.debug("Received!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! kwargs from JS: %s", kwargs)

        # RESET CANCELLATION FLAG BEFORE STARTING
        self._is_cancelled = False 

        file_path_str = kwargs.get("file_path")
        if not file_path_str:
            return {"status": "error", "message": "No file path provided."}

        file_path = Path(file_path_str)

        zero_at_right = bool(kwargs.get("zero_at_right", False))
        
        # Pass self._check_cancellation into params
        params = {
            "angle_bins": kwargs.get("angle_bins", 720),
            "radial_bins": kwargs.get("radial_bins", 900),
            "inner_diameter_mm": kwargs.get("inner_diameter_mm", 230.0),
            "outer_diameter_mm": kwargs.get("outer_diameter_mm", 320.0),
            "q_low": float(kwargs.get("q_low", 1.0) if kwargs.get("q_low") is not None else 1.0),
            "q_high": float(kwargs.get("q_high", 99.0) if kwargs.get("q_high") is not None else 99.0),
            "start_scan_range": kwargs.get("start_scan_range", 0.0),
            "end_scan_range": kwargs.get("end_scan_range", 360.0),

            "ignore_minimum": kwargs.get("ignore_minimum"),
            "ignore_maximum": kwargs.get("ignore_maximum"),

            "blur": kwargs.get("blur"),

            "reference_zeroing": bool(kwargs.get("reference_zeroing", False)),
            "flatness_adjust": bool(kwargs.get("flatness_adjust", False)),
            "null_filling": bool(kwargs.get("null_filling", False)),
            "radial_flattening": bool(kwargs.get("radial_flattening", False)),

            "cancel_check": self._check_cancellation,  # <-- Pass check function here
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

            else:

                logger.error("Unsupported file extension provided: %s", file_path.suffix)
                return {
                    "status": "error",
                    "message": f"Unsupported format: {file_path.suffix}",
                }
        except InterruptedError:
            return {"status": "cancelled", "message": "Processing was cancelled."}
        except Exception as e:
            logger.exception("An exception occurred while processing dataset:")
            return {"status": "error", "message": f"Processing Error: {str(e)}"}

        # Step 2: Render & save visualization to PNG file
        self._update_js_progress(85)
        
        output_dir = Path("assets")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_png_path = output_dir / "output.png"
        output_json_path = output_dir / "outputInfo.json"
        logger.info("Rendering heatmap visualization to PNG: %s", output_png_path)

        fig = Figure(figsize=(9, 8), dpi=500)
        ax = fig.add_subplot(111, projection="polar")

        theta, radius = np.meshgrid(data.theta_edges, data.radius_edges)

        
        # Calculate percentiles directly using the payload values on valid points within the range
        finite = data.grid[np.isfinite(data.grid)]
        q_low_val = float(params["q_low"])
        q_high_val = float(params["q_high"])
        vmin, vmax = np.nanpercentile(finite, [q_low_val, q_high_val])



        logger.debug("Applying color scale range (vmin: %.3f, vmax: %.3f) for q_low: %.1f%%, q_high: %.1f%%", 
                     vmin, vmax, q_low_val, q_high_val)

        mesh = ax.pcolormesh(
            theta,
            radius,
            data.grid.T,
            shading="auto",
            antialiased=False,
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )


        # if user picks zero to right, shift everything
        if zero_at_right:
            ax.set_theta_zero_location("E") 
            ax.set_theta_direction(1)
        else:
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1) # Clockwise, reg config

        ax.set_ylim(0.0, float(data.radius_edges[-1]))
        ax.set_ylabel("Radius (mm)")
        ax.set_title(data.source.name)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.08, shrink=0.78)
        cbar.set_label(data.label)

        fig.savefig(output_png_path, bbox_inches="tight")
        logger.info("PNG export complete: %s", output_png_path)

        # --- STEP 2.5: EXPORT outputInfo.json ---
        # Convert NaNs to None so json.dump creates standard `null` values
        clean_grid = np.where(np.isnan(data.grid), None, data.grid)
        polar_coordinate_list = clean_grid.tolist()

        json_data = {
            "source": data.source.name,
            "angle_bins": data.grid.shape[0],
            "radial_bins": data.grid.shape[1],
            "inner_diameter_mm": kwargs.get("inner_diameter_mm", 230.0),
            "outer_diameter_mm": kwargs.get("outer_diameter_mm", 320.0),
            "q_low": float(kwargs.get("q_low", 1.0) if kwargs.get("q_low") is not None else 1.0),
            "q_high": float(kwargs.get("q_high", 99.0) if kwargs.get("q_high") is not None else 99.0),
            "data": polar_coordinate_list,
        }


        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f)
        
        logger.info("JSON export complete: %s", output_json_path)

        self._update_js_progress(100)

        # Step 3: Compute basic dataset statistics
        z_min = float(np.nanmin(finite))
        z_max = float(np.nanmax(finite))
        z_mean = float(np.nanmean(finite))
        z_sdv = float(np.nanstd(finite))


        return {
            "status": "success",
            "file_path": str(file_path),
            "saved_image": "output.png",
            "saved_json": "outputInfo.json",
            "stats": {
                "min": z_min,
                "max": z_max,
                "range": z_max - z_min,
                "mean": z_mean,
                "sdv": z_sdv,
                "inmUsed": innerDim / outerDim
            },
        }


    def generatePDFReport(self, kwargs):
        try:
            logger.info("Starting PDF report generation...")
            output_dir = Path("assets")
            pdf_path = output_dir / "report.pdf"
            image_path = output_dir / "output.png"

            if not image_path.exists():
                return {"status": "error", "message": "No visualization image found."}

            doc = SimpleDocTemplate(str(pdf_path), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
            styles = getSampleStyleSheet()
            story = []

            # 1. Header & Source File
            title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor=colors.HexColor("#1e293b"))
            story.append(Paragraph("<b>Brake Disk Metrology & Analysis Report</b>", title_style))
            story.append(Spacer(1, 10))

            source_file = kwargs.get("source_file", "Unknown File")
            story.append(Paragraph(f"<b>Source File:</b> {source_file}", styles['Normal']))
            story.append(Spacer(1, 12))

            # 2. Geometry & Grid Metadata Table
            story.append(Paragraph("<b>1. Geometry & Grid Resolution</b>", styles['Heading2']))
            story.append(Spacer(1, 6))

            stats = kwargs.get("stats", {})
            geom_data = [
                ["Parameter", "Value", "Parameter", "Value"],
                ["Inner Diameter", f"{kwargs.get('inner_diameter_mm', 230.0)} mm", "Angular Bins", f"{kwargs.get('angle_bins', 720)}"],
                ["Outer Diameter", f"{kwargs.get('outer_diameter_mm', 320.0)} mm", "Radial Bins", f"{kwargs.get('radial_bins', 900)}"],
                ["Inner/Outer Ratio", f"{stats.get('inmUsed', 0):.4f}", "Scan Range", f"{kwargs.get('start_scan_range', 0)}° - {kwargs.get('end_scan_range', 360)}°"],
            ]
            t_geom = Table(geom_data, colWidths=[110, 110, 110, 110])
            t_geom.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3b82f6")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            story.append(t_geom)
            story.append(Spacer(1, 12))

            # 3. Processing Pipeline Configuration
            story.append(Paragraph("<b>2. Data Processing Pipeline</b>", styles['Heading2']))
            story.append(Spacer(1, 6))

            proc_data = [
                ["Pipeline Option", "Status / Parameter"],
                ["Flatness Adjustment (Tilt)", "Enabled" if kwargs.get("flatness_adjust") else "Disabled"],
                ["Null Filling / Spike Removal", "Enabled" if kwargs.get("null_filling") else "Disabled"],
                ["Radial Flattening", "Profile Subtraction" if kwargs.get("radial_flattening") else "Disabled"],
                ["Reference Zeroing", "Centered to Mean" if kwargs.get("reference_zeroing") else "Absolute"],
                ["Polar Blur", f"Kernel Size: {kwargs.get('blur', 0)}" if kwargs.get('blur') else "Disabled"],
                ["Quantile Clip Range", f"{kwargs.get('q_low', 1.0)}% - {kwargs.get('q_high', 99.0)}%"],
            ]
            t_proc = Table(proc_data, colWidths=[220, 220])
            t_proc.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#64748b")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            story.append(t_proc)
            story.append(Spacer(1, 12))

            # 4. Surface Statistics Table
            story.append(Paragraph("<b>3. Surface Metrology Statistics</b>", styles['Heading2']))
            story.append(Spacer(1, 6))

            data_table = [
                ["Metric", "Symbol", "Value"],
                ["Maximum Height", "Z_max", f"{stats.get('max', 0):.3f} µm"],
                ["Minimum Height", "Z_min", f"{stats.get('min', 0):.3f} µm"],
                ["Peak-to-Valley (PV)", "ΔZ", f"{stats.get('range', 0):.3f} µm"],
                ["Mean Height", "Z_avg", f"{stats.get('mean', 0):.3f} µm"],
                ["Standard Deviation / Rq", "σ", f"{stats.get('sdv', 0):.3f} µm"],
            ]

            table = Table(data_table, colWidths=[180, 80, 180])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
            ]))
            story.append(table)
            story.append(Spacer(1, 14))

            # 5. Visualization Attachment
            story.append(Paragraph("<b>4. Polar Heatmap Visualization</b>", styles['Heading2']))
            story.append(Spacer(1, 8))
            story.append(Image(str(image_path), width=420, height=370))

            doc.build(story)
            logger.info("PDF export complete: %s", pdf_path)
            return {"status": "success", "pdf_path": str(pdf_path)}

        except Exception as e:
            logger.exception("Failed to generate PDF report:")
            return {"status": "error", "message": f"PDF Generation Error: {str(e)}"}




    

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