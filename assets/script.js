let selectedFilePath = null;
let visualizationData = null;

let innerDiameterToOuterDiameterRatio;

// Global Transform State
const transformState = {
    scale: 1,
    translateX: 0,
    translateY: 0,
    minScale: 1,
    maxScale: 15,
    isDragging: false,
    dragStart: { x: 0, y: 0 }
};

// Mouse tracking relative to the viewport container
let currentMouseViewportPos = { x: 0, y: 0 };
let isMouseInsideViewport = false;

async function selectNativeFile() {
    const result = await window.pywebview.api.openFile();
    if (result && result.status === "success") {
        selectedFilePath = result.file_path;
        document.getElementById("filePathInput").textContent = selectedFilePath;
    }
}

function updateProgressBar(percentage) {
    const fill = document.getElementById("progressBarFill");
    const text = document.getElementById("progressBarText");
    const clampedPercent = Math.min(100, Math.max(0, Math.round(percentage)));
    if (fill) fill.style.width = clampedPercent + "%";
    if (text) text.textContent = "Processing: " + clampedPercent + "%";
}

async function runVisualization(event) {
    if (event?.shiftKey) return reloadVisualization();
    if (!selectedFilePath) {
        alert("Please select a file first!");
        return;
    }

    if (!selectedFilePath) {
        alert("Please select a file first!");
        return;
    }

    updateProgressBar(0);


    const payload = {
        file_path: selectedFilePath,
        angle_bins: parseInt(document.getElementById("parameterAngleBins")?.value) || 720,
        radial_bins: parseInt(document.getElementById("parameterRadialBins")?.value) || 900,
        inner_diameter_mm: parseFloat(document.getElementById("parameterInnerDiameter")?.value) || 230.0,
        outer_diameter_mm: parseFloat(document.getElementById("parameterOuterDiameter")?.value) || 320.0,

        q_low: parseFloat(document.getElementById("parameterQLimitLow")?.value) || 1,
        q_high: parseFloat(document.getElementById("parameterQLimitHigh")?.value) || 99,

        start_scan_range: parseFloat(document.getElementById("parameterStartScan")?.value) || 0.0,
        end_scan_range: parseFloat(document.getElementById("parameterEndScan")?.value) || 360.0,
        ignore_minimum: parseFloat(document.getElementById("parameterIgnoreMinimum")?.value) || null,
        ignore_maximum: parseFloat(document.getElementById("parameterIgnoreMaximum")?.value) || null,
        blur: parseFloat(document.getElementById("parameterGaussianBlur")?.value) || 0,

        flatness_adjust: document.getElementById("parameterFlatnessAdjustment").checked,
        reference_zeroing: document.getElementById("parameterReferenceZeroing").checked,
        zero_at_right: document.getElementById("parameterZeroAtRight").checked,
        radial_flattening: document.getElementById("parameterRadialFlattening").checked,
        null_filling: document.getElementById("parameterNullFilling").checked,


        scan_angle_deg: 360.0,
        remap_lines: true
    };

    try {
        const response = await window.pywebview.api.processAndVisualize(payload);
        
        if (response.status === "success") {
            document.getElementById("siInfoMax").textContent = "Max: " + response.stats.max
            document.getElementById("siInfoMin").textContent = "Min: " + response.stats.min
            document.getElementById("siInfoAverage").textContent = "Average: " + response.stats.mean
            document.getElementById("siInfoStdv").textContent = "Standard Deviation: " + response.stats.sdv

            innerDiameterToOuterDiameterRatio = response.stats.inmUsed;

            const transformLayer = document.getElementById("transformLayer");
            // timestamping trick to allow multiple images to be ran in the same session. all in cache, just separated by timestamp
            if (transformLayer) {
                transformLayer.style.backgroundImage = `url(${response.saved_image + "?t=" + new Date().getTime()})`;
            }

            resetZoom();
            await loadVisualizationData(true);
        } else {
            alert("Error: " + response.message);
        }
    } catch (err) {
        console.error("PyWebView call failed:", err);
        alert("Execution failed: " + err);
    }
}






// Reloads the output image and JSON data without rerunning processing
async function reloadVisualization() {
    const transformLayer = document.getElementById("transformLayer");
    if (transformLayer) {
        const currentBg = transformLayer.style.backgroundImage;
        // Strip the old cache-buster and reapply with a fresh timestamp
        const urlMatch = currentBg.match(/url\(([^?)"]+)/);
        const baseUrl = urlMatch ? urlMatch[1] : 'output.png';
        transformLayer.style.backgroundImage = `url(${baseUrl}?t=${new Date().getTime()})`;
    }

    resetZoom();
    await loadVisualizationData(true);
}


// Help info for docs
const helpIDList = [
    ["Inner Diameter", "Controls the size of the inner ring diameter in the visualization. It is best practice to utilize the actual inner diameter of the breakdisk and use IGNORE MINIMUM or IGNORE MAXIMUM if the laser line goes over."],
    ["Outer Diameter", "Controls the size of the outer ring diameter in the visualization. Similar to inner diameter, is best practice to utilize the actual outer diameter of the breakdisk and use IGNORE MINIMUM or IGNORE MAXIMUM if the laser line goes over."],
    ["Angle Bins", "Controls the amount of angle accumulation bins that the program samples to represent the entire breakdisk. 720 angle bins means one bin for every half degree on the breakdisk. \n\n Note that this only takes effect in the visualization and the mouse hover information tab. Calculations for static max/min/sdv/mean are calculated across all values."],
    ["Radial Bins", "Controls the amount of value accumulation bins that the program samples to represent each angle bin. 900 radial bins means one bin for every angle bin, there are 900 radial bins at that angle to differentiate distance from origin. \n\n Note that this only takes effect in the visualization and the mouse hover information tab. Calculations for static max/min/sdv/mean are calculated across all values."],
    ["Q Low", "Controls the lowest percentage of outliers that won't be considered by the visualization's color scale during drawing. Best used if you see that your visualization's color range is dominanted by low outliers."],
    ["Start Scan Range", "Controls the start of the scan range. Can go in the negatives too for extra control. Does not speed up binning or visualiztation time."],
    ["End Scan Range", "Controls the end of the scan range. Can go in the negatives too for extra control. Does not speed up binning or visualiztation time."],
    ["Ignore Minimum", "Anything lower this value is ignored, either as outliers or as an unwanted, not useful, or actively detrimental portion of the breakdisk. Measured in mm."],
    ["Ignore Maximum", "Anything above this value is ignored, either as outliers or as an unwanted, not useful, or actively detrimental portion of the breakdisk. Measured in mm."],
    ["Q High", "Controls the lowest percentage of outliers that won't be considered by the visualization's color scale during drawing. Best used if you see that your visualization's color range is dominanted by low outliers."],
    ["Flatness Adjustment", "Automatically adjusts the visualization for rotation of the breakdisk during scanning. There is a possibility that the rig or servo that the breakdisk is attached to is slightly tilted to one axis. This variation will overwhelm any useful patterns or data. Automatically turned on."],
    ["Reference Zeroing", "Changes the scale of the entire visualization to be plus or minus the average value."],
    ["Outlier Flattening", "Replaces all extreme outliers outside of the 2 standard deviations over the 25x25 bin local average with the local average. WARNING! This option is meant to clean up messy bits and obviously incorrect data outliers. However, since this is just a data analysis program it cannot differentiate between data points that are a result of data processing artifacts or actual, physical features. Use as a tool to better analyze broad structural changes or distortion to the breakdisk."],
    ["Radial Flattening", "Automatically adjusts the visualization for rotation of the laser during scanning. There is a possibility that the rig the line laser is attached to is slightly tilted to one axis. This variation will overwhelm any useful patterns or data. Automatically turned on."],
    ["0 at Right", "Automatically adjusts the 0 degree position to the right and goes counterclockwise. Purely an aestethetic (I know I spelled that wrong) change meant to replicate the unit circle and other standard trigonometric conventions of displaying polar data"],
    ["Two point height delta", "Finds the height difference between two points."],
    ["Line profile analysis", "Finds the line height profile between two points."]
];

function showHelp(id) {
    document.getElementById("overlayBG").style.display = "flex";
    document.getElementById("helpType").textContent = helpIDList[id][0];
    document.getElementById("helpInfoText").textContent = helpIDList[id][1];
}

async function loadVisualizationData(forceReload = false) {
    if (!forceReload && visualizationData) return visualizationData;

    try {
        const cacheBuster = "?t=" + new Date().getTime();
        const response = await fetch('outputInfo.json' + cacheBuster);
        visualizationData = await response.json();
        return visualizationData;
    } catch (err) {
        console.error("Failed to load outputInfo.json:", err);
        return null;
    }
}

// Applies CSS transform matrix based on scale and pan variables
function applyTransform() {
    const layer = document.getElementById("transformLayer");
    if (layer) {
        layer.style.transform = `translate(${transformState.translateX}px, ${transformState.translateY}px) scale(${transformState.scale})`;
    }
}

// Resets view back to 100% (1:1) scale
function resetZoom() {
    transformState.scale = 1;
    transformState.translateX = 0;
    transformState.translateY = 0;
    applyTransform();
}

// Zooms in/out anchored to a specific focal point in Viewport space
function zoomAtPoint(zoomFactor, focalX, focalY) {
    const newScale = Math.min(
        transformState.maxScale,
        Math.max(transformState.minScale, transformState.scale * zoomFactor)
    );

    if (newScale === transformState.scale) return;

    // Adjust translations so point under mouse stays anchored
    transformState.translateX = focalX - (focalX - transformState.translateX) * (newScale / transformState.scale);
    transformState.translateY = focalY - (focalY - transformState.translateY) * (newScale / transformState.scale);
    transformState.scale = newScale;

    // Lock position if zoom is completely reset
    if (transformState.scale === 1) {
        transformState.translateX = 0;
        transformState.translateY = 0;
    }

    applyTransform();
}





window.addEventListener("DOMContentLoaded", () => {
    const overlay = document.getElementById("overlayBG");
    if (overlay) {
        overlay.addEventListener("click", () => {
            overlay.style.display = "none";
        });
    }

    loadVisualizationData();

    const canvas = document.getElementById('finalImageOverlay');
    const viewport = document.getElementById('imageViewport');
    if (!canvas || !viewport) return;

    // Original analysis frame constants
    const CENTER_X = 256;
    const CENTER_Y = 264;
    // 275, 286
    const OUTER_R = 217;
    let INNER_R = OUTER_R * innerDiameterToOuterDiameterRatio;

    // Track mouse position over Viewport
    viewport.addEventListener('pointermove', (event) => {
        const rect = viewport.getBoundingClientRect();
        currentMouseViewportPos.x = event.clientX - rect.left;
        currentMouseViewportPos.y = event.clientY - rect.top;
        isMouseInsideViewport = true;
        INNER_R = OUTER_R * innerDiameterToOuterDiameterRatio;

        // Handle viewport panning logic
        if (transformState.isDragging) {
            transformState.translateX = event.clientX - transformState.dragStart.x;
            transformState.translateY = event.clientY - transformState.dragStart.y;
            applyTransform();
        }
    });

    viewport.addEventListener('pointerenter', () => { isMouseInsideViewport = true; });
    viewport.addEventListener('pointerleave', () => { 
        isMouseInsideViewport = false; 
        transformState.isDragging = false;
    });

    // Middle-click / Drag implementation to pan around zoomed image
    viewport.addEventListener('pointerdown', (event) => {
        if (event.button === 0 && transformState.scale > 1) { // Left click drag when zoomed in
            transformState.isDragging = true;
            transformState.dragStart.x = event.clientX - transformState.translateX;
            transformState.dragStart.y = event.clientY - transformState.translateY;
        }
    });

    window.addEventListener('pointerup', () => {
        transformState.isDragging = false;
    });

    // Keyboard Zoom Controls (+ / = to zoom in, - to zoom out)
    window.addEventListener('keydown', (event) => {
        if (!isMouseInsideViewport) return;

        // Prevent standard page zoom when hovering output viewer
        if (event.key === '=' || event.key === '+' || event.key === '-') {
            event.preventDefault();
        }

        if (event.key === '=' || event.key === '+') {
            zoomAtPoint(1.25, currentMouseViewportPos.x, currentMouseViewportPos.y);
            triggerHoverCalculation(currentMouseViewportPos.x, currentMouseViewportPos.y);
        } else if (event.key === '-') {
            zoomAtPoint(0.8, currentMouseViewportPos.x, currentMouseViewportPos.y);
            triggerHoverCalculation(currentMouseViewportPos.x, currentMouseViewportPos.y);
        }
    });

    // UI Buttons Zoom Control
    document.getElementById('btnZoomIn')?.addEventListener('click', () => {
        zoomAtPoint(1.25, viewport.clientWidth / 2, viewport.clientHeight / 2);
    });
    document.getElementById('btnZoomOut')?.addEventListener('click', () => {
        zoomAtPoint(0.8, viewport.clientWidth / 2, viewport.clientHeight / 2);
    });
    document.getElementById('btnZoomReset')?.addEventListener('click', reloadVisualization);

    // Height Map Calculation logic with Zoom Matrix Inversion
    async function triggerHoverCalculation(viewportX, viewportY) {
        // Convert screen coordinates -> Un-transformed raw Image coordinates
        const mouseX = (viewportX - transformState.translateX) / transformState.scale;
        const mouseY = (viewportY - transformState.translateY) / transformState.scale;

        const rawJson = await loadVisualizationData();
        if (!rawJson || !rawJson.data) return;

        // Dynamically retrieve actual bin sizes from JSON!
        const totalAngles = rawJson.angle_bins;
        const totalRadial = rawJson.radial_bins;
        const grid = rawJson.data;

        // 1. Relative radial distance calculation
        const dx = mouseX - CENTER_X;
        const dy = mouseY - CENTER_Y;
        const radius = Math.hypot(dx, dy);

        // Check bounds
        if (radius < INNER_R || radius > OUTER_R) {
            document.getElementById("mhInfoCartesianPos").innerHTML = `Cartesian coordinates: <br><b>${mouseX.toFixed(1)}, ${mouseY.toFixed(1)} (Out of bounds)</b>`;
            return;
        }

        // 2. Polar Angle Computation
        let radians = Math.atan2(dx, -dy);
        if (radians < 0) {
            radians += 2 * Math.PI;
        }

        const degrees = radians * (180 / Math.PI);

        // 3. Polar JSON Matrix Lookups
        const angleIdx = Math.floor((degrees / 360) * totalAngles) % totalAngles;
        const normalizedR = (radius - INNER_R) / (OUTER_R - INNER_R);
        const radialIdx = Math.min(totalRadial - 1, Math.floor(normalizedR * totalRadial));

        if (grid && grid[angleIdx] && grid[angleIdx][radialIdx] !== undefined) {
            const height = grid[angleIdx][radialIdx];

            document.getElementById("mhInfoCartesianPos").innerHTML = `Cartesian coordinates: <br><b>${mouseX.toFixed(1)}, ${mouseY.toFixed(1)}</b>`;
            document.getElementById("mhInfoPolarPos").innerHTML = `Polar coordinates: <br><b>${degrees.toFixed(1)}°, ${radius.toFixed(1)}um</b>`;
            document.getElementById("mhInfoHeightVal").innerHTML = `Height value: <br><b>${height !== null ? height.toFixed(3) + "um" : 'null (NaN)'}</b>`;
            document.getElementById("mhInfoPolarBinPos").innerHTML = `Polar bin: <br><b>Angle - ${angleIdx}, Radius - ${radialIdx}</b>`;
        }
    }

    // Attach tracking calculation to move events
    canvas.addEventListener('pointermove', (event) => {
        const rect = viewport.getBoundingClientRect();
        const vpX = event.clientX - rect.left;
        const vpY = event.clientY - rect.top;
        triggerHoverCalculation(vpX, vpY);
    });









    // State variables
    let p1 = null, p2 = null, target = null;

    // Attach functions to global window object so HTML inline onclicks can see them
    window.setPoint = function(pt) { 
        target = pt; 
    };

    window.calculateHeightDelta = function() {
        if (p1 === null || p2 === null) return alert("Select both points first!");
        document.getElementById("mtsDeltaShow").textContent = Math.abs(p1 - p2).toFixed(3) + "um";
    };

    window.resetHeightDelta = function() {
        p1 = p2 = target = null;
        ["mtsSelectPoint1Show", "mtsSelectPoint2Show", "mtsDeltaShow"].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = "None";
        });
    };

    // Click listener reading directly from mhInfoHeightVal
    const overlayCanvas = document.getElementById('finalImageOverlay');
    if (overlayCanvas) {
        overlayCanvas.addEventListener('click', () => {
            if (!target) return;
            const match = document.getElementById("mhInfoHeightVal")?.innerText.match(/[\d.-]+/);
            if (!match) return alert("Hover over a valid point first!");
            
            const val = parseFloat(match[0]);
            if (target === 1) {
                p1 = val;
                document.getElementById("mtsSelectPoint1Show").textContent = val.toFixed(3) + "um";
            } else if (target === 2) {
                p2 = val;
                document.getElementById("mtsSelectPoint2Show").textContent = val.toFixed(3) + "um";
            }
            target = null;
        });
    }


});


async function cancelProcessing() {
    try {
        await window.pywebview.api.cancelProcessing();
        updateProgressBar(0);
        const text = document.getElementById("progressBarText");
    if (text) text.textContent = "Processing cancelled.";
    } catch (err) {
        console.error("Failed to send cancel signal:", err);
    }
} 