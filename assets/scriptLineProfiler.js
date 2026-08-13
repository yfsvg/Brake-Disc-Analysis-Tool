// --- Line Profile Analysis State ---
let profileTarget = null; // Stores 1 or 2 to know which point is being set
let p1_coords = null;     // { x, y }
let p2_coords = null;     // { x, y }

// 1. Core Profile Sampling Logic (300 points)
function generateLineProfile(p1, p2, numSamples = 300) {
    if (!visualizationData || !visualizationData.data) {
        console.error("Visualization data not loaded yet.");
        return null;
    }

    // Fixed constants for your lab setup
    const CENTER_X = 257;
    const CENTER_Y = 267;
    const OUTER_R_PX = 217;

    const totalAngles = visualizationData.angle_bins;
    const totalRadial = visualizationData.radial_bins;
    const grid = visualizationData.data;

    // Fetch inner/outer diameters directly from loaded visualizationData (JSON)
    const innerDiameterMm = visualizationData.inner_diameter_mm;
    const outerDiameterMm = visualizationData.outer_diameter_mm;

    const innerToOuterRatio = innerDiameterMm / outerDiameterMm;
    const INNER_R_PX = OUTER_R_PX * innerToOuterRatio;

    const outerRadiusMm = outerDiameterMm / 2.0;

    // Line interpolation in Cartesian space (px)
    const x1 = p1.x;
    const y1 = p1.y;
    const x2 = p2.x;
    const y2 = p2.y;

    // Convert pixel distance to physical mm distance
    const pxToMm = outerRadiusMm / OUTER_R_PX;
    const totalLengthMm = Math.hypot(x2 - x1, y2 - y1) * pxToMm;

    const profileData = [];

    for (let i = 0; i < numSamples; i++) {
        const t = i / (numSamples - 1);
        const currX = x1 + (x2 - x1) * t;
        const currY = y1 + (y2 - y1) * t;

        const distanceAlongLineMm = t * totalLengthMm;

        // Convert sample point to polar relative to center
        const dx = currX - CENTER_X;
        const dy = currY - CENTER_Y;
        const rPx = Math.hypot(dx, dy);

        // Check if point falls out of disk bounds
        if (rPx < INNER_R_PX || rPx > OUTER_R_PX) {
            profileData.push({ distanceMm: distanceAlongLineMm, z: null });
            continue;
        }

        // Calculate angle (0° top, clockwise)
        let radians = Math.atan2(dx, -dy);
        if (radians < 0) radians += 2 * Math.PI;
        const degrees = radians * (180 / Math.PI);

        // Map to array indices in 2D grid
        const angleIdx = Math.floor((degrees / 360.0) * totalAngles) % totalAngles;
        const normalizedR = (rPx - INNER_R_PX) / (OUTER_R_PX - INNER_R_PX);
        const radialIdx = Math.min(totalRadial - 1, Math.floor(normalizedR * totalRadial));

        // Fetch height from matrix
        const height = grid[angleIdx]?.[radialIdx] ?? null;

        profileData.push({
            distanceMm: distanceAlongLineMm,
            z: height
        });
    }

    return { totalLengthMm, samples: profileData };
}








// 2. Render Line Profile onto your <canvas id="lineProfileOutputArea">
function drawLineProfileChart(canvasId, lineProfileResult) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !lineProfileResult) return;

    const ctx = canvas.getContext('2d');

    // 1. Handle Canvas Resizing without stretching (DPI scaling)
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    // Set internal render resolution matching display size
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    // Scale drawing context to match device pixel ratio
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    ctx.clearRect(0, 0, width, height);

    const data = lineProfileResult.samples;
    const validHeights = data.map(d => d.z).filter(z => z !== null && !isNaN(z));
    if (validHeights.length === 0) return;

    const minZ = Math.min(...validHeights);
    const maxZ = Math.max(...validHeights);
    const zRange = (maxZ - minZ) || 1.0;

    // 2. Margins and Margined Plot Dimensions
    const padY = height * 0.10; // 10% Top and Bottom margin
    const padLeft = 60;         // Margin for the Micron scale on the left
    const padRight = 20;

    const plotWidth = width - padLeft - padRight;
    const plotHeight = height - (padY * 2);

    // 3. Draw Left Scale Axis (Microns)
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.fillStyle = '#cccccc';
    ctx.font = '11px sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.lineWidth = 1;

    // Draw vertical axis line
    ctx.moveTo(padLeft, padY);
    ctx.lineTo(padLeft, height - padY);
    ctx.stroke();

    // Draw 5 tick marks and labels along the Y-axis
    const numTicks = 5;
    for (let i = 0; i < numTicks; i++) {
        const ratio = i / (numTicks - 1); // 0.0 to 1.0
        const tickY = height - padY - (ratio * plotHeight);
        const tickVal = minZ + (ratio * zRange);

        // Tick mark line
        ctx.beginPath();
        ctx.moveTo(padLeft - 5, tickY);
        ctx.lineTo(padLeft, tickY);
        ctx.stroke();

        // Tick label with micron units (µm)
        ctx.fillText(`${tickVal.toFixed(2)} µm`, padLeft - 8, tickY);
    }

    // 4. Draw Profile Line
    ctx.beginPath();
    ctx.strokeStyle = '#0571ff';
    ctx.lineWidth = 2;

    let drawing = false;
    data.forEach((point, i) => {
        if (point.z === null || isNaN(point.z)) {
            drawing = false;
            return;
        }

        const px = padLeft + (i / (data.length - 1)) * plotWidth;
        const py = height - padY - ((point.z - minZ) / zRange) * plotHeight;

        if (!drawing) {
            ctx.moveTo(px, py);
            drawing = true;
        } else {
            ctx.lineTo(px, py);
        }
    });

    ctx.stroke();

    // 5. Draw Plot Boundary
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.0)';
    ctx.strokeRect(padLeft, padY, plotWidth, plotHeight);
}










// 3. Setup HTML Event Listeners
window.addEventListener("DOMContentLoaded", () => {
    
    // Button setup
    document.getElementById("mtsLPASelectPoint1")?.addEventListener("click", () => {
        profileTarget = 1;
    });

    document.getElementById("mtsLPASelectPoint2")?.addEventListener("click", () => {
        profileTarget = 2;
    });

    document.getElementById("mtsLPACalculate")?.addEventListener("click", () => {
        if (!p1_coords || !p2_coords) return;
        const result = generateLineProfile(p1_coords, p2_coords, 300);
        drawLineProfileChart("lineProfileOutputArea", result);
    });

    document.getElementById("mtsLPARestartUnselect")?.addEventListener("click", () => {
        profileTarget = null;
        p1_coords = null;
        p2_coords = null;

        document.getElementById("mtsLPASelectPoint1Show").textContent = "None";
        document.getElementById("mtsLPASelectPoint2Show").textContent = "None";
        document.getElementById("mtsLPACalculate").disabled = true;

        // Clear canvas
        const canvas = document.getElementById("lineProfileOutputArea");
        if (canvas) {
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    });

    // Handle canvas clicks to capture coordinates
    const overlayCanvas = document.getElementById('finalImageOverlay');
    const viewport = document.getElementById('imageViewport');

    if (overlayCanvas && viewport) {
        overlayCanvas.addEventListener('click', (event) => {
            if (!profileTarget) return;

            const rect = viewport.getBoundingClientRect();
            const vpX = event.clientX - rect.left;
            const vpY = event.clientY - rect.top;

            // Invert scale & pan transforms to find raw un-scaled image coords
            const mouseX = (vpX - transformState.translateX) / transformState.scale;
            const mouseY = (vpY - transformState.translateY) / transformState.scale;

            if (profileTarget === 1) {
                p1_coords = { x: mouseX, y: mouseY };
                document.getElementById("mtsLPASelectPoint1Show").textContent = 
                    `(${mouseX.toFixed(1)}, ${mouseY.toFixed(1)})`;
            } else if (profileTarget === 2) {
                p2_coords = { x: mouseX, y: mouseY };
                document.getElementById("mtsLPASelectPoint2Show").textContent = 
                    `(${mouseX.toFixed(1)}, ${mouseY.toFixed(1)})`;
            }

            profileTarget = null;

            // Enable calculate button when both points are set
            if (p1_coords && p2_coords) {
                document.getElementById("mtsLPACalculate").disabled = false;
            }
        });
    }
});