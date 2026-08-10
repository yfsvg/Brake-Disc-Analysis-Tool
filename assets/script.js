let selectedFilePath = null;

async function selectNativeFile() {
    // Trigger backend native file browser
    const result = await window.pywebview.api.openFile();
    if (result && result.status === "success") {
        selectedFilePath = result.file_path;
        document.getElementById("filePathInput").textContent = selectedFilePath;
    }
}

// Function called directly by Python's evaluate_js
function updateProgressBar(percentage) {
    const fill = document.getElementById("progressBarFill");
    const text = document.getElementById("progressBarText");
    const clampedPercent = Math.min(100, Math.max(0, Math.round(percentage)));
    if (fill) fill.style.width = clampedPercent + "%";
    if (text) text.textContent = clampedPercent + "%";
}

async function runVisualization() {
    if (!selectedFilePath) {
        alert("Please select a file first!");
        return;
    }

    // Reset progress bar
    updateProgressBar(0);

    // FIX 1: Construct the payload object expected by processAndVisualize(kwargs)
    const payload = {
        file_path: selectedFilePath,
        angle_bins: parseInt(document.getElementById("angleBinsInput")?.value) || 720,
        radial_bins: parseInt(document.getElementById("radialBinsInput")?.value) || 900,
        inner_diameter_mm: parseFloat(document.getElementById("innerDiaInput")?.value) || 230.0,
        outer_diameter_mm: parseFloat(document.getElementById("outerDiaInput")?.value) || 330.0,
        q_low: parseFloat(document.getElementById("qLowInput")?.value) || 1.0,
        q_high: parseFloat(document.getElementById("qHighInput")?.value) || 99.0,

        
        scan_angle_deg: 360.0,
        remap_lines: true
    };

    try {
        // Trigger process in backend
        const response = await window.pywebview.api.processAndVisualize(payload);
        
        if (response.status === "success") {
            const outputArea = document.getElementById("otherOutputArea");
            if (outputArea) {
                outputArea.textContent = JSON.stringify(response.stats, null, 2);
            }
            
            // Optional: Update image preview if you have an <img> tag on your HTML
            const imgElement = document.getElementById("heatmapImagePreview");
            if (imgElement) {
                // Appending timestamp prevents browser image caching
                imgElement.src = response.saved_image + "?t=" + new Date().getTime();
            }
        } else {
            alert("Error: " + response.message);
        }
    } catch (err) {
        console.error("PyWebView call failed:", err);
        alert("Execution failed: " + err);
    }
}

// FIX 2: Replaced array with a Key-Value Map object for clean lookups by key
const helpIDList = {
    "inner_diameter": ["Inner Diameter", "Controls the size of the inner ring diameter in the visualization."],
    "outer_diameter": ["Outer Diameter", "Controls the size of the outer ring diameter in the visualization."],
    "angle_bins": ["Angle Bins", "Angle bins help message"],
    "radial_bins": ["Radial Bins", "Radial bins help message"],
    "clamping": ["Clamping", "Clamping help message"]
};

function showHelp(id) {
    if (helpIDList[id]) {
        document.getElementById("overlayBG").style.display = "flex";
        document.getElementById("helpType").textContent = helpIDList[id][0];
        document.getElementById("helpInfoText").textContent = helpIDList[id][1];
    }
}

window.addEventListener("DOMContentLoaded", () => {
    const overlay = document.getElementById("overlayBG");
    if (overlay) {
        overlay.addEventListener("click", () => {
            overlay.style.display = "none";
        });
    }
});