// Global flag to track API ready status
let isApiReady = false;

window.addEventListener('pywebviewready', () => {
    isApiReady = true;
    console.log("PyWebView API is ready.");
});

async function exportPDFReport() {
    if (!selectedFilePath) {
        alert("Please run a visualization first!");
        return;
    }

    // Guard check: Ensure API bridge is initialized
    if (!window.pywebview || !window.pywebview.api || typeof window.pywebview.api.generatePDFReport !== 'function') {
        alert("PyWebView API is not ready yet. Please wait a moment and try again.");
        return;
    }

    // Helper functions
    const getNumVal = (id, fallback) => {
        const el = document.getElementById(id);
        if (!el) return fallback;
        const val = parseFloat(el.value !== undefined ? el.value : el.textContent.replace(/[^\d.-]/g, ''));
        return isNaN(val) ? fallback : val;
    };

    const getBoolVal = (id) => {
        const el = document.getElementById(id);
        return el ? Boolean(el.checked) : false;
    };

    const payload = {
        source_file: document.getElementById("filePathInput")?.textContent || selectedFilePath || "N/A",
        inner_diameter_mm: getNumVal("innerDimInput", 230.0),
        outer_diameter_mm: getNumVal("outerDimInput", 320.0),
        angle_bins: getNumVal("angleBinsInput", 720),
        radial_bins: getNumVal("radialBinsInput", 900),
        start_scan_range: getNumVal("startScanInput", 0.0),
        end_scan_range: getNumVal("endScanInput", 360.0),
        flatness_adjust: getBoolVal("flatnessAdjustCheck"),
        null_filling: getBoolVal("nullFillingCheck"),
        radial_flattening: getBoolVal("radialFlatteningCheck"),
        reference_zeroing: getBoolVal("referenceZeroingCheck"),
        blur: getNumVal("blurInput", 0),
        q_low: getNumVal("qLowInput", 1.0),
        q_high: getNumVal("qHighInput", 99.0),
        stats: {
            max: getNumVal("siInfoMax", 0),
            min: getNumVal("siInfoMin", 0),
            mean: getNumVal("siInfoAverage", 0),
            sdv: getNumVal("siInfoStdv", 0),
            range: getNumVal("siInfoMax", 0) - getNumVal("siInfoMin", 0),
            inmUsed: getNumVal("innerDimInput", 230.0) / getNumVal("outerDimInput", 320.0)
        }
    };

    try {
        const response = await window.pywebview.api.generatePDFReport(payload);
        if (response.status === "success") {
            alert("PDF Report successfully generated: " + response.pdf_path);
        } else {
            alert("Failed to generate PDF: " + response.message);
        }
    } catch (err) {
        console.error("PDF generation call failed:", err);
        alert("An error occurred while requesting the PDF report.");
    }
}

window.addEventListener("DOMContentLoaded", () => {
    // Bind the export function to your UI export button if present
    const exportBtn = document.getElementById("btnGenReport");
    if (exportBtn) {
        exportBtn.addEventListener("click", exportPDFReport);
    }
});