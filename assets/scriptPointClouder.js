



let is3DMode = false;
let threeScene, threeCamera, threeRenderer, threeControls, pointCloudMesh;
let animationFrameId = null;

document.addEventListener("DOMContentLoaded", () => {
    const toggleBtn = document.getElementById("switchVisualizationType");
    if (toggleBtn) {
        toggleBtn.addEventListener("click", toggleVisualizationMode);
    }
});

async function toggleVisualizationMode() {
    is3DMode = !is3DMode;

    const toggleBtn = document.getElementById("switchVisualizationType");
    const transformLayer = document.getElementById("transformLayer");
    const viewport = document.getElementById("imageViewport");
    
    const mouseHoverHeader = document.querySelector("#dataAnalysisDashboard div:nth-child(2) h3");
    const cartesianPos = document.getElementById("mhInfoCartesianPos");
    const polarPos = document.getElementById("mhInfoPolarPos");
    const polarBinPos = document.getElementById("mhInfoPolarBinPos");
    const heightVal = document.getElementById("mhInfoHeightVal");
    
    const elementsToToggle = [mouseHoverHeader, cartesianPos, polarPos, polarBinPos, heightVal];

    if (is3DMode) {
        if (toggleBtn) toggleBtn.textContent = "2D Polar Map";
        
        if (transformLayer) transformLayer.style.display = "none";
        elementsToToggle.forEach(el => { if (el) el.style.display = "none"; });

        await init3DPointCloud(viewport);

    } else {
        if (toggleBtn) toggleBtn.textContent = "3D Point Cloud";
        
        if (transformLayer) transformLayer.style.display = "block";
        elementsToToggle.forEach(el => { if (el) el.style.display = "block"; });

        destroy3DScene();
    }
}

// Helper function to calculate exact percentiles (equivalent to np.percentile in Python)
function getPercentile(flatSortedValues, percentile) {
    if (flatSortedValues.length === 0) return 0;
    const p = Math.max(0, Math.min(100, percentile));
    const index = (p / 100) * (flatSortedValues.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index - lower;
    
    if (upper >= flatSortedValues.length) return flatSortedValues[flatSortedValues.length - 1];
    return flatSortedValues[lower] * (1 - weight) + flatSortedValues[upper] * weight;
}

async function init3DPointCloud(container) {
    if (!container) return;

    threeScene = new THREE.Scene();
    threeScene.background = new THREE.Color(0x161A1E);

    const width = container.clientWidth;
    const height = container.clientHeight;
    threeCamera = new THREE.PerspectiveCamera(45, width / height, 0.1, 10000);
    threeCamera.position.set(0, -400, 400);

    threeRenderer = new THREE.WebGLRenderer({ antialias: true });
    threeRenderer.setSize(width, height);
    threeRenderer.setPixelRatio(window.devicePixelRatio);
    threeRenderer.domElement.id = "threeCanvas3D";
    threeRenderer.domElement.style.position = "absolute";
    threeRenderer.domElement.style.top = "0";
    threeRenderer.domElement.style.left = "0";
    threeRenderer.domElement.style.zIndex = "10";
    container.appendChild(threeRenderer.domElement);

    threeControls = new THREE.OrbitControls(threeCamera, threeRenderer.domElement);
    threeControls.enableDamping = true;
    threeControls.dampingFactor = 0.05;

    threeControls.minDistance = 0.1; 
    // Prevent zooming out into infinity
    threeControls.maxDistance = 5000;

    const axesHelper = new THREE.AxesHelper(100);
    threeScene.add(axesHelper);

    // Load visualization data from JSON
    const rawJson = await loadVisualizationData();
    if (rawJson && rawJson.data) {
        const totalAngles = rawJson.angle_bins;
        const totalRadial = rawJson.radial_bins;
        const grid = rawJson.data;

        // Retrieve percentile bounds from JSON payload (defaulting to 1.0 and 99.0 if absent)
        const qLow = rawJson.q_low !== undefined ? rawJson.q_low : 1.0;
        const qHigh = rawJson.q_high !== undefined ? rawJson.q_high : 99.0;

        const OUTER_R = parseFloat(document.getElementById("parameterOuterDiameter")?.value) / 2 || 160;
        const INNER_R = parseFloat(document.getElementById("parameterInnerDiameter")?.value) / 2 || 115;

        // 1. Extract all valid numeric points to compute q_low and q_high cutoffs
        const validValues = [];
        for (let a = 0; a < totalAngles; a++) {
            for (let r = 0; r < totalRadial; r++) {
                const val = grid[a]?.[r];
                if (val !== null && val !== undefined && !isNaN(val)) {
                    validValues.push(val);
                }
            }
        }

        // 2. Sort values & compute vMin and vMax bounds using q_low and q_high
        validValues.sort((a, b) => a - b);
        const vMin = getPercentile(validValues, qLow);
        const vMax = getPercentile(validValues, qHigh);

        console.log(`3D Scale Limits Applied -> q_low (${qLow}%): ${vMin.toFixed(3)}, q_high (${qHigh}%): ${vMax.toFixed(3)}`);

        const positions = [];
        const colors = [];
        const colorMap = new THREE.Color();

        // 3. Generate geometry and map colors based on the q_low / q_high threshold bounds
        for (let angleIdx = 0; angleIdx < totalAngles; angleIdx++) {
            const theta = (angleIdx / totalAngles) * Math.PI * 2;
            for (let radialIdx = 0; radialIdx < totalRadial; radialIdx++) {
                const hVal = grid[angleIdx]?.[radialIdx];
                if (hVal === null || hVal === undefined || isNaN(hVal)) continue;

                const normR = radialIdx / totalRadial;
                const radius = INNER_R + normR * (OUTER_R - INNER_R);

                const x = radius * Math.cos(theta);
                const y = radius * Math.sin(theta);
                const z = hVal; 

                positions.push(x, y, z);

                // Clamp point height color within [vMin, vMax] percentile range
                const clampedVal = Math.max(vMin, Math.min(vMax, hVal));
                const normH = vMax === vMin ? 0.5 : (clampedVal - vMin) / (vMax - vMin);

                // Blue = Minimum, Red = Maximum
                colorMap.setHSL((1 - normH) * 0.7, 1.0, 0.5);
                colors.push(colorMap.r, colorMap.g, colorMap.b);
            }
        }

        const geometry = new THREE.BufferGeometry();
        geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

        const material = new THREE.PointsMaterial({
            size: 1.0,
            vertexColors: true,
            sizeAttenuation: true
        });

        pointCloudMesh = new THREE.Points(geometry, material);
        threeScene.add(pointCloudMesh);

        // Center camera view around the geometry center
        geometry.computeBoundingSphere();
        const sphere = geometry.boundingSphere;
        if (sphere) {
            const center = sphere.center;
            const radiusGeo = sphere.radius;

            threeControls.target.set(center.x, center.y, center.z);
            
            const offset = radiusGeo * 2.5; 
            threeCamera.position.set(center.x + offset, center.y - offset, center.z + offset / 2);
            threeCamera.updateProjectionMatrix();
        }

    } else {
        console.warn("No visualization data available for 3D generation.");
    }

    function animate() {
        animationFrameId = requestAnimationFrame(animate);
        threeControls.update();
        threeRenderer.render(threeScene, threeCamera);
    }
    animate();
}

function destroy3DScene() {
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }

    const existingCanvas = document.getElementById("threeCanvas3D");
    if (existingCanvas) {
        existingCanvas.remove();
    }

    if (pointCloudMesh) {
        pointCloudMesh.geometry.dispose();
        pointCloudMesh.material.dispose();
        pointCloudMesh = null;
    }

    if (threeRenderer) {
        threeRenderer.dispose();
        threeRenderer = null;
    }

    threeScene = null;
    threeCamera = null;
    threeControls = null;
}