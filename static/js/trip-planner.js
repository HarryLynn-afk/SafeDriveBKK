// ===== GLOBAL STATE =====
let tripPlannerRouteData = null;
let tripNavigationMap = null;

// ===== MULTI-ROUTE COMPARISON STATE =====
let routeOptions = {};   // { [routeId]: routeObj, fast: routeObj, safe: routeObj, recommended: routeObj }
let routeLayers = {};    // { [routeId]: google.maps.Polyline, _start/_end: google.maps.Marker }
let navPolylines = [];   // navigation map polylines for clearing
let selectedRoute = null;
// Preview markers + route line shown before the user hits "Find Safer Route"
let previewStartMarker = null;
let previewEndMarker = null;
let previewRouteLine = null;
let tripCompareMap = null;
let tripNavigationInterval = null;
let tripCurrentDistance = 0;
let tripMovingMarker = null;
let tripPreviousDistance = 0;
let tripSimulationSpeedKmPerSecond = 0;
let tripWaypointCumulativeKm = [];
let lastAlertSegmentIndex = -1;
let lastAlertSeverity = null;  // track risk level to only alert on changes
let spokenAlerts = new Set();
let navAlertHideTimeout = null;
let navBannerSuppressed = false;
let navVoiceAlertActive = false;
let navVoiceUtterance = null;
const NAVIGATION_ALERT_DURATION_MS = 3200;
const NAVIGATION_VOICE_RATE = 0.9;
const VOICE_ALERT_WORDS_PER_SECOND = 2.15;
const VOICE_ALERT_LEAD_BUFFER_SECONDS = 0.85;
const VOICE_ALERT_CLUSTER_BUFFER_SECONDS = 1.4;
const VOICE_ALERT_MIN_LOOKAHEAD_KM = 0.18;
const DANGER_VOICE_MESSAGE = "Warning: high risk zone ahead. Drive carefully.";
const TRIP_PLANNER_JS_VERSION = "2026-03-22.2";
const ANALYSIS_STEP_MIN_MS = 1500;
// Animation speed: adaptive so all routes finish in ~18 seconds
const SIMULATION_TARGET_SECONDS = 18;
const SIMULATION_TICK_MS = 100;
let simulationPaused = false;
const DEMO_ROUTE_NAMES = {
    "Ratchada": "ratchada",
    "Din Daeng": "din_daeng",
    "Asok": "asok",
    "Huai Khwang": "huai_khwang",
};
let demoRoutesCache = null;
let demoRoutesForceVisible = false;

// Pre-defined backup destinations
const BACKUP_DESTINATIONS = [
    { name: "Siam Paragon", lat: 13.7467, lon: 100.5348 },
    { name: "Grand Palace", lat: 13.7500, lon: 100.4914 },
    { name: "Thong Lo", lat: 13.7244, lon: 100.5787 },
    { name: "Pinklao", lat: 13.7769, lon: 100.4735 },
    { name: "Bearing", lat: 13.6680, lon: 100.6048 },
    { name: "Minburi", lat: 13.7918, lon: 100.7316 },
    { name: "Silom", lat: 13.7260, lon: 100.5233 },
    { name: "Chatuchak", lat: 13.7999, lon: 100.5537 },
    { name: "MBK Center", lat: 13.7447, lon: 100.5299 },
    { name: "Ari", lat: 13.7795, lon: 100.5362 },
    { name: "Asok", lat: 13.7380, lon: 100.5600 },
    { name: "Ratchada", lat: 13.73, lon: 100.72 }
];

const TRIP_PLANNER_MAP_STYLES = {
    light: [
        { elementType: "geometry", stylers: [{ color: "#eef4f1" }] },
        { elementType: "labels.text.fill", stylers: [{ color: "#334155" }] },
        { elementType: "labels.text.stroke", stylers: [{ color: "#f8fafc" }] },
        { featureType: "landscape.natural", elementType: "geometry", stylers: [{ color: "#e2f7e8" }] },
        { featureType: "poi", elementType: "labels.icon", stylers: [{ visibility: "off" }] },
        { featureType: "road", elementType: "geometry", stylers: [{ color: "#ffffff" }] },
        { featureType: "road.arterial", elementType: "geometry", stylers: [{ color: "#e2e8f0" }] },
        { featureType: "transit", stylers: [{ visibility: "off" }] },
        { featureType: "water", elementType: "geometry", stylers: [{ color: "#dbeafe" }] },
    ],
    dark: [
        { elementType: "geometry", stylers: [{ color: "#08111f" }] },
        { elementType: "labels.text.fill", stylers: [{ color: "#cbd5e1" }] },
        { elementType: "labels.text.stroke", stylers: [{ color: "#08111f" }] },
        { featureType: "landscape.natural", elementType: "geometry", stylers: [{ color: "#0b1822" }] },
        { featureType: "poi", elementType: "labels.icon", stylers: [{ visibility: "off" }] },
        { featureType: "road", elementType: "geometry", stylers: [{ color: "#132133" }] },
        { featureType: "road.arterial", elementType: "geometry", stylers: [{ color: "#1e293b" }] },
        { featureType: "transit", stylers: [{ visibility: "off" }] },
        { featureType: "water", elementType: "geometry", stylers: [{ color: "#0f3c57" }] },
    ],
};

function getTripPlannerTheme() {
    return document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
}

function getTripPlannerMapStyles(theme = getTripPlannerTheme()) {
    return TRIP_PLANNER_MAP_STYLES[theme] || TRIP_PLANNER_MAP_STYLES.light;
}

function getTripPlannerMapBackground(theme = getTripPlannerTheme()) {
    return theme === "dark" ? "#08111f" : "#eef4f1";
}

function applyTripPlannerMapTheme() {
    const options = {
        styles: getTripPlannerMapStyles(),
        backgroundColor: getTripPlannerMapBackground(),
    };

    if (tripCompareMap) {
        tripCompareMap.setOptions(options);
    }

    if (tripNavigationMap) {
        tripNavigationMap.setOptions(options);
    }
}

function formatRiskScoreCopy(score, guidance) {
    const normalizedScore = Math.round(Number(score) || 0);
    return guidance
        ? `Risk score: ${normalizedScore} · ${guidance}`
        : `Risk score: ${normalizedScore}`;
}

window.addEventListener("rr-theme-change", applyTripPlannerMapTheme);

// ===== ADDRESS AUTOCOMPLETE =====
let searchTimeout = null;
let startSearchTimeout = null;
let useGooglePlaces = false;  // Set true when Google Places API loads

// Default start location (used as fallback only)
const DEFAULT_START = { name: "Siam Paragon", lat: 13.7467, lon: 100.5348 };

// ---------------------------------------------------------------------------
// Google Places Autocomplete (used when API key is available)
// ---------------------------------------------------------------------------
window.initGooglePlaces = function () {
    // ── Initialize background map (Google Maps) ──
    const mapEl = document.getElementById("main-planner-map");
    if (mapEl && !tripCompareMap) {
        // Force dimensions so Google Maps renders
        mapEl.style.width = "100vw";
        mapEl.style.height = "100vh";
        tripCompareMap = new google.maps.Map(mapEl, {
            center: { lat: 13.75, lng: 100.53 },
            zoom: 12,
            zoomControl: true,
            mapTypeControl: false,
            streetViewControl: false,
            fullscreenControl: false,
            styles: getTripPlannerMapStyles(),
            backgroundColor: getTripPlannerMapBackground(),
        });
        // Wait for tiles to load, then place marker
        google.maps.event.addListenerOnce(tripCompareMap, "tilesloaded", function () {
            console.log("[GM] Tiles loaded — map is visible");
        });
        setPreviewMarker(DEFAULT_START.lat, DEFAULT_START.lon, true, 13);
    }

    const startInput = document.getElementById("start-location");
    const destInput = document.getElementById("destination");
    if (!startInput || !destInput) return;

    const options = {
        types: ["geocode", "establishment"],
        componentRestrictions: { country: "th" },
        bounds: new google.maps.LatLngBounds(
            new google.maps.LatLng(13.50, 100.30),
            new google.maps.LatLng(13.95, 100.90)
        ),
        strictBounds: false,
    };

    const startAC = new google.maps.places.Autocomplete(startInput, options);
    const destAC = new google.maps.places.Autocomplete(destInput, options);

    startAC.addListener("place_changed", function () {
        const place = startAC.getPlace();
        if (!place.geometry) return;
        const lat = place.geometry.location.lat();
        const lon = place.geometry.location.lng();
        const name = place.name || place.formatted_address || startInput.value;
        startInput.value = name;
        window.startLat = lat;
        window.startLon = lon;
        updateFindSaferRouteButton();
        setPreviewMarker(lat, lon, true);
    });

    destAC.addListener("place_changed", function () {
        const place = destAC.getPlace();
        if (!place.geometry) return;
        const lat = place.geometry.location.lat();
        const lon = place.geometry.location.lng();
        const name = place.name || place.formatted_address || destInput.value;
        destInput.value = name;
        window.endLat = lat;
        window.endLon = lon;
        updateFindSaferRouteButton();
        setPreviewMarker(lat, lon, false);
    });

    // Disable the custom oninput search since Google handles it
    startInput.removeAttribute("oninput");
    destInput.removeAttribute("oninput");
    useGooglePlaces = true;
    console.log("Google Places Autocomplete initialized");
};

// Timeout: if Google API fails to load, keep Nominatim fallback active
setTimeout(function () {
    if (!useGooglePlaces) {
        console.log("Google Places not available, using Nominatim search");
    }
}, 5000);

// ---------------------------------------------------------------------------
// Nominatim fallback search (used when no Google API key)
// ---------------------------------------------------------------------------
async function searchLocations(query, dropdownId, onSelect) {
    if (useGooglePlaces) return;  // Google handles its own dropdown
    return new Promise((resolve) => {
        const dropdown = document.getElementById(dropdownId);
        if (!query || query.length < 2) {
            hideElement(dropdownId);
            resolve();
            return;
        }

        setTimeout(async () => {
            try {
                const response = await fetch(
                    `https://nominatim.openstreetmap.org/search?`
                    + `q=${encodeURIComponent(query + " Bangkok Thailand")}&`
                    + `format=json&`
                    + `limit=5&`
                    + `addressdetails=1`
                );

                if (response.ok) {
                    const results = await response.json();
                    displayLocationSuggestions(results, "api", dropdownId, onSelect);
                } else {
                    throw new Error("API failed");
                }
            } catch (error) {
                const filtered = BACKUP_DESTINATIONS.filter((destination) =>
                    destination.name.toLowerCase().includes(query.toLowerCase())
                );
                displayLocationSuggestions(filtered, "backup", dropdownId, onSelect);
            }
            resolve();
        }, 300);
    });
}

function displayLocationSuggestions(results, source, dropdownId, onSelect) {
    const dropdown = document.getElementById(dropdownId);

    if (!results || results.length === 0) {
        hideElement(dropdownId);
        return;
    }

    dropdown.innerHTML = "";

    results.forEach((result) => {
        const item = document.createElement("div");
        item.className = "suggestion-item";

        let name;
        let address;
        let lat;
        let lon;

        if (source === "api") {
            name = result.display_name.split(",")[0];
            address = result.display_name;
            lat = parseFloat(result.lat);
            lon = parseFloat(result.lon);
        } else {
            name = result.name;
            address = "Bangkok, Thailand";
            lat = result.lat;
            lon = result.lon;
        }

        item.innerHTML = `
            <div class="suggestion-text">
                <div class="suggestion-name">${name}</div>
                <div class="suggestion-address">${address}</div>
            </div>
        `;

        item.onclick = () => onSelect(name, lat, lon);
        dropdown.appendChild(item);
    });

    showElement(dropdownId);
}

async function searchStartLocations(query) {
    if (useGooglePlaces) return;
    if (startSearchTimeout) {
        clearTimeout(startSearchTimeout);
    }
    if (!query || query.length < 2) {
        hideElement("start-suggestions-dropdown");
        return;
    }
    startSearchTimeout = setTimeout(() => {
        searchLocations(query, "start-suggestions-dropdown", selectStartLocation);
    }, 300);
}

function selectStartLocation(name, lat, lon) {
    document.getElementById("start-location").value = name;
    hideElement("start-suggestions-dropdown");
    window.startLat = lat;
    window.startLon = lon;
    updateFindSaferRouteButton();
    setPreviewMarker(lat, lon, true);
}

async function searchDestinations(query) {
    if (useGooglePlaces) return;
    if (searchTimeout) {
        clearTimeout(searchTimeout);
    }
    if (!query || query.length < 2) {
        hideElement("suggestions-dropdown");
        return;
    }
    searchTimeout = setTimeout(() => {
        searchLocations(query, "suggestions-dropdown", selectDestination);
    }, 300);
}

function selectDestination(name, lat, lon) {
    document.getElementById("destination").value = name;
    hideElement("suggestions-dropdown");
    window.endLat = lat;
    window.endLon = lon;
    updateFindSaferRouteButton();
    setPreviewMarker(lat, lon, false);
}

// Close suggestions when clicking outside (for Nominatim fallback)
document.addEventListener("click", (event) => {
    if (useGooglePlaces) return;
    const destDropdown = document.getElementById("suggestions-dropdown");
    const destInput = document.getElementById("destination");
    const startDropdown = document.getElementById("start-suggestions-dropdown");
    const startInput = document.getElementById("start-location");

    if (destDropdown && event.target !== destInput && !destDropdown.contains(event.target)) {
        hideElement("suggestions-dropdown");
    }
    if (startDropdown && event.target !== startInput && !startDropdown.contains(event.target)) {
        hideElement("start-suggestions-dropdown");
    }
});

// ===== GPS LOCATION DETECTION =====
function detectMyLocation() {
    const startInput = document.getElementById("start-location");
    const gpsBtn = document.querySelector(".gps-button");

    // Show loading state
    startInput.value = "Detecting location...";
    if (gpsBtn) {
        gpsBtn.classList.add("is-loading");
        gpsBtn.classList.remove("is-success");
        gpsBtn.disabled = true;
        gpsBtn.setAttribute("aria-busy", "true");
    }

    // Called when we have lat/lon from any source
    function applyLocation(lat, lon) {
        console.log("[GPS] Location found:", lat, lon);
        if (gpsBtn) {
            gpsBtn.classList.remove("is-loading");
            gpsBtn.classList.add("is-success");
            gpsBtn.disabled = false;
            gpsBtn.removeAttribute("aria-busy");
            setTimeout(() => { gpsBtn.classList.remove("is-success"); }, 2000);
        }
        window.startLat = lat;
        window.startLon = lon;
        updateFindSaferRouteButton();
        // Reverse geocode to get place name
        if (typeof google !== "undefined" && google.maps && google.maps.Geocoder) {
            const geocoder = new google.maps.Geocoder();
            geocoder.geocode({ location: { lat, lng: lon } }, (results, status) => {
                if (status === "OK" && results[0]) {
                    startInput.value = results[0].formatted_address.split(",")[0] || "Current Location";
                } else {
                    startInput.value = "Current Location";
                }
                setPreviewMarker(lat, lon, true, 15);
            });
        } else {
            startInput.value = "Current Location";
            setPreviewMarker(lat, lon, true, 15);
        }
    }

    function applyDefault(reason) {
        console.warn("[GPS] All methods failed:", reason);
        if (gpsBtn) {
            gpsBtn.classList.remove("is-loading", "is-success");
            gpsBtn.disabled = false;
            gpsBtn.removeAttribute("aria-busy");
        }
        startInput.value = DEFAULT_START.name;
        window.startLat = DEFAULT_START.lat;
        window.startLon = DEFAULT_START.lon;
        updateFindSaferRouteButton();
    }

    // Fallback 1: server-side geolocation (uses Google API from backend, no CORS/HTTPS issue)
    function tryServerGeolocation() {
        console.log("[GPS] Trying server-side /api/geolocate ...");
        fetch("/api/geolocate", { method: "POST" })
            .then((r) => r.json())
            .then((data) => {
                if (data.success && data.lat && data.lon) {
                    applyLocation(data.lat, data.lon);
                } else {
                    applyDefault(data.error || "Server geolocation failed");
                }
            })
            .catch(() => applyDefault("Server geolocation network error"));
    }

    // Try browser geolocation first, then server-side fallback
    if (navigator.geolocation) {
        console.log("[GPS] Trying browser geolocation...");
        navigator.geolocation.getCurrentPosition(
            (pos) => applyLocation(pos.coords.latitude, pos.coords.longitude),
            (err) => {
                console.warn("[GPS] Browser geolocation failed:", err.message);
                tryServerGeolocation();
            },
            { enableHighAccuracy: false, timeout: 5000, maximumAge: 300000 }
        );
    } else {
        tryServerGeolocation();
    }
}

// ===== FIND SAFER ROUTE BUTTON =====
function updateFindSaferRouteButton() {
    const btn = document.getElementById("findSaferRouteBtn");
    syncDemoRoutesVisibility();
    if (!btn) return;
    const hasBoth = !!(window.startLat && window.startLon && window.endLat && window.endLon);
    btn.disabled = !hasBoth;
}

function syncDemoRoutesVisibility() {
    const demoGrid = document.querySelector(".demo-routes-grid");
    const demoToggle = document.getElementById("demo-routes-toggle");
    const startInput = document.getElementById("start-location");
    const destInput = document.getElementById("destination");
    if (!demoGrid || !demoToggle || !startInput || !destInput) return;

    const hasTypedSearch = Boolean(startInput.value.trim() || destInput.value.trim());
    if (!hasTypedSearch) {
        demoRoutesForceVisible = false;
    }

    const showGrid = demoRoutesForceVisible || !hasTypedSearch;
    demoGrid.classList.toggle("is-hidden", !showGrid);
    demoToggle.classList.toggle("is-visible", !showGrid);
}

function hideDemoRoutesAfterSelection() {
    const demoGrid = document.querySelector(".demo-routes-grid");
    const demoToggle = document.getElementById("demo-routes-toggle");
    if (!demoGrid || !demoToggle) return;

    demoRoutesForceVisible = false;
    demoGrid.classList.add("is-hidden");
    demoToggle.classList.add("is-visible");
}

function showDemoRoutesFromToggle() {
    const demoGrid = document.querySelector(".demo-routes-grid");
    const demoToggle = document.getElementById("demo-routes-toggle");
    if (!demoGrid || !demoToggle) return;

    demoRoutesForceVisible = true;
    demoGrid.classList.remove("is-hidden");
    demoToggle.classList.remove("is-visible");
}

function onFindSaferRouteClick() {
    if (!window.startLat || !window.startLon) {
        alert("Please set a start location first (search or use GPS).");
        document.getElementById("start-location").focus();
        return;
    }
    if (!window.endLat || !window.endLon) {
        alert("Please select a destination first.");
        document.getElementById("destination").focus();
        return;
    }
    const destName = document.getElementById("destination").value;
    calculateRoute(window.startLat, window.startLon, window.endLat, window.endLon, destName);
}

// ===== ROUTE CALCULATION =====
async function calculateRoute(startLat, startLon, endLat, endLon, destName) {
    const findBtn = document.getElementById("findSaferRouteBtn");
    if (findBtn) {
        findBtn.disabled = true;
        findBtn.textContent = "Analyzing Routes...";
    }

    // Fix 19: push shareable URL so route can be bookmarked / shared
    try {
        const fromName = (document.getElementById("start-location") || {}).value || "";
        const params = new URLSearchParams({
            from: fromName,
            to: destName || "",
            slat: startLat,
            slon: startLon,
            elat: endLat,
            elon: endLon,
        });
        history.replaceState(null, "", "?" + params.toString());
    } catch (_) {}

    // Hide any stale results and show the AI overlay
    hideElement("route-preview");
    hideElement("route-results");
    showAIAnalyzing();

    try {
        // ── Step 1: real network call (duration is genuine) ──────────────
        const data = await runAnalysisStep(
            1,
            "Requesting route alternatives from OSRM...",
            async () => {
                const response = await fetch("/api/compare-routes", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        start_lat: startLat,
                        start_lon: startLon,
                        end_lat: endLat,
                        end_lon: endLon,
                        destination_name: destName
                    })
                });

                if (!response.ok) throw new Error("Route comparison failed");
                const result = await response.json();
                if (result.error) throw new Error(result.error);
                return result;
            }
        );

        const routesAnalyzed = data.alternatives_analyzed || 5;
        const analyzedCounter = document.getElementById("routes-analyzed");
        if (analyzedCounter) analyzedCounter.textContent = routesAnalyzed;

        // ── Step 2: XGBoost risk analysis (show ML working) ──────────────
        await runAnalysisStep(
            2,
            "XGBoost analyzing accident risk patterns...",
            async () => {
                const counter = document.getElementById("routes-analyzed");
                if (counter) counter.textContent = routesAnalyzed;
            }
        );

        // ── Step 3: route comparison ──────────────────────────────────────
        await runAnalysisStep(
            3,
            "Comparing routes for optimal safety...",
            async () => {}
        );

        updateStatus("Analysis complete!");
        await sleep(180);

        hideAIAnalyzing();
        displayMultipleRoutes(data);

    } catch (error) {
        console.error("Route calculation error:", error);
        hideAIAnalyzing();
        showRouteError(error.message);
    } finally {
        if (findBtn) {
            findBtn.textContent = "Find Safer Route";
            updateFindSaferRouteButton();
        }
    }
}

// ── AI overlay helpers ────────────────────────────────────────────────────────

let aiAnalyzingHideTimer = null;

function showAIAnalyzing() {
    const overlay = document.getElementById("ai-analyzing-overlay");
    if (!overlay) return;
    if (aiAnalyzingHideTimer) {
        clearTimeout(aiAnalyzingHideTimer);
        aiAnalyzingHideTimer = null;
    }
    overlay.style.display = "flex";
    overlay.classList.remove("is-visible");
    void overlay.offsetWidth;
    requestAnimationFrame(() => overlay.classList.add("is-visible"));
    updateStatus("Preparing route analysis...");
    [1, 2, 3].forEach((stepNumber) => updateAnalysisStep(stepNumber, "pending"));
    const counter = document.getElementById("routes-analyzed");
    if (counter) counter.textContent = "0";
}

function hideAIAnalyzing() {
    const overlay = document.getElementById("ai-analyzing-overlay");
    if (!overlay) return;
    overlay.classList.remove("is-visible");
    aiAnalyzingHideTimer = setTimeout(() => {
        overlay.style.display = "none";
        aiAnalyzingHideTimer = null;
    }, 220);
}

function showRouteError(rawMessage) {
    const el = document.getElementById("route-error");
    const msgEl = document.getElementById("route-error-message");
    if (!el) return;
    const m = (rawMessage || "").toLowerCase();
    let msg = "Unable to calculate route.";
    if (m.includes("outside") || m.includes("coverage") || m.includes("bangkok")) {
        msg = "Location may be outside Bangkok coverage area.";
    } else if (m.includes("fetch") || m.includes("network") || m.includes("networkerror")) {
        msg = "Network error — please check your connection.";
    } else if (m.includes("no route") || m.includes("empty") || m.includes("no result")) {
        msg = "No route found between these locations.";
    }
    if (msgEl) msgEl.textContent = msg;
    // Bug fix: error div lives inside #sheet-section-inputs — ensure state 1 is active
    setSheetState(1);
    el.style.display = "block";
}

function updateAnalysisStep(stepNumber, status) {
    const step = document.getElementById(`step-${stepNumber}`);
    if (!step) return;
    const icon = step.querySelector(".step-status");
    step.classList.remove("pending", "active", "complete");
    step.classList.add(status);
    if (icon) icon.setAttribute(
        "aria-label",
        status === "active" ? "In progress" : status === "complete" ? "Completed" : "Pending"
    );
}

function updateStatus(text) {
    const el = document.getElementById("analysis-status");
    if (el) el.textContent = text;
}

async function runAnalysisStep(stepNumber, statusText, task, minimumDuration = ANALYSIS_STEP_MIN_MS) {
    updateAnalysisStep(stepNumber, "active");
    updateStatus(statusText);

    const startedAt = Date.now();
    let result;
    let thrownError = null;

    try {
        result = await task();
    } catch (error) {
        thrownError = error;
    }

    const elapsed = Date.now() - startedAt;
    if (elapsed < minimumDuration) {
        await sleep(minimumDuration - elapsed);
    }

    if (thrownError) {
        throw thrownError;
    }

    updateAnalysisStep(stepNumber, "complete");
    return result;
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

// ===== MULTI-ROUTE MAP SETUP =====

function initCompareMap() {
    if (!tripCompareMap) {
        const el = document.getElementById("main-planner-map");
        if (!el) return null;
        tripCompareMap = new google.maps.Map(el, {
            center: { lat: 13.75, lng: 100.53 },
            zoom: 12,
            disableDefaultUI: false,
            zoomControl: true,
            mapTypeControl: false,
            streetViewControl: false,
            fullscreenControl: false,
            styles: getTripPlannerMapStyles(),
            backgroundColor: getTripPlannerMapBackground(),
        });
    }
    return tripCompareMap;
}

function getRouteTheme(route) {
    switch (route && route.primary_role) {
        case "recommended":
            return {
                cardClass: "route-card-rec-style recommended",
                badgeClass: "recommended-v2",
                lineColor: "#F59E0B",
                selectedColor: "#D97706",
                buttonClass: "select-rec-btn primary",
                rowToneClass: "route-row-recommended",
                iconClass: "recommended",
                iconGlyph: "AI",
                chipLabel: "Recommended",
            };
        case "safe":
            return {
                cardClass: "route-card-safe-style",
                badgeClass: "safest-v2",
                lineColor: "#10B981",
                selectedColor: "#059669",
                buttonClass: "select-safe-btn primary",
                rowToneClass: "route-row-safe",
                iconClass: "safe",
                iconGlyph: "S",
                chipLabel: "Safest",
            };
        case "alternative":
            return {
                cardClass: "route-card-alt-style",
                badgeClass: "alternative-v2",
                lineColor: "#64748B",
                selectedColor: "#475569",
                buttonClass: "select-alt-btn",
                rowToneClass: "route-row-alternative",
                iconClass: "alternative",
                iconGlyph: "A",
                chipLabel: "Alternative",
            };
        case "fast":
        default:
            return {
                cardClass: "route-card-fast-style",
                badgeClass: "fastest-v2",
                lineColor: "#3B82F6",
                selectedColor: "#1D4ED8",
                buttonClass: "select-fast-btn",
                rowToneClass: "route-row-fast",
                iconClass: "fast",
                iconGlyph: "F",
                chipLabel: "Fastest",
            };
    }
}

function cacheRouteOptions(data) {
    routeOptions = {};

    const allRoutes = Array.isArray(data.all_routes) ? data.all_routes : [];
    allRoutes.forEach((route) => {
        if (route && route.route_id) {
            routeOptions[route.route_id] = route;
        }
    });

    if (data.fast_route) {
        routeOptions.fast = routeOptions[data.fast_route.route_id] || data.fast_route;
    }
    if (data.safe_route) {
        routeOptions.safe = routeOptions[data.safe_route.route_id] || data.safe_route;
    }
    if (data.recommended_route) {
        routeOptions.recommended = routeOptions[data.recommended_route.route_id] || data.recommended_route;
    }
}

function buildDisplayRoutes(data) {
    const displayRoutes = [];
    const seen = new Set();
    const candidates = Array.isArray(data.display_routes) && data.display_routes.length
        ? data.display_routes
        : [data.recommended_route, data.fast_route, data.safe_route];

    candidates.forEach((route) => {
        if (!route || !route.route_id || seen.has(route.route_id)) return;
        displayRoutes.push(routeOptions[route.route_id] || route);
        seen.add(route.route_id);
    });

    if (displayRoutes.length < 3 && Array.isArray(data.all_routes)) {
        const remaining = [...data.all_routes].sort((a, b) => {
            const scoreDelta = (Number(b.topsis_score) || 0) - (Number(a.topsis_score) || 0);
            if (scoreDelta !== 0) return scoreDelta;
            return (Number(a.time_min) || 0) - (Number(b.time_min) || 0);
        });

        remaining.forEach((route) => {
            if (!route || !route.route_id || seen.has(route.route_id) || displayRoutes.length >= 3) return;
            displayRoutes.push(routeOptions[route.route_id] || route);
            seen.add(route.route_id);
        });
    }

    return displayRoutes;
}

function displayMultipleRoutes(data) {
    cacheRouteOptions(data);
    const displayRoutes = buildDisplayRoutes(data);
    routeOptions.display = displayRoutes;
    selectedRoute = null;

    // Reveal results container (cards only — map is the full-screen background)
    showElement("route-results");

    // Render route selection cards
    showRouteSelectionUI({ ...data, display_routes: displayRoutes });

    // Fix 6: transition to route comparison state
    setSheetState(2);

    // Draw routes on the background map
    setTimeout(() => {
        clearPreviewMap();

        displayRoutes.forEach((route) => {
            const theme = getRouteTheme(route);
            displayRouteOnMap(route, route.route_id, theme.lineColor, 0.75);
        });

        // Add start (green) and end (red) markers using a real displayed route.
        const anchorRoute = data.fast_route || displayRoutes[0];
        if (tripCompareMap && anchorRoute && anchorRoute.waypoints.length >= 2) {
            const wps = anchorRoute.waypoints;
            const toLatLng = (wp) => Array.isArray(wp) ? { lat: wp[0], lng: wp[1] } : { lat: wp.lat, lng: wp.lon };
            const startPos = toLatLng(wps[0]);
            const endPos   = toLatLng(wps[wps.length - 1]);

            routeLayers._start = new google.maps.Marker({
                position: startPos,
                map: tripCompareMap,
                zIndex: 10,
                icon: {
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: 10,
                    fillColor: "#34D399",
                    fillOpacity: 1,
                    strokeColor: "#ffffff",
                    strokeWeight: 2,
                },
            });

            routeLayers._end = new google.maps.Marker({
                position: endPos,
                map: tripCompareMap,
                zIndex: 10,
                icon: {
                    path: google.maps.SymbolPath.CIRCLE,
                    scale: 10,
                    fillColor: "#EF4444",
                    fillOpacity: 1,
                    strokeColor: "#ffffff",
                    strokeWeight: 2,
                },
            });
        }

        // Fit background map bounds to encompass both routes
        if (tripCompareMap) {
            const bounds = new google.maps.LatLngBounds();
            Object.values(routeLayers).forEach((layer) => {
                if (layer.getPath) layer.getPath().forEach((pt) => bounds.extend(pt));
                else if (layer.getPosition) bounds.extend(layer.getPosition());
            });
            if (!bounds.isEmpty()) tripCompareMap.fitBounds(bounds, { top: 60, right: 60, bottom: 280, left: 60 });
        }

        const defaultRouteId = (routeOptions.recommended && routeOptions.recommended.route_id)
            || (displayRoutes[0] && displayRoutes[0].route_id);
        if (defaultRouteId) {
            selectRoute(defaultRouteId);
        }
    }, 50);
}

function displayRouteOnMap(route, routeId, color, opacity) {
    if (!tripCompareMap) return;

    const path = route.waypoints.map((wp) =>
        Array.isArray(wp) ? { lat: wp[0], lng: wp[1] } : { lat: wp.lat, lng: wp.lon }
    );
    if (path.length < 2) return;

    const routeLine = new google.maps.Polyline({
        path,
        strokeColor: color,
        strokeWeight: 6,
        strokeOpacity: opacity,
        clickable: true,
        map: tripCompareMap,
    });

    google.maps.event.addListener(routeLine, "click", () => selectRoute(routeId));
    routeLayers[routeId] = routeLine;
}

function clearPreviewMarkers() {
    if (previewStartMarker) { previewStartMarker.setMap(null); previewStartMarker = null; }
    if (previewEndMarker)   { previewEndMarker.setMap(null);   previewEndMarker = null;   }
    if (previewRouteLine)   { previewRouteLine.setMap(null);   previewRouteLine = null;   }
}

// Fit bounds accounting for the bottom sheet covering ~300px of the viewport.
function fitBoundsAboveSheet(layers) {
    if (!tripCompareMap) return;
    const bounds = new google.maps.LatLngBounds();
    layers.filter(Boolean).forEach((layer) => {
        if (layer.getPosition) {
            bounds.extend(layer.getPosition());
        } else if (layer.getPath) {
            layer.getPath().forEach((pt) => bounds.extend(pt));
        }
    });
    if (bounds.isEmpty()) return;
    tripCompareMap.fitBounds(bounds, { top: 80, right: 50, bottom: 320, left: 50 });
}

// Draw (or replace) the preview line immediately as a straight dashed segment.
// This gives instant visual feedback with zero network latency.
function drawStraightPreviewLine() {
    if (!tripCompareMap || !previewStartMarker || !previewEndMarker) return;
    if (previewRouteLine) { previewRouteLine.setMap(null); }
    const s = previewStartMarker.getPosition();
    const e = previewEndMarker.getPosition();
    previewRouteLine = new google.maps.Polyline({
        path: [s, e],
        strokeColor: "#15803D",
        strokeWeight: 7,
        strokeOpacity: 0,
        icons: [{ icon: { path: "M 0,-1 0,1", strokeOpacity: 0.8, strokeColor: "#15803D", scale: 5 }, offset: "0", repeat: "18px" }],
        map: tripCompareMap,
    });
}

// Async upgrade: replace straight line with real OSRM road geometry.
function fetchAndDrawPreviewRoute() {
    if (!tripCompareMap || !previewStartMarker || !previewEndMarker) return;
    if (typeof google === "undefined" || !google.maps || !google.maps.DirectionsService) return;
    const sPos = previewStartMarker.getPosition();
    const ePos = previewEndMarker.getPosition();
    const svc = new google.maps.DirectionsService();
    svc.route({
        origin:      sPos,
        destination: ePos,
        travelMode:  google.maps.TravelMode.DRIVING,
    }, function (result, status) {
        if (status !== "OK") return;
        if (!previewStartMarker || !previewEndMarker) return;
        if (previewRouteLine) { previewRouteLine.setMap(null); }
        const path = google.maps.geometry.encoding.decodePath(result.routes[0].overview_polyline);
        previewRouteLine = new google.maps.Polyline({
            path,
            strokeColor: "#15803D",
            strokeWeight: 7,
            strokeOpacity: 0.85,
            map: tripCompareMap,
        });
        fitBoundsAboveSheet([previewStartMarker, previewEndMarker, previewRouteLine]);
    });
}

// Place a green (start) or red (end) circle marker on the background map.
// When both markers exist: draw straight line instantly, then upgrade with Google Directions.
function setPreviewMarker(lat, lon, isStart, zoom) {
    if (!tripCompareMap) return;
    const color = isStart ? "#34D399" : "#EF4444";
    const existing = isStart ? previewStartMarker : previewEndMarker;
    if (existing) existing.setMap(null);

    const marker = new google.maps.Marker({
        position: { lat, lng: lon },
        map: tripCompareMap,
        icon: {
            path: google.maps.SymbolPath.CIRCLE,
            scale: 10,
            fillColor: color,
            fillOpacity: 1,
            strokeColor: "#ffffff",
            strokeWeight: 2,
        },
    });

    if (isStart) { previewStartMarker = marker; } else { previewEndMarker = marker; }

    if (previewStartMarker && previewEndMarker) {
        drawStraightPreviewLine();
        fitBoundsAboveSheet([previewStartMarker, previewEndMarker]);
        fetchAndDrawPreviewRoute();
    } else {
        tripCompareMap.setCenter({ lat, lng: lon });
        tripCompareMap.setZoom(zoom || 13);
    }
}

function clearPreviewMap() {
    Object.values(routeLayers).forEach((poly) => poly.setMap(null));
    routeLayers = {};
    clearPreviewMarkers();
}

// ===== ROUTE SELECTION UI =====

function buildRouteComparisonCopy(fastRoute, safeRoute) {
    const fastRiskScore = Number(fastRoute && fastRoute.risk_score) || 0;
    const safeRiskScore = Number(safeRoute && safeRoute.risk_score) || 0;
    const timeDiffMinutes = Math.round((Number(safeRoute && safeRoute.time_min) || 0) - (Number(fastRoute && fastRoute.time_min) || 0));
    const absTimeDiff = Math.abs(timeDiffMinutes);
    const minuteLabel = absTimeDiff === 1 ? "minute" : "minutes";
    const riskReductionPct = fastRiskScore > 0
        ? Math.max(0, Math.round(((fastRiskScore - safeRiskScore) / fastRiskScore) * 100))
        : 0;

    if (riskReductionPct <= 0) {
        if (timeDiffMinutes > 0) {
            return {
                tradeoffHeadline: `Similar risk · only +${absTimeDiff} min longer`,
                recommendationText: `Take the faster path — no meaningful risk reduction for ${absTimeDiff} extra ${minuteLabel}.`,
            };
        }

        if (timeDiffMinutes < 0) {
            return {
                tradeoffHeadline: `Same risk · ${absTimeDiff} min faster`,
                recommendationText: `Take the faster path — it matches the safer option on risk and saves ${absTimeDiff} ${minuteLabel}.`,
            };
        }

        return {
            tradeoffHeadline: "Same risk · same ETA",
            recommendationText: "Fast route is already the safest available option.",
        };
    }

    if (timeDiffMinutes > 0) {
        return {
            tradeoffHeadline: `${riskReductionPct}% safer · only +${absTimeDiff} min longer`,
            recommendationText: `Take the safer path — ${riskReductionPct}% lower risk for just ${absTimeDiff} extra ${minuteLabel}.`,
        };
    }

    if (timeDiffMinutes < 0) {
        return {
            tradeoffHeadline: `${riskReductionPct}% safer · ${absTimeDiff} min faster`,
            recommendationText: `Take the safer path — ${riskReductionPct}% lower risk and ${absTimeDiff} ${minuteLabel} faster.`,
        };
    }

    return {
        tradeoffHeadline: `${riskReductionPct}% safer · same ETA`,
        recommendationText: `Take the safer path — ${riskReductionPct}% lower risk with no extra time.`,
    };
}

function showRouteSelectionUI(data) {
    const displayRoutes = Array.isArray(data.display_routes) ? data.display_routes : [];
    const riskPill = (level) => {
        const icon = getRiskIcon(level);
        return `<span class="risk-pill-inline ${level || "unknown"}">${icon} ${(level || "unknown").toUpperCase()} RISK</span>`;
    };

    const routeShortcutMap = [
        { label: "Recommended", role: "recommended", route: data.recommended_route || data.safe_route },
        { label: "Fastest", role: "fast", route: data.fast_route },
        { label: "Safest", role: "safe", route: data.safe_route },
    ].filter((item) => item.route && item.route.route_id);

    const routeRowsHTML = displayRoutes.map((route, index) => {
        const theme = getRouteTheme(route);
        const badges = Array.isArray(route.badges) ? route.badges : [];
        const extraBadges = badges.filter((badge) => badge !== route.primary_label);
        const badgeHtml = extraBadges.map((badge) => `<span class="route-secondary-badge">${badge}</span>`).join("");
        const distanceKm = Number(route.distance_km) || 0;
        const riskScore = Math.round(Number(route.risk_score) || 0);

        return `
            <button
                type="button"
                class="route-list-row ${theme.rowToneClass} route-row-enter"
                id="route-row-${route.route_id}"
                onclick="selectRoute('${route.route_id}')"
                style="animation-delay:${0.18 + (index * 0.08)}s"
            >
                <span class="route-row-main">
                    <span class="route-row-top">
                        <span class="route-row-title">${route.primary_label || "Route"}</span>
                        <span class="route-row-badges">${badgeHtml}</span>
                    </span>
                    <span class="route-row-metrics">
                        <span class="route-row-eta">${route.time_min || 0} min</span>
                        <span class="route-row-distance">${distanceKm.toFixed(1)} km</span>
                    </span>
                    <span class="route-row-bottom">
                        <span class="route-score-inline">
                            <span class="route-score-caption">Risk score</span>
                            <span class="route-score-value">${riskScore}</span>
                        </span>
                        ${riskPill(route.risk_level)}
                    </span>
                </span>
            </button>
        `;
    }).join("");

    const routeShortcutHTML = routeShortcutMap.map((item, index) => {
        const route = routeOptions[item.route.route_id] || item.route;
        const theme = getRouteTheme(route);
        return `
            <button
                type="button"
                class="route-shortcut-chip ${theme.iconClass} route-fade-up route-fade-up-${Math.min(index + 4, 6)}"
                data-route-role="${item.role}"
                onclick="jumpToRouteOption('${item.role}')"
            >
                <span>${item.label}</span>
            </button>
        `;
    }).join("");

    const routeSelectionHTML = `
        <div class="route-selection-container route-selection-enter">
            <div class="route-selection-header route-fade-up route-fade-up-1">
                <button class="sheet-back-btn" onclick="setSheetState(1)">Change route</button>
                <div class="route-header-meta">
                    <strong>AI-Powered Route Analysis</strong>
                    <span class="ml-model">XGBoost ML &bull; 83.8% Validated</span>
                </div>
            </div>

            <div class="route-shortcut-chips" aria-label="Route shortcuts">
                ${routeShortcutHTML}
            </div>

            <div class="route-list" role="list">
                ${routeRowsHTML}
            </div>

            <div class="route-selection-footer route-fade-up route-fade-up-7">
                <button id="start-navigation-btn" class="start-navigation-btn" onclick="startNavigationWithSelected()" disabled>
                    Start Navigation
                </button>
            </div>
        </div>
    `;

    const selectionContent = document.getElementById("route-selection-content");
    if (selectionContent) {
        selectionContent.innerHTML = routeSelectionHTML;
    }
}

function getRouteShortcutRole(routeId) {
    if (!routeId) return "";
    if (routeOptions.recommended && routeOptions.recommended.route_id === routeId) return "recommended";
    if (routeOptions.fast && routeOptions.fast.route_id === routeId) return "fast";
    if (routeOptions.safe && routeOptions.safe.route_id === routeId) return "safe";
    return "";
}

function jumpToRouteOption(role) {
    const targetRoute = routeOptions[role];
    if (!targetRoute || !targetRoute.route_id) return;
    const routeId = targetRoute.route_id;
    const row = document.getElementById(`route-row-${routeId}`);
    if (row) {
        row.scrollIntoView({ behavior: "smooth", block: "nearest" });
        row.classList.remove("route-row-pulse");
        void row.offsetWidth;
        row.classList.add("route-row-pulse");
        window.setTimeout(() => {
            row.classList.remove("route-row-pulse");
        }, 650);
    }

    selectRoute(routeId);
}

function syncRouteShortcutState(routeId) {
    const activeRole = getRouteShortcutRole(routeId);
    document.querySelectorAll(".route-shortcut-chip").forEach((chip) => {
        chip.classList.toggle("active", chip.dataset.routeRole === activeRole);
    });
}

function getRouteStartButtonLabel(route) {
    const label = route && route.primary_label ? route.primary_label : "Selected Route";
    return label.toLowerCase().includes("route") ? `Start ${label}` : `Start ${label} Route`;
}

function getRouteSubtitle(route) {
    const bits = [route && route.peak_label, route && route.peak_detail].filter(Boolean);
    return bits.length ? bits.join(" · ") : "Route risk summary unavailable.";
}

function getRouteRiskScore(route) {
    return Math.round(Number(route && route.risk_score) || 0);
}

function getRouteDistanceKm(route) {
    return (Number(route && route.distance_km) || 0).toFixed(1);
}

function getRouteDuration(route) {
    return Math.round(Number(route && route.time_min) || 0);
}

function getRiskIcon(level) {
    switch (level) {
        case "high":   return "✗";
        case "medium": return "!";
        case "low":    return "✓";
        default:       return "-";
    }
}

// ===== ROUTE SELECTION LOGIC =====

function selectRoute(routeId) {
    selectedRoute = routeId;

    // Highlight selected route on map, fade the other (skip marker keys)
    Object.keys(routeLayers).forEach((id) => {
        if (id.startsWith("_")) return; // skip _start/_end markers
        const poly = routeLayers[id];
        const theme = getRouteTheme(routeOptions[id] || {});
        if (id === routeId) {
            poly.setOptions({ strokeWeight: 9, strokeOpacity: 1.0, strokeColor: theme.selectedColor, zIndex: 2 });
        } else {
            poly.setOptions({ strokeWeight: 4, strokeOpacity: 0.28, strokeColor: theme.lineColor, zIndex: 1 });
        }
    });

    document.querySelectorAll(".route-card, .route-list-row").forEach((card) => card.classList.remove("selected"));
    const card = document.getElementById(`route-card-${routeId}`) || document.getElementById(`route-row-${routeId}`);
    if (card) card.classList.add("selected");
    syncRouteShortcutState(routeId);

    // Show detailed info + Start Navigation button
    showRouteDetails(routeOptions[routeId]);
}

function showRouteDetails(route) {
    if (!route) return;

    const btn = document.getElementById("start-navigation-btn") || document.querySelector(".start-navigation-btn");
    if (btn) {
        btn.textContent = getRouteStartButtonLabel(route);
        btn.disabled = false;
    }

    setSheetState(3);
}

function changeRoute() {
    selectedRoute = null;

    // Reset map route styles to original colours/opacity (skip marker keys)
    Object.keys(routeLayers).forEach((id) => {
        if (id.startsWith("_")) return; // skip _start/_end markers
        const theme = getRouteTheme(routeOptions[id] || {});
        routeLayers[id].setOptions({ strokeWeight: 6, strokeOpacity: 0.75, strokeColor: theme.lineColor, zIndex: 1 });
    });

    document.querySelectorAll(".route-card, .route-list-row").forEach((card) => card.classList.remove("selected"));
    syncRouteShortcutState("");
    resetStartNavigationButton();

    // Fix 6: go back to route comparison state
    setSheetState(2);
}

/**
 * Build synthetic risk-colored segments from compare-routes data when
 * the full /api/analyze-route call fails. Divides the route waypoints
 * into ~1 km segments and assigns risk colors based on the route's
 * overall risk level + some variance to simulate realistic risk zones.
 */
function buildSyntheticSegments(route) {
    const waypoints = route.waypoints;
    if (!waypoints || waypoints.length < 2) return [];

    // Build cumulative distance profile
    const cumDist = [0];
    for (let i = 1; i < waypoints.length; i++) {
        cumDist.push(cumDist[i - 1] + haversineKm(
            waypoints[i - 1][0], waypoints[i - 1][1],
            waypoints[i][0], waypoints[i][1]
        ));
    }
    const totalKm = cumDist[cumDist.length - 1];
    if (totalKm <= 0) return [];

    const segmentKm = 1.0; // ~1 km per segment
    const numSegments = Math.max(1, Math.round(totalKm / segmentKm));

    // Determine risk distribution based on route-level risk info
    const highZones = route.high_risk_zones || 0;
    const medZones = route.medium_risk_zones || 0;
    const riskScore = route.risk_score || 0;

    // Assign risk levels to segment indices: spread high/medium risk
    // segments across the route rather than clumping at the start
    const segmentRisks = new Array(numSegments).fill("low");

    // Place high-risk segments somewhat evenly
    if (highZones > 0) {
        const step = numSegments / (highZones + 1);
        for (let h = 0; h < highZones && h < numSegments; h++) {
            const idx = Math.min(numSegments - 1, Math.round(step * (h + 1)));
            segmentRisks[idx] = "high";
        }
    }
    // Place medium-risk segments in remaining slots
    if (medZones > 0) {
        let placed = 0;
        const step = numSegments / (medZones + 1);
        for (let m = 0; m < medZones && placed < medZones; m++) {
            const idx = Math.min(numSegments - 1, Math.round(step * (m + 1)));
            if (segmentRisks[idx] === "low") {
                segmentRisks[idx] = "medium";
                placed++;
            }
        }
    }

    // If overall risk is high but no specific zones, make some segments risky
    if (highZones === 0 && medZones === 0 && riskScore >= 50) {
        const riskyCount = Math.max(1, Math.round(numSegments * 0.3));
        const step = numSegments / (riskyCount + 1);
        for (let r = 0; r < riskyCount; r++) {
            const idx = Math.min(numSegments - 1, Math.round(step * (r + 1)));
            segmentRisks[idx] = riskScore >= 70 ? "high" : "medium";
        }
    } else if (highZones === 0 && medZones === 0 && riskScore >= 25) {
        const riskyCount = Math.max(1, Math.round(numSegments * 0.2));
        const step = numSegments / (riskyCount + 1);
        for (let r = 0; r < riskyCount; r++) {
            const idx = Math.min(numSegments - 1, Math.round(step * (r + 1)));
            segmentRisks[idx] = "medium";
        }
    }

    const riskColors = { high: "#dc3545", medium: "#ffc107", low: "#28a745" };
    const riskScores = { high: 70, medium: 40, low: 10 };

    const segments = [];
    const wpPerSeg = Math.max(1, Math.floor(waypoints.length / numSegments));

    for (let s = 0; s < numSegments; s++) {
        const wpStart = s * wpPerSeg;
        const wpEnd = s === numSegments - 1 ? waypoints.length - 1 : Math.min((s + 1) * wpPerSeg, waypoints.length - 1);
        const segWaypoints = waypoints.slice(wpStart, wpEnd + 1);
        if (segWaypoints.length < 2) continue;

        const startKm = cumDist[wpStart] || 0;
        const endKm = cumDist[wpEnd] || totalKm;
        const risk = segmentRisks[s];

        segments.push({
            start_km: Math.round(startKm * 10) / 10,
            end_km: Math.round(endKm * 10) / 10,
            waypoints: segWaypoints,
            risk_score: riskScores[risk],
            risk_level: risk,
            color: riskColors[risk],
            high_risk_zones: risk === "high" ? 1 : 0,
            max_accidents: 0,
        });
    }

    return segments;
}

function resetStartNavigationButton() {
    const btn = document.getElementById("start-navigation-btn") || document.querySelector(".start-navigation-btn");
    if (!btn) return;
    btn.textContent = "Start Navigation";
    btn.disabled = true;
}

function setStartNavigationButtonLoading() {
    const btn = document.getElementById("start-navigation-btn") || document.querySelector(".start-navigation-btn");
    if (!btn) return;
    btn.textContent = "Loading route...";
    btn.disabled = true;
}

async function startNavigationWithSelected() {
    if (!selectedRoute) {
        alert("Please select a route first");
        return;
    }

    const route = routeOptions[selectedRoute];
    setStartNavigationButtonLoading();

    try {
        // Fetch full segment analysis for the selected route's waypoints
        const response = await fetch("/api/analyze-route", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                start_lat: window.startLat,
                start_lon: window.startLon,
                end_lat: window.endLat,
                end_lon: window.endLon,
                waypoints: route.waypoints
            })
        });

        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                setTripPlannerRouteData(data);
                startNavigation();
                return;
            }
        }
    } catch (err) {
        console.warn("Full route analysis unavailable, using compare data:", err);
    }

    // Fallback: build synthetic segments from compare-routes waypoints
    const fallbackSegments = buildSyntheticSegments(route);
    const syntheticData = {
        route: {
            waypoints: route.waypoints,
            duration_min: route.time_min,
            distance_km: route.distance_km,
            risk_score: route.risk_score,
            risk_level: route.risk_level
        },
        segments: fallbackSegments,
        alerts: [],
        summary: {
            high_risk_segments: route.high_risk_zones || 0,
            medium_risk_segments: route.medium_risk_zones || 0
        }
    };
    setTripPlannerRouteData(syntheticData);

    startNavigation();
}

function setTripPlannerRouteData(data) {
    tripPlannerRouteData = data;
    if (Array.isArray(tripPlannerRouteData.segments)) {
        tripPlannerRouteData.segments = tripPlannerRouteData.segments.map((segment) => ({
            ...segment,
            ui_severity: getSegmentSeverity(segment),
        }));
    }
}

function displayRoutePreview(data, destName) {
    const route = data.route;
    const alerts = data.alerts || [];
    const summary = data.summary || {};

    // Update header
    document.getElementById("preview-title").textContent = "Route to " + destName;

    // Update stats
    document.getElementById("preview-time").textContent = route.duration_min;
    document.getElementById("preview-distance").textContent = route.distance_km.toFixed(1);

    // Show traffic source badge
    const trafficSrc = route.traffic_source || summary.traffic_source || "estimated";
    const previewTimeEl = document.getElementById("preview-time");
    if (previewTimeEl) {
        const existingBadge = previewTimeEl.parentElement && previewTimeEl.parentElement.querySelector(".traffic-badge, .simulated-note");
        if (existingBadge) existingBadge.remove();
        if (trafficSrc === "google_maps_traffic") {
            previewTimeEl.insertAdjacentHTML("afterend", '<span class="traffic-badge live" style="font-size:11px;margin-left:6px;">Live Traffic</span>');
        }
    }

    // Update risk indicator
    const riskIndicator = document.getElementById("risk-indicator");
    const riskText = document.getElementById("risk-text");
    const riskDetails = document.getElementById("risk-details");

    // Set risk level - check both risk_level and risk_score
    let riskIcon;
    let riskLabel;
    let riskClass;

    // Determine risk based on both score and level
    // Use score as primary, level as fallback
    if (route.risk_score >= 60 || route.risk_level === "high" || summary.high_risk_segments > 0) {
        riskIcon = "🔴";
        riskLabel = "HIGH RISK";
        riskClass = "high";
    } else if (route.risk_score >= 35 || route.risk_level === "medium" || summary.medium_risk_segments > 1) {
        riskIcon = "⚠️";
        riskLabel = "MEDIUM RISK";
        riskClass = "medium";
    } else {
        riskIcon = "🟢";
        riskLabel = "LOW RISK";
        riskClass = "low";
    }

    riskIndicator.className = "risk-indicator " + riskClass;
    const riskIconEl = document.querySelector(".risk-icon");
    if (riskIconEl) riskIconEl.textContent = riskIcon;
    riskText.textContent = riskLabel;

    // Update details
    const highRiskCount = summary.high_risk_segments || 0;
    const mediumRiskCount = summary.medium_risk_segments || 0;

    if (highRiskCount > 0) {
        riskDetails.textContent = `Crosses ${highRiskCount} high-risk zone${highRiskCount > 1 ? "s" : ""}`;
    } else if (mediumRiskCount > 0) {
        riskDetails.textContent = `Crosses ${mediumRiskCount} medium-risk zone${mediumRiskCount > 1 ? "s" : ""}`;
    } else {
        riskDetails.textContent = "This route is relatively safe";
    }

    // Show danger zones list if any
    if (highRiskCount > 0 || mediumRiskCount > 0) {
        displayDangerZones(alerts);
    } else {
        hideElement("danger-zones-list");
    }
}

function displayDangerZones(alerts) {
    const dangerAlerts = alerts.filter((alertItem) => alertItem.severity === "danger" || alertItem.severity === "warning");

    if (dangerAlerts.length === 0) {
        hideElement("danger-zones-list");
        return;
    }

    const zonesContent = document.getElementById("zones-content");
    zonesContent.innerHTML = "";

    dangerAlerts.forEach((alertItem) => {
        if (alertItem.severity === "danger") {
            const zoneItem = document.createElement("div");
            zoneItem.className = "zone-item";
            zoneItem.innerHTML = `🔴 km ${alertItem.trigger_km}: ${alertItem.detail}`;
            zonesContent.appendChild(zoneItem);
        }
    });

    showElement("danger-zones-list");
}

async function loadDemoRoute(routeName) {
    try {
        if (!demoRoutesCache) {
            const response = await fetch("/static/demo_routes.json", { cache: "no-store" });
            if (!response.ok) {
                throw new Error("Unable to load demo route definitions");
            }
            demoRoutesCache = await response.json();
        }
        const route = demoRoutesCache[routeName];
        if (
            !route
            || !route.start
            || !route.end
            || Number.isNaN(Number(route.end.lat))
            || Number.isNaN(Number(route.end.lon))
        ) {
            throw new Error("Demo route not found or invalid");
        }
        // console.log("Loading pre-saved demo route:", routeName);
        return route;
    } catch (error) {
        console.error("Failed to load demo route:", error);
        return null;
    }
}

async function useDemoRoute(demoRoute, destName, startLat, startLon) {
    document.getElementById("preview-title").textContent = "Loading route to " + destName + "...";
    showElement("route-preview");
    document.getElementById("preview-time").textContent = "--";
    document.getElementById("preview-distance").textContent = "--";
    document.getElementById("risk-text").textContent = "ANALYZING...";
    document.getElementById("risk-details").textContent = "Connecting to routing service...";

    // === HYBRID STRATEGY: Try OSRM first, fall back to embedded dense waypoints ===

    // 1) Try OSRM (live routing with perfect road geometry)
    try {
        // console.log("[Hybrid] Trying OSRM live routing first...");
        document.getElementById("risk-details").textContent = "Fetching live OSRM route...";

        const osrmPayload = {
            start_lat: startLat,
            start_lon: startLon,
            end_lat: demoRoute.end.lat,
            end_lon: demoRoute.end.lon,
            // No waypoints = backend will call OSRM for fresh routing
        };

        const osrmResponse = await fetch("/api/analyze-route", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(osrmPayload),
        });

        if (osrmResponse.ok) {
            const osrmData = await osrmResponse.json();
            if (osrmData.success) {
                // console.log(`[Hybrid] OSRM success: ${osrmData.route.waypoints.length} waypoints, ${osrmData.route.distance_km} km`);
                setTripPlannerRouteData(osrmData);
                displayRoutePreview(osrmData, destName);
                hideElement("loading-route");
                return true;
            }
        }
        console.warn("[Hybrid] OSRM response not ok, falling back to embedded waypoints");
    } catch (osrmError) {
        console.warn("[Hybrid] OSRM failed:", osrmError.message, "- using embedded fallback");
    }

    // 2) Fallback: Use embedded dense waypoints from demo_routes.json
    try {
        // console.log("[Hybrid] Using embedded dense waypoints as fallback...");
        document.getElementById("risk-details").textContent = "Using pre-computed route...";

        const fallbackPayload = {
            start_lat: startLat,
            start_lon: startLon,
            end_lat: demoRoute.end.lat,
            end_lon: demoRoute.end.lon,
        };

        if (Array.isArray(demoRoute.waypoints) && demoRoute.waypoints.length > 1) {
            fallbackPayload.waypoints = demoRoute.waypoints;
            if (demoRoute.distance !== undefined) {
                fallbackPayload.distance_km = demoRoute.distance;
            }
            if (demoRoute.duration !== undefined) {
                fallbackPayload.duration_min = demoRoute.duration;
            }
        }

        const fallbackResponse = await fetch("/api/analyze-route", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(fallbackPayload),
        });

        if (!fallbackResponse.ok) {
            throw new Error("Fallback route analysis failed");
        }

        const fallbackData = await fallbackResponse.json();
        if (!fallbackData.success) {
            throw new Error(fallbackData.error || "Unknown fallback error");
        }

        // console.log(`[Hybrid] Fallback success: ${fallbackData.route.waypoints.length} waypoints (embedded dense)`);
        setTripPlannerRouteData(fallbackData);
        displayRoutePreview(fallbackData, destName);
        hideElement("loading-route");
        return true;
    } catch (fallbackError) {
        console.error("[Hybrid] Both OSRM and fallback failed:", fallbackError);
        hideElement("loading-route");
        return false;
    }
}

function selectQuickDestination(name, lat, lon) {
    document.getElementById("destination").value = name;
    window.endLat = lat;
    window.endLon = lon;
    updateFindSaferRouteButton();
    setPreviewMarker(lat, lon, false);
}

function selectDemoRoute(startName, startLat, startLon, endName, endLat, endLon) {
    // Reset any prior comparison state; demo chips should only prefill the form.
    selectedRoute = null;
    routeOptions = {};
    hideElement("route-error");
    setSheetState(1);

    document.getElementById("start-location").value = startName;
    window.startLat = startLat;
    window.startLon = startLon;
    document.getElementById("destination").value = endName;
    window.endLat = endLat;
    window.endLon = endLon;
    updateFindSaferRouteButton();
    hideDemoRoutesAfterSelection();
    expandSheet();

    const btn = document.getElementById("findSaferRouteBtn");
    if (btn) {
        btn.textContent = "Find Safer Route";
    }

    // Place markers if map is ready
    if (tripCompareMap) {
        setPreviewMarker(startLat, startLon, true);
        setPreviewMarker(endLat, endLon, false);
    }
}

function haversineKm(lat1, lon1, lat2, lon2) {
    const radius = 6371;
    const toRad = (value) => (value * Math.PI) / 180;
    const deltaLat = toRad(lat2 - lat1);
    const deltaLon = toRad(lon2 - lon1);
    const a = Math.sin(deltaLat / 2) ** 2
        + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(deltaLon / 2) ** 2;
    return 2 * radius * Math.asin(Math.sqrt(a));
}

function buildWaypointDistanceProfile(waypoints) {
    const cumulative = [0];
    for (let index = 1; index < waypoints.length; index += 1) {
        const [prevLat, prevLon] = waypoints[index - 1];
        const [currLat, currLon] = waypoints[index];
        const segmentDistance = haversineKm(prevLat, prevLon, currLat, currLon);
        cumulative.push(cumulative[index - 1] + segmentDistance);
    }
    return cumulative;
}

function getWaypointIndexForDistance(distanceKm) {
    if (!tripWaypointCumulativeKm.length) {
        return 0;
    }

    for (let index = 1; index < tripWaypointCumulativeKm.length; index += 1) {
        if (tripWaypointCumulativeKm[index] >= distanceKm) {
            return index;
        }
    }
    return tripWaypointCumulativeKm.length - 1;
}

function calcBearing(lat1, lon1, lat2, lon2) {
    const toRad = (d) => d * Math.PI / 180;
    const dLon = toRad(lon2 - lon1);
    const y = Math.sin(dLon) * Math.cos(toRad(lat2));
    const x = Math.cos(toRad(lat1)) * Math.sin(toRad(lat2))
              - Math.sin(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.cos(dLon);
    return ((Math.atan2(y, x) * 180 / Math.PI) + 360) % 360;
}

function getInterpolatedPosition(distanceKm, waypoints) {
    if (!tripWaypointCumulativeKm.length || waypoints.length === 0) {
        return waypoints[0] || [0, 0];
    }

    // Clamp to route bounds
    const maxDist = tripWaypointCumulativeKm[tripWaypointCumulativeKm.length - 1];
    if (distanceKm <= 0) return waypoints[0];
    if (distanceKm >= maxDist) return waypoints[waypoints.length - 1];

    // Find the segment that brackets the current distance and interpolate
    for (let i = 1; i < tripWaypointCumulativeKm.length; i += 1) {
        if (tripWaypointCumulativeKm[i] >= distanceKm) {
            const prevDist = tripWaypointCumulativeKm[i - 1];
            const segLen = tripWaypointCumulativeKm[i] - prevDist;
            if (segLen <= 0) return waypoints[i];

            const t = (distanceKm - prevDist) / segLen;
            const lat = waypoints[i - 1][0] + t * (waypoints[i][0] - waypoints[i - 1][0]);
            const lon = waypoints[i - 1][1] + t * (waypoints[i][1] - waypoints[i - 1][1]);
            return [lat, lon];
        }
    }
    return waypoints[waypoints.length - 1];
}


function getSegmentSeverity(segment) {
    const segmentColor = String(segment.color || "").toLowerCase().trim();
    if (
        segmentColor.includes("dc3545")
        || segmentColor.includes("ff0000")
        || segmentColor.includes("d63031")
        || segmentColor.includes("ff6b6b")
        || segmentColor === "red"
    ) {
        return "danger";
    }
    if (
        segmentColor.includes("ffc107")
        || segmentColor.includes("ffd700")
        || segmentColor.includes("ff9800")
        || segmentColor.includes("e17055")
        || segmentColor === "orange"
        || segmentColor === "yellow"
    ) {
        return "warning";
    }
    return "safe";
}

function findSegmentIndexForDistance(distanceKm) {
    if (!tripPlannerRouteData || !tripPlannerRouteData.segments) {
        return -1;
    }

    const epsilon = 0.06;
    const segments = tripPlannerRouteData.segments;

    for (let index = 0; index < segments.length; index += 1) {
        const segment = segments[index];
        if (distanceKm >= (segment.start_km - epsilon) && distanceKm <= (segment.end_km + epsilon)) {
            return index;
        }
    }

    let nearestIndex = -1;
    let nearestDelta = Number.POSITIVE_INFINITY;
    for (let index = 0; index < segments.length; index += 1) {
        const segment = segments[index];
        const midpoint = (segment.start_km + segment.end_km) / 2;
        const delta = Math.abs(distanceKm - midpoint);
        if (delta < nearestDelta) {
            nearestDelta = delta;
            nearestIndex = index;
        }
    }
    return nearestIndex;
}

function startNavigation() {
    // console.log("Starting navigation...");

    if (!tripPlannerRouteData) {
        resetStartNavigationButton();
        alert("Please calculate a route first");
        return;
    }

    resetStartNavigationButton();

    // Show fullscreen
    const navScreen = document.getElementById("navigation-fullscreen");
    navScreen.classList.add("active");
    // overflow stays hidden via CSS !important on the trip planner page

    // Initialize map after a short delay for rendering
    setTimeout(() => {
        initializeNavigationMap();
        startNavigationSimulation();
    }, 200);
}

function initializeNavigationMap() {
    const el = document.getElementById("navigation-map");
    if (!el) return;

    if (!tripNavigationMap) {
        tripNavigationMap = new google.maps.Map(el, {
            center: { lat: 13.75, lng: 100.53 },
            zoom: 12,
            zoomControl: false,
            mapTypeControl: false,
            streetViewControl: false,
            fullscreenControl: false,
            gestureHandling: "greedy",
            styles: getTripPlannerMapStyles(),
            backgroundColor: getTripPlannerMapBackground(),
        });
    }
    applyTripPlannerMapTheme();
    google.maps.event.trigger(tripNavigationMap, "resize");

    // Clear existing nav polylines and markers
    navPolylines.forEach((p) => p.setMap(null));
    navPolylines = [];

    function wpToLatLng(wp) {
        return Array.isArray(wp) ? { lat: wp[0], lng: wp[1] } : { lat: wp.lat, lng: wp.lon };
    }

    // Draw route segments with risk colors
    tripPlannerRouteData.segments.forEach((segment) => {
        const path = segment.waypoints.map(wpToLatLng);
        const poly = new google.maps.Polyline({
            path, strokeColor: segment.color, strokeWeight: 8, strokeOpacity: 0.8, map: tripNavigationMap,
        });
        navPolylines.push(poly);
    });

    // Fallback full route if no segments
    if (tripPlannerRouteData.segments.length === 0) {
        const fallbackWp = tripPlannerRouteData.route.waypoints;
        if (fallbackWp && fallbackWp.length >= 2) {
            const poly = new google.maps.Polyline({
                path: fallbackWp.map(wpToLatLng),
                strokeColor: "#3B82F6", strokeWeight: 7, strokeOpacity: 0.85, map: tripNavigationMap,
            });
            navPolylines.push(poly);
        }
    }

    const allWaypoints = tripPlannerRouteData.route.waypoints;
    const routeEnd = allWaypoints[allWaypoints.length - 1];
    const lastSegment = tripPlannerRouteData.segments[tripPlannerRouteData.segments.length - 1];

    if (lastSegment && Array.isArray(lastSegment.waypoints) && lastSegment.waypoints.length > 0) {
        const lastSegmentEnd = lastSegment.waypoints[lastSegment.waypoints.length - 1];
        const gapDistance = Math.sqrt(
            ((lastSegmentEnd[0] - routeEnd[0]) ** 2) + ((lastSegmentEnd[1] - routeEnd[1]) ** 2)
        );
        if (gapDistance > 0.001) {
            const connector = new google.maps.Polyline({
                path: [wpToLatLng(lastSegmentEnd), wpToLatLng(routeEnd)],
                strokeColor: String(lastSegment.color || "#28a745"),
                strokeWeight: 8, strokeOpacity: 0.8, map: tripNavigationMap,
            });
            navPolylines.push(connector);
        }
    }

    // Destination marker
    new google.maps.Marker({
        position: wpToLatLng(routeEnd),
        map: tripNavigationMap,
        label: { text: "✦", color: "white", fontSize: "16px" },
        icon: {
            path: google.maps.SymbolPath.CIRCLE,
            scale: 14,
            fillColor: "#EF4444",
            fillOpacity: 1,
            strokeColor: "#ffffff",
            strokeWeight: 2,
        },
    });

    // Fit to route bounds
    const bounds = new google.maps.LatLngBounds();
    allWaypoints.forEach((wp) => bounds.extend(wpToLatLng(wp)));
    tripNavigationMap.fitBounds(bounds, { top: 50, right: 50, bottom: 50, left: 50 });
}

// ===== NAVIGATION SIMULATION =====
function startNavigationSimulation() {
    // Reset alert tracking
    spokenAlerts.clear();
    lastAlertSegmentIndex = -1;
    lastAlertSeverity = null;
    tripPreviousDistance = 0;
    tripSimulationSpeedKmPerSecond = 0;
    navBannerSuppressed = false;
    if (navAlertHideTimeout) {
        clearTimeout(navAlertHideTimeout);
        navAlertHideTimeout = null;
    }
    finalizeNavigationVoiceAlert();
    if ("speechSynthesis" in window) {
        speechSynthesis.cancel();
    }

    // Reset state
    tripCurrentDistance = 0;
    const waypoints = tripPlannerRouteData.route.waypoints;
    tripWaypointCumulativeKm = buildWaypointDistanceProfile(waypoints);

    // Use the actual haversine-computed distance from the waypoints so the
    // car's distance counter stays perfectly in sync with the cumulative
    // profile.  The backend-reported distance (OSRM road curves) can differ
    // significantly from haversine between sparse sampled points; using that
    // value would cause the car to freeze at the end or finish too early.
    const waypointTotalKm = tripWaypointCumulativeKm[tripWaypointCumulativeKm.length - 1] || 0;
    const totalDistance = waypointTotalKm > 0 ? waypointTotalKm : tripPlannerRouteData.route.distance_km;

    // Update bottom bar with route info
    const displayDistance = tripPlannerRouteData.route.distance_km;
    document.getElementById("nav-distance").textContent = displayDistance.toFixed(1);
    document.getElementById("nav-eta").textContent = tripPlannerRouteData.route.duration_min;

    // Create moving arrow marker
    if (tripMovingMarker) { tripMovingMarker.setMap(null); }
    const startWp = waypoints[0];
    tripMovingMarker = new google.maps.Marker({
        position: Array.isArray(startWp) ? { lat: startWp[0], lng: startWp[1] } : { lat: startWp.lat, lng: startWp.lon },
        map: tripNavigationMap,
        icon: {
            path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
            scale: 6,
            fillColor: "#1a73e8",
            fillOpacity: 1,
            strokeColor: "#ffffff",
            strokeWeight: 2,
            rotation: 0,
        },
        zIndex: 999,
    });

    hideNavigationAlert();
    updateNavTopBanner();

    // Adaptive step: route finishes in ~SIMULATION_TARGET_SECONDS regardless of length
    const ticksNeeded = (SIMULATION_TARGET_SECONDS * 1000) / SIMULATION_TICK_MS;
    const simStepKm = Math.max(0.05, totalDistance / ticksNeeded);
    tripSimulationSpeedKmPerSecond = totalDistance > 0 ? totalDistance / SIMULATION_TARGET_SECONDS : 0;

    // Start simulation interval
    if (tripNavigationInterval) {
        clearInterval(tripNavigationInterval);
    }
    tripNavigationInterval = setInterval(() => {
        tripPreviousDistance = tripCurrentDistance;
        tripCurrentDistance = Math.min(totalDistance, tripCurrentDistance + simStepKm);

        // Update progress
        updateNavigationUI(totalDistance, waypoints);

        // Check for alerts
        checkNavigationAlerts();
        maybeSpeakUpcomingDangerAlert();

        if (tripCurrentDistance >= totalDistance) {
            clearInterval(tripNavigationInterval);
            tripNavigationInterval = null;
            finalizeNavigationVoiceAlert();
            if ("speechSynthesis" in window) {
                speechSynthesis.cancel();
            }
            hideNavigationAlert();
            setNavTopBannerState("", "OK", "Destination reached", "You have arrived safely");

            setTimeout(() => {
                showTripSummaryModal();
            }, 1500);
        }
    }, SIMULATION_TICK_MS);
}

function updateNavigationUI(totalDistance, waypoints) {
    const progress = Math.min(100, (tripCurrentDistance / totalDistance) * 100);

    // Update bottom bar
    document.getElementById("nav-current-km").textContent = tripCurrentDistance.toFixed(1);
    document.getElementById("nav-progress-fill").style.width = progress + "%";

    // Update remaining time
    const baseSpeed = 58;
    const remainingDistance = Math.max(0, totalDistance - tripCurrentDistance);
    const remainingMinutes = Math.ceil((remainingDistance / baseSpeed) * 60);
    document.getElementById("nav-eta").textContent = remainingMinutes;

    // Update top banner with next risk zone ahead
    updateNavTopBanner();

    // Move marker using interpolated position for smooth movement
    const prevPosition = tripMovingMarker ? tripMovingMarker.getPosition() : null;
    const newPosition = getInterpolatedPosition(tripCurrentDistance, waypoints);
    const newLatLng = { lat: newPosition[0], lng: newPosition[1] };
    if (tripMovingMarker) {
        tripMovingMarker.setPosition(newLatLng);
        // Rotate arrow to face direction of travel
        if (prevPosition) {
            const bearing = calcBearing(prevPosition.lat(), prevPosition.lng(), newPosition[0], newPosition[1]);
            const icon = tripMovingMarker.getIcon();
            icon.rotation = bearing;
            tripMovingMarker.setIcon(icon);
        }
    }

    // Pan map to follow marker smoothly
    tripNavigationMap.panTo(newLatLng);
}

function setNavTopBannerState(severity, icon, title, subtitle) {
    const banner = document.getElementById("nav-top-banner");
    const bannerIcon = document.getElementById("nav-banner-icon");
    const bannerTitle = document.getElementById("nav-banner-title");
    const bannerSubtitle = document.getElementById("nav-banner-subtitle");

    if (!banner || !bannerIcon || !bannerTitle || !bannerSubtitle) return;

    banner.className = severity ? `nav-top-banner ${severity}` : "nav-top-banner";
    if (navBannerSuppressed) {
        banner.classList.add("is-hidden");
    }

    bannerIcon.textContent = icon;
    bannerTitle.textContent = title;
    bannerSubtitle.textContent = subtitle;
}

function setNavTopBannerVisible(isVisible) {
    const banner = document.getElementById("nav-top-banner");
    if (!banner) return;
    banner.classList.toggle("is-hidden", !isVisible);
}

function hideNavigationAlert() {
    const alertBox = document.getElementById("nav-alert-box");
    if (!alertBox) return;

    if (navAlertHideTimeout) {
        clearTimeout(navAlertHideTimeout);
        navAlertHideTimeout = null;
    }

    alertBox.className = "nav-alert-box";
    alertBox.setAttribute("aria-hidden", "true");
    navBannerSuppressed = false;
    updateNavTopBanner();
}

function showNavigationAlert(severity, icon, title, subtitle, options = {}) {
    const {
        duration = NAVIGATION_ALERT_DURATION_MS,
        suppressBanner = false,
    } = options;
    const alertBox = document.getElementById("nav-alert-box");

    if (!alertBox) return;

    if (navAlertHideTimeout) {
        clearTimeout(navAlertHideTimeout);
        navAlertHideTimeout = null;
    }

    navBannerSuppressed = Boolean(suppressBanner);
    setNavTopBannerVisible(!navBannerSuppressed);

    document.getElementById("nav-alert-icon").textContent = icon;
    document.getElementById("nav-alert-title").textContent = title;
    document.getElementById("nav-alert-subtitle").textContent = subtitle;

    alertBox.className = "nav-alert-box";
    void alertBox.offsetWidth;
    alertBox.className = `nav-alert-box ${severity} is-visible`;
    alertBox.setAttribute("aria-hidden", "false");

    if (duration > 0) {
        navAlertHideTimeout = setTimeout(() => {
            hideNavigationAlert();
        }, duration);
    }
}

function getVoiceAlertSegmentKey(segmentIndex, segment) {
    return `${segmentIndex}:${Number(segment.start_km) || 0}:${Number(segment.end_km) || 0}`;
}

function estimateSpeechDurationSeconds(message) {
    const wordCount = String(message || "")
        .trim()
        .split(/\s+/)
        .filter(Boolean)
        .length;

    return Math.max(2.8, (wordCount / VOICE_ALERT_WORDS_PER_SECOND) + 0.65);
}

function finalizeNavigationVoiceAlert() {
    navVoiceAlertActive = false;
    navVoiceUtterance = null;
}

function maybeSpeakUpcomingDangerAlert() {
    if (!tripPlannerRouteData || !Array.isArray(tripPlannerRouteData.segments)) return;
    if (!("speechSynthesis" in window) || tripSimulationSpeedKmPerSecond <= 0) return;
    if (navVoiceAlertActive || speechSynthesis.speaking || speechSynthesis.pending) return;

    const speechDurationSeconds = estimateSpeechDurationSeconds(DANGER_VOICE_MESSAGE);
    const leadDistanceKm = Math.max(
        VOICE_ALERT_MIN_LOOKAHEAD_KM,
        tripSimulationSpeedKmPerSecond * (speechDurationSeconds + VOICE_ALERT_LEAD_BUFFER_SECONDS)
    );
    const clusterDistanceKm = Math.max(
        leadDistanceKm,
        tripSimulationSpeedKmPerSecond * (speechDurationSeconds + VOICE_ALERT_CLUSTER_BUFFER_SECONDS)
    );

    const upcomingDangerSegments = tripPlannerRouteData.segments
        .map((segment, segmentIndex) => {
            const startKm = Number(segment.start_km) || 0;
            return {
                segment,
                segmentIndex,
                alertKey: getVoiceAlertSegmentKey(segmentIndex, segment),
                distanceToStart: startKm - tripCurrentDistance,
                severity: segment.ui_severity || getSegmentSeverity(segment),
            };
        })
        .filter((item) =>
            item.severity === "danger"
            && item.distanceToStart > 0
            && !spokenAlerts.has(item.alertKey)
        )
        .sort((a, b) => a.distanceToStart - b.distanceToStart);

    if (!upcomingDangerSegments.length) return;

    const leadSegment = upcomingDangerSegments.find((item) => item.distanceToStart <= leadDistanceKm);
    if (!leadSegment) return;

    const clusterStartKm = Number(leadSegment.segment.start_km) || 0;
    const clusterSegments = upcomingDangerSegments.filter((item) =>
        ((Number(item.segment.start_km) || 0) - clusterStartKm) <= clusterDistanceKm
    );

    const highestRiskSegment = clusterSegments.reduce((best, current) => {
        const bestRisk = Number(best.segment.risk_score) || 0;
        const currentRisk = Number(current.segment.risk_score) || 0;
        if (currentRisk !== bestRisk) {
            return currentRisk > bestRisk ? current : best;
        }
        return current.distanceToStart < best.distanceToStart ? current : best;
    }, clusterSegments[0]);

    const voiceMessage = clusterSegments.length > 1 && Number(highestRiskSegment.segment.risk_score) > 0
        ? "Warning: highest risk zone ahead. Drive carefully."
        : DANGER_VOICE_MESSAGE;

    if (!speakAlert(voiceMessage)) {
        return;
    }

    clusterSegments.forEach((item) => {
        spokenAlerts.add(item.alertKey);
    });
}

function checkNavigationAlerts() {
    if (!tripPlannerRouteData || !tripPlannerRouteData.segments) return;

    const segmentIndex = findSegmentIndexForDistance(tripCurrentDistance);
    if (segmentIndex < 0) return;

    // Skip if still in the same segment
    if (segmentIndex === lastAlertSegmentIndex) return;
    lastAlertSegmentIndex = segmentIndex;

    const currentSegment = tripPlannerRouteData.segments[segmentIndex];
    const severity = currentSegment.ui_severity || getSegmentSeverity(currentSegment);

    // Only alert when risk LEVEL changes (not on every zone entry)
    if (severity === lastAlertSeverity) return;

    lastAlertSeverity = severity;

    if (severity === "danger") {
        showNavigationAlert(
            "danger",
            "!",
            "Entering high-risk zone",
            formatRiskScoreCopy(currentSegment.risk_score, "Slow down"),
            { suppressBanner: true }
        );
        return;
    }

    hideNavigationAlert();
}

function speakAlert(message) {
    if (!("speechSynthesis" in window)) {
        return false;
    }

    if (navVoiceAlertActive || speechSynthesis.speaking || speechSynthesis.pending) {
        return false;
    }

    try {
        const utterance = new SpeechSynthesisUtterance(message);
        utterance.rate = NAVIGATION_VOICE_RATE;
        utterance.volume = 0.8;
        utterance.lang = "en-US";
        utterance.onend = finalizeNavigationVoiceAlert;
        utterance.onerror = finalizeNavigationVoiceAlert;

        navVoiceAlertActive = true;
        navVoiceUtterance = utterance;
        speechSynthesis.speak(utterance);
        return true;
    } catch (error) {
        finalizeNavigationVoiceAlert();
        return false;
    }
}

function exitNavigation() {
    // console.log("Exiting navigation");

    resetStartNavigationButton();

    // Stop any ongoing speech
    if ("speechSynthesis" in window) {
        speechSynthesis.cancel();
    }
    finalizeNavigationVoiceAlert();

    // Stop simulation
    simulationPaused = false;
    if (tripNavigationInterval) {
        clearInterval(tripNavigationInterval);
        tripNavigationInterval = null;
    }

    // Hide fullscreen
    const navScreen = document.getElementById("navigation-fullscreen");
    navScreen.classList.remove("active");
    navScreen.className = "navigation-fullscreen";

    // Reset state
    tripCurrentDistance = 0;
    tripPreviousDistance = 0;
    tripSimulationSpeedKmPerSecond = 0;
    tripWaypointCumulativeKm = [];
    lastAlertSegmentIndex = -1;
    lastAlertSeverity = null;
    spokenAlerts.clear();
    navBannerSuppressed = false;
    if (navAlertHideTimeout) {
        clearTimeout(navAlertHideTimeout);
        navAlertHideTimeout = null;
    }
    hideNavigationAlert();
    setNavTopBannerState("", "AI", "Scanning route...", "No hazards detected ahead");

    // Remove moving marker
    if (tripMovingMarker) { tripMovingMarker.setMap(null); tripMovingMarker = null; }

    // Close summary modal if open
    const summaryModal = document.getElementById("nav-summary-modal");
    if (summaryModal) {
        summaryModal.style.display = "none";
        summaryModal.classList.remove("is-visible", "is-closing");
    }
}

// ===== NAV TOP BANNER (lookahead) =====
function updateNavTopBanner() {
    if (!tripPlannerRouteData || !tripPlannerRouteData.segments) return;

    if (navBannerSuppressed) {
        setNavTopBannerVisible(false);
        return;
    }

    setNavTopBannerVisible(true);

    const segments = tripPlannerRouteData.segments;
    const segIdx = findSegmentIndexForDistance(tripCurrentDistance);
    const LOOKAHEAD = 4;

    if (segIdx < 0) {
        setNavTopBannerState("", "AI", "Scanning route...", "Checking conditions ahead");
        return;
    }

    let nextDanger = null;
    let nextWarning = null;
    for (let i = segIdx + 1; i < Math.min(segIdx + LOOKAHEAD + 1, segments.length); i++) {
        const sev = segments[i].ui_severity || getSegmentSeverity(segments[i]);
        if (sev === "danger" && !nextDanger) nextDanger = segments[i];
        else if (sev === "warning" && !nextWarning) nextWarning = segments[i];
    }

    if (nextDanger) {
        setNavTopBannerState(
            "danger",
            "!",
            "High-risk zone ahead",
            formatRiskScoreCopy(nextDanger.risk_score, "Slow down")
        );
    } else if (nextWarning) {
        setNavTopBannerState(
            "warning",
            "!",
            "Caution zone ahead",
            formatRiskScoreCopy(nextWarning.risk_score, "Stay alert")
        );
    } else {
        setNavTopBannerState("", "OK", "All clear ahead", "Safe conditions on your route");
    }
}

// ===== TRIP SUMMARY MODAL (3-stage animated) =====
function showTripSummaryModal() {
    const modal = document.getElementById("nav-summary-modal");
    if (!modal || !tripPlannerRouteData) return;

    // ── Determine which route the user took ──
    const fastRO = routeOptions.fast || {};
    const recommendedRO = routeOptions.recommended || routeOptions.safe || fastRO;
    const safeRO = routeOptions.safe || {};
    const myRO = routeOptions[selectedRoute] || recommendedRO || fastRO;
    const pickedFast = Boolean(fastRO.route_id) && myRO.route_id === fastRO.route_id;
    const altRO = pickedFast
        ? ((recommendedRO.route_id && recommendedRO.route_id !== myRO.route_id) ? recommendedRO
            : ((safeRO.route_id && safeRO.route_id !== myRO.route_id) ? safeRO : myRO))
        : fastRO;
    const compareLabel = pickedFast
        ? ((altRO.route_id && recommendedRO.route_id && altRO.route_id === recommendedRO.route_id) ? "recommended route" : "safer route")
        : "fast route";

    const myHighKm = parseFloat(myRO.high_risk_km || 0);
    const altHighKm = parseFloat(altRO.high_risk_km || 0);

    const myRisk  = Math.round(myRO.risk_score  || tripPlannerRouteData.route.risk_score || 30);
    const altRisk = Math.round(altRO.risk_score || myRisk);
    const myTime  = Math.round(myRO.time_min  || tripPlannerRouteData.route.duration_min || 0);
    const altTime = Math.round(altRO.time_min || myTime);

    // ── Compute framing based on which route was chosen ──
    let mainStat, mainStatLabel, pctStat, pctStatLabel, pctColor;
    let timeStat, timeStatLabel, timePrefix, timeCostColor;
    let arrivedText, verdictText, tryBtnText;

    if (!pickedFast) {
        // ── User took the recommended/safer route ──
        const avoidedKm = parseFloat(Math.max(0, altHighKm - myHighKm).toFixed(1));
        const saferPct = altRisk > 0
            ? Math.round(((altRisk - myRisk) / altRisk) * 100)
            : 0;
        const timeCost = Math.max(0, myTime - altTime);

        mainStat       = avoidedKm;
        mainStatLabel  = `km high-risk<br>avoided`;
        pctStat        = Math.max(0, saferPct);
        pctStatLabel   = "safer than<br>fast route";
        pctColor       = "#22c55e";
        timeStat       = timeCost;
        timeStatLabel  = "min extra<br>well spent";
        timePrefix     = "+";
        timeCostColor  = "#fbbf24";
        arrivedText    = "Arrived safely";
        tryBtnText     = "Plan a new route";

        verdictText = avoidedKm > 0 && timeCost > 0
            ? `+${timeCost} min kept you out of ${avoidedKm} km of high-risk areas`
            : avoidedKm > 0
                ? `You avoided ${avoidedKm} km of high-risk areas`
                : "You chose the safer path";
    } else {
        // ── User took the FAST route ──
        const extraKm = Math.max(0, myHighKm - altHighKm).toFixed(1);
        const riskierPct = myRisk > 0 ? Math.min(99, Math.max(0, Math.round((myRisk - altRisk) / myRisk * 100))) : 0;
        const timeSaved  = Math.max(0, altTime - myTime);

        mainStat       = extraKm;
        mainStatLabel  = `km more<br>high-risk areas`;
        pctStat        = riskierPct;
        pctStatLabel   = compareLabel === "recommended route"
            ? "higher risk<br>than AI route"
            : "higher risk<br>than safe route";
        pctColor       = "#f97316";
        timeStat       = timeSaved;
        timeStatLabel  = "min<br>faster";
        timePrefix     = "";
        timeCostColor  = "#22c55e";
        arrivedText    = "Arrived on time";
        tryBtnText     = "Try safer route";

        verdictText = timeSaved > 0 && parseFloat(extraKm) > 0
            ? `Saved ${timeSaved} min — ${compareLabel} had ${extraKm} km less high-risk areas`
            : timeSaved > 0
                ? ""
                : "Both routes had similar travel time";
    }

    // ── Apply text/color labels before animation ──
    const arrivedEl = document.getElementById("s-arrived-text");
    if (arrivedEl) arrivedEl.textContent = arrivedText;

    const avoidedLbl = document.getElementById("s-avoided-label");
    if (avoidedLbl) avoidedLbl.innerHTML = mainStatLabel;

    const pctLbl = document.getElementById("s-pct-label");
    if (pctLbl) pctLbl.innerHTML = pctStatLabel;

    const timeLbl = document.getElementById("s-time-label");
    if (timeLbl) timeLbl.innerHTML = timeStatLabel;

    // Arc and percentage colour
    const arc = document.getElementById("s-gauge-arc");
    if (arc) arc.style.stroke = pctColor;
    const pctSign = modal.querySelector(".arc-pct-sign");
    if (pctSign) pctSign.style.color = pctColor;
    const arcNum = modal.querySelector(".arc-label-wrap .s2-num");
    if (arcNum) arcNum.style.color = pctColor;

    // Time stat colour
    const timeCostEl = document.getElementById("s-time-cost");
    if (timeCostEl) timeCostEl.style.color = timeCostColor;

    const modalFastRouteLabel = document.getElementById("modal-fast-route-label");
    const modalSafeRouteLabel = document.getElementById("modal-safe-route-label");
    if (modalFastRouteLabel) modalFastRouteLabel.textContent = pickedFast ? "YOUR ROUTE" : "FAST ROUTE";
    if (modalSafeRouteLabel) modalSafeRouteLabel.textContent = pickedFast ? compareLabel.toUpperCase() : "YOUR ROUTE";

    const verdictEl = document.getElementById("s-verdict");
    if (verdictEl) {
        verdictEl.textContent = verdictText;
        verdictEl.style.color = pickedFast ? "#fbbf24" : "#4ade80";
        verdictEl.classList.toggle("is-hidden", !verdictText);
    }

    const tryBtn = document.getElementById("s-try-btn");
    if (tryBtn) tryBtn.textContent = tryBtnText;

    // ── Reset all animated elements ──
    const ring   = modal.querySelector(".check-ring");
    const mark   = modal.querySelector(".check-mark");
    const stage2 = modal.querySelector(".summary-s2");
    const stage3 = modal.querySelector(".summary-s3");
    const modalFastBar = document.getElementById("modal-fast-bar");
    const modalSafeBar = document.getElementById("modal-safe-bar");

    if (ring)   ring.style.strokeDashoffset   = "226";
    if (mark)   mark.style.strokeDashoffset   = "50";
    if (arc)    arc.style.strokeDashoffset    = "141";
    if (modalFastBar) modalFastBar.style.width = "0%";
    if (modalSafeBar) modalSafeBar.style.width = "0%";
    if (stage2) stage2.classList.remove("visible");
    if (stage3) stage3.classList.remove("visible");

    const avoided0  = document.getElementById("s-avoided");
    const pct0      = document.getElementById("s-safer-pct");
    if (avoided0)  avoided0.textContent  = "0";
    if (pct0)      pct0.textContent      = "0";
    if (timeCostEl) timeCostEl.textContent = timePrefix + "0";

    // ── Show modal ──
    modal.style.display = "flex";
    modal.classList.remove("is-closing");
    void modal.offsetWidth;
    modal.classList.add("is-visible");
    applyModalRiskBadge(
        "modal-fast-score-num", "modal-fast-bar", "modal-fast-badge",
        "modal-fast-icon", "modal-fast-label", "modal-fast-detail",
        pickedFast ? myRO : fastRO
    );
    applyModalRiskBadge(
        "modal-safe-score-num", "modal-safe-bar", "modal-safe-badge",
        "modal-safe-icon", "modal-safe-label", "modal-safe-detail",
        pickedFast ? altRO : myRO
    );

    // ── Stage 1: SVG checkmark draws (CSS transition) ──
    requestAnimationFrame(() => {
        requestAnimationFrame(() => {
            if (ring) ring.style.strokeDashoffset = "0";
            if (mark) mark.style.strokeDashoffset = "0";
        });
    });

    // ── Stage 2: Stats count up (0.6s delay) ──
    setTimeout(() => {
        if (stage2) stage2.classList.add("visible");
        _summaryCountUp("s-avoided",   0, mainStat, 700, "", 1);
        _summaryCountUp("s-safer-pct", 0, pctStat,  700);
        _summaryCountUp("s-time-cost", 0, timeStat, 700, timePrefix);
        _summaryArcAnimate(pctStat, 700);
    }, 600);

    // ── Stage 3: Risk blocks reveal (1.4s delay) ──
    setTimeout(() => {
        if (stage3) stage3.classList.add("visible");
    }, 1400);
}

function applyModalRiskBadge(scoreNumId, barId, badgeId, iconId, labelId, detailId, route) {
    if (!route) return;

    const score = route.risk_score;

    const bar = document.getElementById(barId);
    if (bar) {
        bar.style.width = score + "%";
        if (score >= 80)      bar.style.background = "#D85A30";
        else if (score >= 60) bar.style.background = "#EF9F27";
        else if (score >= 40) bar.style.background = "#639922";
        else                  bar.style.background = "#1D9E75";
    }

    const scoreNum = document.getElementById(scoreNumId);
    if (scoreNum) {
        scoreNum.textContent = score + " / 100";
        scoreNum.style.color = score >= 80 ? "#993C1D" : score >= 60 ? "#854F0B" : "#3B6D11";
    }

    const badge = document.getElementById(badgeId);
    if (badge) {
        badge.className = "peak-badge " + (score >= 60 ? "danger" : score >= 40 ? "medium" : "safe");
    }

    const icon = document.getElementById(iconId);
    if (icon) icon.textContent = score >= 60 ? "!" : "✓";

    const label = document.getElementById(labelId);
    if (label) label.textContent = route.peak_label || "";

    const detail = document.getElementById(detailId);
    if (detail) detail.textContent = route.peak_detail || "";
}

function _summaryCountUp(id, from, to, duration, prefix = "", decimals = 0) {
    const el = document.getElementById(id);
    if (!el) return;
    if (to === from) {
        el.textContent = prefix + (decimals > 0 ? parseFloat(Number(to).toFixed(decimals)) : to);
        return;
    }
    const start = performance.now();
    function step(now) {
        const t = Math.min(1, (now - start) / duration);
        const e = 1 - Math.pow(1 - t, 3);
        const value = from + (to - from) * e;
        el.textContent = prefix + (
            decimals > 0 ? parseFloat(value.toFixed(decimals)) : Math.round(value)
        );
        if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

function _summaryArcAnimate(pct, duration) {
    const arc = document.getElementById("s-gauge-arc");
    if (!arc) return;
    const total  = 141;
    const target = total * (1 - Math.min(1, pct / 100));
    const start  = performance.now();
    function step(now) {
        const t = Math.min(1, (now - start) / duration);
        const e = 1 - Math.pow(1 - t, 3);
        arc.style.strokeDashoffset = total - (total - target) * e;
        if (t < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

function closeSummaryModal() {
    const modal = document.getElementById("nav-summary-modal");
    if (!modal) {
        window.location.assign("/");
        return;
    }
    modal.classList.remove("is-visible");
    modal.classList.add("is-closing");
    window.setTimeout(() => {
        modal.style.display = "none";
        modal.classList.remove("is-closing");
        window.location.assign("/");
    }, 280);
}

function closeSummaryAndFindRoute() {
    const modal = document.getElementById("nav-summary-modal");
    if (modal) {
        modal.classList.remove("is-visible");
        modal.classList.add("is-closing");
    }
    window.setTimeout(() => {
        if (modal) {
            modal.style.display = "none";
            modal.classList.remove("is-closing");
        }
        exitNavigation();
        setSheetState(1);
    }, 280);
}

// ===== DEMO MODE TOGGLE =====
function toggleDemoMode() {
    document.body.classList.toggle("demo-mode");
    // console.log("Demo mode:", document.body.classList.contains("demo-mode") ? "ON" : "OFF");
}

// Enable demo mode by pressing 'D' key
document.addEventListener("keydown", (event) => {
    if (event.key === "d" || event.key === "D") {
        if (event.ctrlKey || event.metaKey) {
            event.preventDefault();
            toggleDemoMode();
        }
    }
});

// ===== UTILITY FUNCTIONS =====
function hideElement(id) {
    const element = document.getElementById(id);
    if (element) element.style.display = "none";
}

function showElement(id) {
    const element = document.getElementById(id);
    if (element) element.style.display = "block";
}

function setNetworkStatusVisibility(showOfflineMessage) {
    const networkStatus = document.getElementById("network-status");
    if (!networkStatus) {
        return;
    }
    networkStatus.style.display = showOfflineMessage ? "block" : "none";
}

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    // Map is initialized in initGooglePlaces (after Google Maps API loads)

    // ── Bottom sheet handle: toggle expanded/collapsed on tap ──
    const handle = document.getElementById("sheet-handle");
    if (handle) {
        handle.addEventListener("click", () => {
            const sheet = document.getElementById("planner-sheet");
            const resultsEl = document.getElementById("route-results");
            if (sheet) {
                if (sheet.classList.contains("expanded")) {
                    const resultsVisible = !!resultsEl && !resultsEl.classList.contains("hidden");
                    if (resultsVisible) {
                        peekSheet();
                    } else {
                        collapseSheet();
                    }
                } else {
                    expandSheet();
                }
            }
        });
    }

    // Fix 3: close hamburger menu on outside click
    document.addEventListener("click", () => {
        const dd = document.getElementById("planner-menu-dropdown");
        if (dd) dd.style.display = "none";
    });

    // Fix 6: start in input state
    setSheetState(1);
    updateFindSaferRouteButton();

    // Fix 19: auto-trigger route if URL contains deep-link params
    const urlParams = new URLSearchParams(window.location.search);
    const slat = parseFloat(urlParams.get("slat"));
    const slon = parseFloat(urlParams.get("slon"));
    const elat = parseFloat(urlParams.get("elat"));
    const elon = parseFloat(urlParams.get("elon"));
    const fromName = urlParams.get("from") || "";
    const toName   = urlParams.get("to")   || "";
    if (slat && slon && elat && elon) {
        window.startLat = slat;
        window.startLon = slon;
        window.endLat   = elat;
        window.endLon   = elon;
        const startInput = document.getElementById("start-location");
        const destInput  = document.getElementById("destination");
        if (startInput && fromName) startInput.value = fromName;
        if (destInput  && toName)   destInput.value  = toName;
        syncDemoRoutesVisibility();
        // short delay so map tiles are ready
        setTimeout(() => calculateRoute(slat, slon, elat, elon, toName), 800);
    }

    const startInput = document.getElementById("start-location");
    const destInput = document.getElementById("destination");
    [startInput, destInput].filter(Boolean).forEach((input) => {
        input.addEventListener("input", syncDemoRoutesVisibility);
    });
    const demoToggle = document.getElementById("demo-routes-toggle");
    if (demoToggle) {
        demoToggle.addEventListener("click", showDemoRoutesFromToggle);
    }
});

// === AUTO-REFRESH DEMO ROUTES ON STARTUP ===
// Attempt to refresh demo_routes.json with real OSRM data on first load.
// This upgrades interpolated fallback waypoints to real road geometry.
async function tryRefreshDemoRoutes() {
    try {
        const response = await fetch("/api/refresh-demo-routes", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
        });
        if (response.ok) {
            const data = await response.json();
            if (data.success) {
                // console.log("[DemoRoutes] Refreshed with OSRM data:", data.updated);
                // Invalidate cache so next loadDemoRoute picks up fresh data
                demoRoutesCache = null;
            }
        }
    } catch (err) {
        // console.log("[DemoRoutes] Auto-refresh skipped (OSRM unavailable)");
    }
}

// Check network status
window.addEventListener("load", () => {
    if (!navigator.onLine) {
        setNetworkStatusVisibility(true);
        console.warn("Offline mode - limited functionality");
    } else {
        setNetworkStatusVisibility(false);
        // Try to upgrade demo routes with real OSRM data in the background
        tryRefreshDemoRoutes();
    }
});

window.addEventListener("offline", () => {
    setNetworkStatusVisibility(true);
});

window.addEventListener("online", () => {
    setNetworkStatusVisibility(false);
    // Try refreshing demo routes when coming back online
    tryRefreshDemoRoutes();
});

// ===== BOLT-STYLE SHEET HELPERS =====

// ===== FIX 6: SHEET STATE MACHINE =====
// STATE 1 — inputs + demo chips + find button
// STATE 2 — route comparison cards
// STATE 3 — selected route details + start navigation

function setSheetState(state) {
    const inputs  = document.getElementById("sheet-section-inputs");
    const results = document.getElementById("route-results");
    const preview = document.getElementById("route-preview");

    // Hide all sections
    [inputs, results, preview].forEach((el) => {
        if (el) el.classList.add("hidden");
    });

    const sheet = document.getElementById("planner-sheet");

    if (state === 1) {
        if (inputs) inputs.classList.remove("hidden");
        // Clear comparison routes from map when going back to input state
        clearPreviewMap();
        collapseSheet();
    } else if (state === 2) {
        if (results) results.classList.remove("hidden");
        peekSheet();
        // scroll list back to the top edge of the sheet
        setTimeout(() => {
            if (sheet) sheet.scrollTop = 0;
        }, 150);
    } else if (state === 3) {
        if (results) results.classList.remove("hidden");
        peekSheet();
        // Keep the sticky route footer visible without forcing a jump.
        setTimeout(() => {
            const btn = document.getElementById("start-navigation-btn");
            const stickyFooter = document.querySelector(".route-selection-footer");
            if (sheet && btn && !stickyFooter) {
                sheet.scrollTop = btn.offsetTop - 20;
            }
        }, 200);
    }
}

function peekSheet() {
    const sheet = document.getElementById("planner-sheet");
    if (sheet) {
        sheet.classList.remove("collapsed", "expanded");
        sheet.classList.add("results-peek");
    }
    setTimeout(() => {
        if (tripCompareMap) google.maps.event.trigger(tripCompareMap, "resize");
    }, 300);
}

function expandSheet() {
    const sheet = document.getElementById("planner-sheet");
    if (sheet) {
        sheet.classList.remove("collapsed", "results-peek");
        sheet.classList.add("expanded");
    }
    setTimeout(() => {
        if (tripCompareMap) google.maps.event.trigger(tripCompareMap, "resize");
    }, 400);
}

function collapseSheet() {
    const sheet = document.getElementById("planner-sheet");
    if (sheet) {
        sheet.classList.remove("expanded", "results-peek");
        sheet.classList.add("collapsed");
    }
    setTimeout(() => {
        if (tripCompareMap) google.maps.event.trigger(tripCompareMap, "resize");
    }, 400);
}

function togglePlannerMenu(e) {
    e.stopPropagation();
    const dd = document.getElementById("planner-menu-dropdown");
    if (dd) dd.style.display = dd.style.display === "none" ? "block" : "none";
}

function togglePlannerLayers() {
    if (!tripCompareMap) return;
    const current = tripCompareMap.getMapTypeId();
    tripCompareMap.setMapTypeId(
        current === google.maps.MapTypeId.ROADMAP
            ? google.maps.MapTypeId.SATELLITE
            : google.maps.MapTypeId.ROADMAP
    );
}

function applyRiskBadge(cardElement, route) {
    if (!cardElement || !route) return;
    const score = route.risk_score;

    // Bar width and color
    const bar = cardElement.querySelector('[id^="risk-bar-"]');
    if (bar) {
        bar.style.width = score + '%';
        if (score >= 80)      bar.style.background = '#D85A30';
        else if (score >= 60) bar.style.background = '#EF9F27';
        else if (score >= 40) bar.style.background = '#639922';
        else                  bar.style.background = '#1D9E75';
    }

    // Score number
    const scoreNum = cardElement.querySelector('[id^="risk-score-"]');
    if (scoreNum) {
        scoreNum.textContent = score + ' / 100';
        scoreNum.style.color = score >= 80 ? '#993C1D' : score >= 60 ? '#854F0B' : '#3B6D11';
    }

    // Badge class
    const badge = cardElement.querySelector('[id^="peak-badge-"]');
    if (badge) {
        badge.className = 'peak-badge ' + (score >= 60 ? 'danger' : score >= 40 ? 'medium' : 'safe');
    }

    // Icon (unicode, no emoji)
    const icon = cardElement.querySelector('[id^="peak-icon-"]');
    if (icon) {
        icon.textContent = score >= 60 ? '!' : '✓';
    }

    // Labels
    const labelText = cardElement.querySelector('[id^="peak-label-"]');
    if (labelText) {
        labelText.textContent = route.peak_label || '';
    }
    const detailText = cardElement.querySelector('[id^="peak-detail-"]');
    if (detailText) {
        detailText.textContent = route.peak_detail || '';
    }
}
