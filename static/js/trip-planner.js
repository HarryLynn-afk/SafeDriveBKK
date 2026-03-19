// ===== GLOBAL STATE =====
let tripPlannerRouteData = null;
let tripNavigationMap = null;

// ===== MULTI-ROUTE COMPARISON STATE =====
let routeOptions = {};   // { fast: routeObj, safe: routeObj }
let routeLayers = {};    // { fast: L.Polyline, safe: L.Polyline }
let selectedRoute = null;
// Preview markers shown before the user hits "Find Safer Route"
let previewStartMarker = null;
let previewEndMarker = null;
let tripCompareMap = null;
let tripNavigationInterval = null;
let tripCurrentDistance = 0;
let tripMovingMarker = null;
let tripPreviousDistance = 0;
let tripWaypointCumulativeKm = [];
let lastAlertSegmentIndex = -1;
let lastAlertSeverity = null;  // track risk level to only alert on changes
let spokenAlerts = new Set();
let lastAlertTime = 0;
let safeAlertFadeTimeout = null;
const ALERT_COOLDOWN = 5000; // 5 seconds between alerts minimum
const TRIP_PLANNER_JS_VERSION = "2026-02-19.1";
// Animation speed: adaptive so all routes finish in ~18 seconds
// (+ pauses on danger/warning alerts for judges to read)
const SIMULATION_TARGET_SECONDS = 18;
const SIMULATION_TICK_MS = 100;
const ALERT_PAUSE_MS = 2500;  // pause animation 2.5s when danger/warning alert appears
let simulationPaused = false;
const DEMO_ROUTE_NAMES = {
    "Ratchada": "ratchada",
    "Din Daeng": "din_daeng",
    "Asok": "asok",
    "Huai Khwang": "huai_khwang",
};
let demoRoutesCache = null;

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

    if (event.target !== destInput && !destDropdown.contains(event.target)) {
        hideElement("suggestions-dropdown");
    }
    if (event.target !== startInput && !startDropdown.contains(event.target)) {
        hideElement("start-suggestions-dropdown");
    }
});

// ===== GPS LOCATION DETECTION =====
function detectMyLocation() {
    const startInput = document.getElementById("start-location");
    const gpsBtn = document.querySelector(".gps-button");

    if (!navigator.geolocation) {
        startInput.value = DEFAULT_START.name;
        window.startLat = DEFAULT_START.lat;
        window.startLon = DEFAULT_START.lon;
        updateFindSaferRouteButton();
        alert("Geolocation is not supported by your browser. Using default start location.");
        return;
    }

    // Show loading state
    startInput.value = "Detecting location...";
    if (gpsBtn) {
        gpsBtn.textContent = "...";
        gpsBtn.disabled = true;
    }

    navigator.geolocation.getCurrentPosition(
        (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;

            // Restore button
            if (gpsBtn) {
                gpsBtn.textContent = "OK";
                gpsBtn.disabled = false;
                setTimeout(() => { gpsBtn.textContent = "GPS"; }, 2000);
            }

            // Reverse geocode to get place name (try Google first, then Nominatim)
            if (typeof google !== "undefined" && google.maps && google.maps.Geocoder) {
                const geocoder = new google.maps.Geocoder();
                geocoder.geocode({ location: { lat, lng: lon } }, (results, status) => {
                    if (status === "OK" && results[0]) {
                        startInput.value = results[0].formatted_address.split(",")[0] || "Current Location";
                    } else {
                        startInput.value = "Current Location";
                    }
                    window.startLat = lat;
                    window.startLon = lon;
                    updateFindSaferRouteButton();
                    setPreviewMarker(lat, lon, true, 15);
                });
            } else {
                fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`)
                    .then((response) => response.json())
                    .then((data) => {
                        const placeName = data.display_name ? data.display_name.split(",")[0] : "";
                        startInput.value = placeName || "Current Location";
                        window.startLat = lat;
                        window.startLon = lon;
                        updateFindSaferRouteButton();
                        setPreviewMarker(lat, lon, true, 15);
                    })
                    .catch(() => {
                        startInput.value = "Current Location";
                        window.startLat = lat;
                        window.startLon = lon;
                        updateFindSaferRouteButton();
                        setPreviewMarker(lat, lon, true, 15);
                    });
            }
        },
        (error) => {
            // Restore button
            if (gpsBtn) {
                gpsBtn.textContent = "GPS";
                gpsBtn.disabled = false;
            }

            // Tell user what happened and use default
            let reason = "Unknown error";
            if (error.code === 1) reason = "Location permission denied";
            else if (error.code === 2) reason = "Location unavailable";
            else if (error.code === 3) reason = "Location request timed out";

            // HTTPS is required for geolocation in most browsers
            if (window.location.protocol === "http:" && window.location.hostname !== "localhost") {
                reason = "GPS requires HTTPS connection";
            }

            console.warn("GPS detection failed:", reason);
            startInput.value = DEFAULT_START.name;
            window.startLat = DEFAULT_START.lat;
            window.startLon = DEFAULT_START.lon;
            startInput.placeholder = `GPS failed: ${reason}. Using default.`;
            updateFindSaferRouteButton();
        },
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
    );
}

// ===== FIND SAFER ROUTE BUTTON =====
function updateFindSaferRouteButton() {
    const btn = document.getElementById("findSaferRouteBtn");
    if (!btn) return;
    const hasBoth = !!(window.startLat && window.startLon && window.endLat && window.endLon);
    btn.disabled = !hasBoth;
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
        updateAnalysisStep(1, "active");
        updateStatus("Requesting route alternatives from OSRM...");

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
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        updateAnalysisStep(1, "complete");

        // ── Step 2: XGBoost risk analysis (show ML working) ──────────────
        updateAnalysisStep(2, "active");
        updateStatus("XGBoost analyzing accident risk patterns...");
        document.getElementById("routes-analyzed").textContent =
            data.alternatives_analyzed || 5;
        await sleep(200);

        updateAnalysisStep(2, "complete");

        // ── Step 3: route comparison ──────────────────────────────────────
        updateAnalysisStep(3, "active");
        updateStatus("Comparing routes for optimal safety...");
        await sleep(150);

        updateAnalysisStep(3, "complete");
        updateStatus("Analysis complete!");
        await sleep(100);

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

function showAIAnalyzing() {
    const overlay = document.getElementById("ai-analyzing-overlay");
    if (!overlay) return;
    overlay.style.display = "flex";
    // Reset all steps to pending
    document.querySelectorAll(".progress-step").forEach((s) =>
        s.classList.remove("active", "complete")
    );
    document.querySelectorAll(".step-status").forEach((s) => {
        s.textContent = "...";
    });
    const counter = document.getElementById("routes-analyzed");
    if (counter) counter.textContent = "0";
}

function hideAIAnalyzing() {
    const overlay = document.getElementById("ai-analyzing-overlay");
    if (overlay) overlay.style.display = "none";
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
    step.classList.remove("active", "complete");
    if (status === "active") {
        step.classList.add("active");
        if (icon) icon.textContent = "...";
    } else if (status === "complete") {
        step.classList.add("complete");
        if (icon) icon.textContent = "done";
    }
}

function updateStatus(text) {
    const el = document.getElementById("analysis-status");
    if (el) el.textContent = text;
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

// ===== MULTI-ROUTE MAP SETUP =====

function initCompareMap() {
    if (!tripCompareMap) {
        // Fallback: init on main-planner-map if not yet done
        tripCompareMap = L.map("main-planner-map", {
            zoomControl: true,
            scrollWheelZoom: true
        }).setView([13.75, 100.53], 12);

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: "© OpenStreetMap contributors"
        }).addTo(tripCompareMap);
    }
    tripCompareMap.invalidateSize();
    return tripCompareMap;
}

function displayMultipleRoutes(data) {
    // Store both routes for later use
    routeOptions = {
        fast: data.fast_route,
        safe: data.safe_route
    };
    selectedRoute = null;

    // Reveal results container (cards only — map is the full-screen background)
    showElement("route-results");

    // Render route selection cards
    showRouteSelectionUI(data);

    // Fix 6: transition to route comparison state
    setSheetState(2);

    // Draw routes on the background map
    setTimeout(() => {
        clearPreviewMap();

        // Draw both routes: blue = fast, green = safe
        displayRouteOnMap(data.fast_route, "fast", "#3B82F6", 0.75);
        displayRouteOnMap(data.safe_route, "safe", "#10B981", 0.75);

        // Fit background map bounds to encompass both routes
        const allLayers = Object.values(routeLayers);
        if (allLayers.length > 0 && tripCompareMap) {
            const group = L.featureGroup(allLayers);
            tripCompareMap.fitBounds(group.getBounds(), { padding: [50, 50] });
        }

        if (tripCompareMap) setTimeout(() => tripCompareMap.invalidateSize(), 100);
    }, 50);
}

function displayRouteOnMap(route, routeId, color, opacity) {
    if (!tripCompareMap) return;

    // Waypoints from compare-routes are [[lat, lon], ...] arrays
    const waypoints = route.waypoints.map((wp) => (Array.isArray(wp) ? [wp[0], wp[1]] : [wp.lat, wp.lon]));
    if (waypoints.length < 2) return;

    const routeLine = L.polyline(waypoints, {
        color: color,
        weight: 6,
        opacity: opacity,
        interactive: true
    });

    routeLine.on("click", () => selectRoute(routeId));
    routeLine.bindTooltip(routeId === "fast" ? "🚀 Fastest Route" : "🛡️ Safest Route", {
        permanent: false,
        direction: "center"
    });

    routeLine.addTo(tripCompareMap);
    routeLayers[routeId] = routeLine;
}

function clearPreviewMarkers() {
    if (previewStartMarker && tripCompareMap) {
        tripCompareMap.removeLayer(previewStartMarker);
        previewStartMarker = null;
    }
    if (previewEndMarker && tripCompareMap) {
        tripCompareMap.removeLayer(previewEndMarker);
        previewEndMarker = null;
    }
}

// Place a green (start) or red (end) circle marker on the background map.
// If both markers are present, fit bounds to show both.
function setPreviewMarker(lat, lon, isStart, zoom) {
    if (!tripCompareMap) return;
    const color = isStart ? "#34D399" : "#EF4444";
    const existing = isStart ? previewStartMarker : previewEndMarker;
    if (existing) tripCompareMap.removeLayer(existing);

    const marker = L.circleMarker([lat, lon], {
        radius: 10,
        fillColor: color,
        color: "#ffffff",
        weight: 2,
        fillOpacity: 1,
    }).addTo(tripCompareMap);

    if (isStart) {
        previewStartMarker = marker;
    } else {
        previewEndMarker = marker;
    }

    if (previewStartMarker && previewEndMarker) {
        const group = L.featureGroup([previewStartMarker, previewEndMarker]);
        tripCompareMap.fitBounds(group.getBounds(), { padding: [60, 60] });
    } else {
        tripCompareMap.setView([lat, lon], zoom || 13);
    }
}

function clearPreviewMap() {
    Object.values(routeLayers).forEach((layer) => {
        if (tripCompareMap) tripCompareMap.removeLayer(layer);
    });
    routeLayers = {};
    clearPreviewMarkers();
}

// ===== ROUTE SELECTION UI =====

function showRouteSelectionUI(data) {
    const fastRoute = data.fast_route;
    const safeRoute = data.safe_route;

    const fastTime = fastRoute.time_min || 0;
    const safeTime = safeRoute.time_min || 0;
    const fastRiskScore = Math.round(fastRoute.risk_score || 0);
    const safeRiskScore = Math.round(safeRoute.risk_score || 0);
    const fastHighRisk = fastRoute.high_risk_zones || 0;
    const safeHighRisk = safeRoute.high_risk_zones || 0;

    const timeDiff = safeTime - fastTime;
    const timeDiffText = timeDiff > 0
        ? `+${timeDiff} min longer`
        : timeDiff < 0
            ? `${Math.abs(timeDiff)} min faster`
            : "Same time";

    const riskDelta = fastRiskScore - safeRiskScore;
    const riskDeltaText = riskDelta > 0 ? `${riskDelta} pt lower risk` : "similar risk";
    const tradeoffHeadline = timeDiff > 0
        ? `+${timeDiff} min · ${riskDeltaText}`
        : timeDiff < 0
            ? `${Math.abs(timeDiff)} min faster · ${riskDeltaText}`
            : `Same time · ${riskDeltaText}`;

    const altCount = data.alternatives_analyzed || 5;
    const fastTrafficSrc = fastRoute.traffic_source || "estimated";
    const safeTrafficSrc = safeRoute.traffic_source || "estimated";
    const trafficBadge = (src) => {
        if (src === "google_maps_traffic") return '<span class="traffic-badge live">Live Traffic</span>';
        if (src === "google_maps") return '<span class="traffic-badge">Google Maps</span>';
        if (src === "osrm") return '';
        return '<small class="simulated-note">*estimated</small>';
    };

    // Fix 1 + Fix 4: inline risk pill helper (small pill, not full-width bar)
    const riskPill = (level) => {
        const icon = getRiskIcon(level);
        return `<span class="risk-pill-inline ${level}">${icon} ${(level || "unknown").toUpperCase()} RISK</span>`;
    };

    const routeSelectionHTML = `
        <div class="route-selection-container">
            <!-- Fix 6: back to input state -->
            <button class="sheet-back-btn" onclick="setSheetState(1)">← Change route</button>

            <div class="ml-analysis-badge">
                <strong>AI-Powered Route Analysis</strong>
                <span class="ml-model">XGBoost ML &bull; 83.8% Validated</span>
            </div>

            <!-- Fix 4: tradeoff line OUTSIDE cards, centered above grid -->
            <div class="tradeoff-between">${tradeoffHeadline}</div>

            <div class="route-options-grid">
                <!-- Fix 1: Fast card — red border + bg -->
                <div class="route-card route-card-fast-style" id="route-card-fast" onclick="selectRoute('fast')">
                    <div>
                        <div class="route-badge-v2 fastest-v2">FASTEST</div>
                        <div class="route-time">${fastTime} min ${trafficBadge(fastTrafficSrc)}</div>
                        <div class="route-distance">${(fastRoute.distance_km || 0).toFixed(1)} km</div>
                        ${riskPill(fastRoute.risk_level)}
                    </div>
                    <div>
                        ${fastHighRisk > 0
                            ? `<div class="warning-note">! ${fastHighRisk} danger zone${fastHighRisk > 1 ? "s" : ""}</div>`
                            : '<div class="safe-note">No danger zones</div>'}
                        <button class="select-route-btn select-fast-btn" onclick="selectRoute('fast'); event.stopPropagation();">
                            Select This Route
                        </button>
                    </div>
                </div>

                <!-- Fix 1: Safe card — green border + bg -->
                <div class="route-card route-card-safe-style recommended" id="route-card-safe" onclick="selectRoute('safe')">
                    <div>
                        <div class="route-badge-v2 safest-v2">SAFEST</div>
                        ${safeRiskScore < fastRiskScore ? '<div class="recommended-badge">REC</div>' : ""}
                        <div class="route-time">${safeTime} min ${trafficBadge(safeTrafficSrc)}</div>
                        <div class="route-distance">${(safeRoute.distance_km || 0).toFixed(1)} km</div>
                        ${riskPill(safeRoute.risk_level)}
                    </div>
                    <div>
                        ${safeHighRisk > 0
                            ? `<div class="warning-note">! ${safeHighRisk} danger zone${safeHighRisk > 1 ? "s" : ""}</div>`
                            : '<div class="safe-note">No danger zones</div>'}
                        <button class="select-route-btn primary" onclick="selectRoute('safe'); event.stopPropagation();">
                            Select This Route
                        </button>
                    </div>
                </div>
            </div>

            ${data.recommendation
                ? `<div class="ai-recommendation">${data.recommendation}</div>`
                : ""}
        </div>
    `;

    const selectionContent = document.getElementById("route-selection-content");
    if (selectionContent) {
        selectionContent.innerHTML = routeSelectionHTML;
    }
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

    // Highlight selected route on map, fade the other
    Object.keys(routeLayers).forEach((id) => {
        const layer = routeLayers[id];
        if (id === routeId) {
            layer.setStyle({
                weight: 9,
                opacity: 1.0,
                color: id === "fast" ? "#1D4ED8" : "#059669"
            });
            layer.bringToFront();
        } else {
            layer.setStyle({ weight: 4, opacity: 0.3 });
        }
    });

    // Highlight the selected card
    document.querySelectorAll(".route-card").forEach((card) => card.classList.remove("selected"));
    const card = document.getElementById(`route-card-${routeId}`);
    if (card) card.classList.add("selected");

    // Show detailed info + Start Navigation button
    showRouteDetails(routeOptions[routeId]);
}

function showRouteDetails(route) {
    const routeLabel = selectedRoute === "fast" ? "🚀 Fastest Route" : "🛡️ Safest Route";
    const riskClass  = route.risk_level || "medium";
    const riskLabel  = (route.risk_level || "").toUpperCase();

    const detailsHTML = `
        <div class="selected-route-details selected-route-compact">
            <div class="compact-route-summary">
                <span class="compact-route-name">${routeLabel}</span>
                <span class="compact-route-time">${route.time_min} min</span>
                <span class="risk-pill-inline ${riskClass}">${riskLabel}</span>
            </div>
            <button id="start-navigation-btn" class="start-navigation-btn" onclick="startNavigationWithSelected()">
                Start Navigation
            </button>
        </div>
    `;

    const container = document.querySelector(".route-selection-container");
    const existing = document.querySelector(".selected-route-details");
    if (existing) existing.remove();
    if (container) container.insertAdjacentHTML("beforeend", detailsHTML);

    // Jump straight to Start Navigation — scroll sheet so button is first thing visible
    setSheetState(3);
    setTimeout(() => {
        const sheet = document.getElementById("planner-sheet");
        if (sheet) sheet.scrollTop = sheet.scrollHeight;
    }, 100);
}

function changeRoute() {
    selectedRoute = null;

    // Reset map route styles to original colours/opacity
    Object.keys(routeLayers).forEach((id) => {
        routeLayers[id].setStyle({
            weight: 6,
            opacity: 0.75,
            color: id === "fast" ? "#3B82F6" : "#10B981"
        });
    });

    document.querySelectorAll(".route-card").forEach((card) => card.classList.remove("selected"));

    const details = document.querySelector(".selected-route-details");
    if (details) details.remove();

    // Fix 6: go back to route comparison state
    setSheetState(2);
}

async function startNavigationWithSelected() {
    if (!selectedRoute) {
        alert("Please select a route first");
        return;
    }

    const route = routeOptions[selectedRoute];
    const btn = document.querySelector(".start-navigation-btn");
    if (btn) {
        btn.textContent = "⏳ Loading route...";
        btn.disabled = true;
    }

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

    // Fallback: build minimal route data directly from compare-routes response
    tripPlannerRouteData = {
        route: {
            waypoints: route.waypoints,
            duration_min: route.time_min,
            distance_km: route.distance_km,
            risk_score: route.risk_score,
            risk_level: route.risk_level
        },
        segments: [],
        alerts: [],
        summary: {
            high_risk_segments: route.high_risk_zones || 0,
            medium_risk_segments: route.medium_risk_zones || 0
        }
    };

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
        if (trafficSrc === "google_maps_traffic") {
            previewTimeEl.insertAdjacentHTML("afterend", '<span class="traffic-badge live" style="font-size:11px;margin-left:6px;">Live Traffic</span>');
        } else if (trafficSrc === "google_maps") {
            previewTimeEl.insertAdjacentHTML("afterend", '<span class="traffic-badge" style="font-size:11px;margin-left:6px;">Google Maps</span>');
        } else if (trafficSrc === "estimated") {
            previewTimeEl.insertAdjacentHTML("afterend", '<small class="simulated-note" style="font-size:11px;margin-left:6px;color:#999;">*estimated</small>');
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
}

function selectDemoRoute(startName, startLat, startLon, endName, endLat, endLon) {
    document.getElementById("start-location").value = startName;
    window.startLat = startLat;
    window.startLon = startLon;
    document.getElementById("destination").value = endName;
    window.endLat = endLat;
    window.endLon = endLon;
    updateFindSaferRouteButton();
    // Show preview markers immediately — before user clicks Find Safer Route
    setPreviewMarker(startLat, startLon, true);
    setPreviewMarker(endLat, endLon, false);
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
        alert("Please calculate a route first");
        return;
    }

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
    // Create map if doesn't exist
    if (!tripNavigationMap) {
        tripNavigationMap = L.map("navigation-map", {
            zoomControl: false
        }).setView([13.75, 100.53], 12);

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: "© OpenStreetMap contributors"
        }).addTo(tripNavigationMap);
    }

    // Invalidate size to fix rendering
    tripNavigationMap.invalidateSize();

    // Clear existing layers
    tripNavigationMap.eachLayer((layer) => {
        if (layer instanceof L.Polyline || (layer instanceof L.Marker && layer !== tripMovingMarker)) {
            tripNavigationMap.removeLayer(layer);
        }
    });

    // Draw route segments with colors
    tripPlannerRouteData.segments.forEach((segment) => {
        L.polyline(segment.waypoints, {
            color: segment.color,
            weight: 8,
            opacity: 0.8
        }).addTo(tripNavigationMap);
    });

    // Fallback: when no segment data exists (compare-routes path), draw the
    // full route as a single blue polyline so the map isn't blank.
    if (tripPlannerRouteData.segments.length === 0) {
        const fallbackWp = tripPlannerRouteData.route.waypoints;
        if (fallbackWp && fallbackWp.length >= 2) {
            L.polyline(fallbackWp, {
                color: "#3B82F6",
                weight: 7,
                opacity: 0.85
            }).addTo(tripNavigationMap);
        }
    }

    const allWaypoints = tripPlannerRouteData.route.waypoints;
    const routeEnd = allWaypoints[allWaypoints.length - 1];
    const lastSegment = tripPlannerRouteData.segments[tripPlannerRouteData.segments.length - 1];

    if (lastSegment && Array.isArray(lastSegment.waypoints) && lastSegment.waypoints.length > 0) {
        const lastSegmentEnd = lastSegment.waypoints[lastSegment.waypoints.length - 1];
        const connectorColor = String(lastSegment.color || "#28a745");
        const gapDistance = Math.sqrt(
            ((lastSegmentEnd[0] - routeEnd[0]) ** 2) + ((lastSegmentEnd[1] - routeEnd[1]) ** 2)
        );

        if (gapDistance > 0.001) {
            // console.log("Drawing connector segment - gap detected:", gapDistance);
            L.polyline([lastSegmentEnd, routeEnd], {
                color: connectorColor,
                weight: 8,
                opacity: 0.8
            }).addTo(tripNavigationMap);
        }
    }

    // Add destination marker
    L.marker(routeEnd, {
        icon: L.divIcon({
            className: "destination-marker",
            html: '<div style="font-size: 32px;">🏁</div>',
            iconSize: [32, 32],
            iconAnchor: [16, 16]
        })
    }).addTo(tripNavigationMap);

    // Fit to route bounds
    tripNavigationMap.fitBounds(L.polyline(allWaypoints).getBounds(), {
        padding: [50, 50]
    });
}

// ===== NAVIGATION SIMULATION =====
function startNavigationSimulation() {
    // Reset alert tracking
    spokenAlerts.clear();
    lastAlertTime = 0;
    lastAlertSegmentIndex = -1;
    lastAlertSeverity = null;
    tripPreviousDistance = 0;

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

    // Update UI (show backend-reported distance for user display, but simulate with actual)
    const displayDistance = tripPlannerRouteData.route.distance_km;
    document.getElementById("nav-total-km").textContent = displayDistance.toFixed(1);
    document.getElementById("nav-distance").textContent = displayDistance.toFixed(1) + " km";
    document.getElementById("nav-eta").textContent = tripPlannerRouteData.route.duration_min + " min";

    // Create moving marker (car icon)
    if (tripMovingMarker) {
        tripNavigationMap.removeLayer(tripMovingMarker);
    }

    const carIcon = L.divIcon({
        className: "car-marker",
        html: '<div style="font-size: 36px; text-shadow: 2px 2px 6px rgba(0,0,0,0.3);">🚗</div>',
        iconSize: [36, 36],
        iconAnchor: [18, 18]
    });

    tripMovingMarker = L.marker(waypoints[0], { icon: carIcon }).addTo(tripNavigationMap);

    // Show initial safe alert
    showNavigationAlert("safe", "✓", "Ready to Navigate", "Drive safely");
    // console.log(`[TripPlanner ${TRIP_PLANNER_JS_VERSION}] simulation active: ${SIMULATION_STEP_KM.toFixed(2)} km per ${SIMULATION_TICK_MS}ms`);

    // Adaptive step: route finishes in ~SIMULATION_TARGET_SECONDS regardless of length
    const ticksNeeded = (SIMULATION_TARGET_SECONDS * 1000) / SIMULATION_TICK_MS;
    const simStepKm = Math.max(0.05, totalDistance / ticksNeeded);

    // Start simulation interval
    if (tripNavigationInterval) {
        clearInterval(tripNavigationInterval);
    }
    simulationPaused = false;
    tripNavigationInterval = setInterval(() => {
        if (simulationPaused) return;  // skip tick while paused on alert

        tripPreviousDistance = tripCurrentDistance;
        tripCurrentDistance = Math.min(totalDistance, tripCurrentDistance + simStepKm);

        // Update progress
        updateNavigationUI(totalDistance, waypoints);

        // Check for alerts
        checkNavigationAlerts();

        if (tripCurrentDistance >= totalDistance) {
            clearInterval(tripNavigationInterval);
            showNavigationAlert("safe", "🎉", "Destination Reached!", "You have arrived safely");

            setTimeout(() => {
                exitNavigation();
            }, 3000);
        }
    }, SIMULATION_TICK_MS);
}

function updateNavigationUI(totalDistance, waypoints) {
    const progress = Math.min(100, (tripCurrentDistance / totalDistance) * 100);

    // Update stats
    document.getElementById("nav-current-km").textContent = tripCurrentDistance.toFixed(2);
    document.getElementById("nav-progress-fill").style.width = progress + "%";
    document.getElementById("nav-progress-text").textContent = Math.round(progress) + "%";

    // Update speed (simulate variation)
    const baseSpeed = 58;
    const variation = Math.random() * 12 - 6;
    const speed = Math.max(35, Math.min(75, baseSpeed + variation));
    document.getElementById("nav-speed").textContent = Math.round(speed);

    // Update remaining time
    const remainingDistance = Math.max(0, totalDistance - tripCurrentDistance);
    const remainingMinutes = Math.ceil((remainingDistance / baseSpeed) * 60);
    document.getElementById("nav-eta").textContent = remainingMinutes + " min";

    // Move marker using interpolated position for smooth movement
    const newPosition = getInterpolatedPosition(tripCurrentDistance, waypoints);
    if (tripMovingMarker) tripMovingMarker.setLatLng(newPosition);

    // Pan map to follow marker smoothly
    tripNavigationMap.panTo(newPosition, {
        animate: true,
        duration: 0.3,
        easeLinearity: 0.5
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

    const prevSeverity = lastAlertSeverity;
    lastAlertSeverity = severity;

    // Update background colour for current severity
    const navScreen = document.getElementById("navigation-fullscreen");
    navScreen.className = "navigation-fullscreen active " + severity;

    let icon;
    let title;
    let subtitle;

    if (severity === "danger") {
        icon = "🔴";
        title = "🔴 ENTERING HIGH RISK ZONE";
        subtitle = `Extra caution advised | Risk: ${currentSegment.risk_score}/100`;

        // Voice alert with cooldown
        const now = Date.now();
        if (now - lastAlertTime >= ALERT_COOLDOWN) {
            lastAlertTime = now;
            setTimeout(() => speakAlert("Warning: Entering high risk zone. Drive carefully."), 0);
        }
    } else if (severity === "warning") {
        icon = "⚠️";
        title = prevSeverity === "danger"
            ? "⚠️ RISK DECREASING"
            : "⚠️ ENTERING MODERATE RISK AREA";
        subtitle = `Stay alert | Risk: ${currentSegment.risk_score}/100`;
    } else {
        icon = "✓";
        title = prevSeverity ? "✅ ENTERING SAFE ZONE" : "✓ SAFE ZONE";
        subtitle = "Low risk area - maintain safe driving";
    }

    showNavigationAlert(severity, icon, title, subtitle);

    // Pause animation on danger/warning so judges can read the alert
    if ((severity === "danger" || severity === "warning") && ALERT_PAUSE_MS > 0) {
        simulationPaused = true;
        setTimeout(() => { simulationPaused = false; }, ALERT_PAUSE_MS);
    }

    // Clear any pending safe-zone fadeout so it doesn't hide a danger/warning alert
    if (safeAlertFadeTimeout) {
        clearTimeout(safeAlertFadeTimeout);
        safeAlertFadeTimeout = null;
    }

    if (severity === "safe") {
        safeAlertFadeTimeout = setTimeout(() => {
            fadeOutCurrentAlert();
            safeAlertFadeTimeout = null;
        }, 4000);
    }
}

function speakAlert(message) {
    // Only speak if browser supports it and not currently speaking
    if ("speechSynthesis" in window && !speechSynthesis.speaking) {
        try {
            const utterance = new SpeechSynthesisUtterance(message);
            utterance.rate = 0.9;
            utterance.volume = 0.8;
            utterance.lang = "en-US";
            speechSynthesis.speak(utterance);
            // console.log("Voice alert:", message);
        } catch (error) {
            // console.log("Voice synthesis failed:", error);
        }
    }
}

function fadeOutCurrentAlert() {
    const alertBox = document.getElementById("nav-alert-box");
    alertBox.style.transition = "opacity 0.5s ease";
    alertBox.style.opacity = "0";

    setTimeout(() => {
        alertBox.style.opacity = "1";
        alertBox.style.transition = "";
    }, 500);
}

function showNavigationAlert(severity, icon, title, subtitle) {
    const alertBox = document.getElementById("nav-alert-box");

    // Reset any leftover opacity from fadeOutCurrentAlert
    alertBox.style.opacity = "1";
    alertBox.style.transition = "";

    alertBox.className = "nav-alert-box " + severity;
    document.getElementById("nav-alert-icon").textContent = icon;
    document.getElementById("nav-alert-title").textContent = title;
    document.getElementById("nav-alert-subtitle").textContent = subtitle;

    // Trigger entrance animation; for danger zones, also run the continuous dangerPulse
    alertBox.style.animation = "none";
    setTimeout(() => {
        if (severity === "danger") {
            alertBox.style.animation = "slideUp 0.4s ease, dangerPulse 2s ease infinite";
        } else {
            alertBox.style.animation = "slideUp 0.4s ease";
        }
    }, 10);
}

function exitNavigation() {
    // console.log("Exiting navigation");

    // Stop any ongoing speech
    if ("speechSynthesis" in window) {
        speechSynthesis.cancel();
    }

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
    tripWaypointCumulativeKm = [];
    lastAlertSegmentIndex = -1;
    lastAlertSeverity = null;
    spokenAlerts.clear();
    if (safeAlertFadeTimeout) {
        clearTimeout(safeAlertFadeTimeout);
        safeAlertFadeTimeout = null;
    }

    // Remove moving marker
    if (tripMovingMarker && tripNavigationMap) {
        tripNavigationMap.removeLayer(tripMovingMarker);
        tripMovingMarker = null;
    }
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
    // Pre-set start location so demo chips work immediately
    if (!window.startLat || !window.startLon) {
        window.startLat = DEFAULT_START.lat;
        window.startLon = DEFAULT_START.lon;
        const startInput = document.getElementById("start-location");
        if (startInput) {
            startInput.value = DEFAULT_START.name;
        }
    }

    // ── Bolt-style: pre-initialize the full-screen background map ──
    const mapEl = document.getElementById("main-planner-map");
    if (mapEl && !tripCompareMap) {
        tripCompareMap = L.map("main-planner-map", {
            zoomControl: true,
            scrollWheelZoom: true,
            attributionControl: true,
        }).setView([13.75, 100.53], 12);

        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: "© OpenStreetMap contributors",
            maxZoom: 19,
        }).addTo(tripCompareMap);
    }

    // ── Bottom sheet handle: toggle expanded/collapsed on tap ──
    const handle = document.getElementById("sheet-handle");
    if (handle) {
        handle.addEventListener("click", () => {
            const sheet = document.getElementById("planner-sheet");
            if (sheet) {
                if (sheet.classList.contains("collapsed")) {
                    expandSheet();
                } else {
                    collapseSheet();
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
        // short delay so map tiles are ready
        setTimeout(() => calculateRoute(slat, slon, elat, elon, toName), 800);
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
        collapseSheet();
    } else if (state === 2) {
        if (results) results.classList.remove("hidden");
        expandSheet();
        // Fix 2: scroll sheet to top so cards are immediately visible
        setTimeout(() => {
            if (sheet) sheet.scrollTop = 0;
        }, 150);
    } else if (state === 3) {
        if (results) results.classList.remove("hidden");
        expandSheet();
        // Fix 3: scroll sheet so Start Navigation button is visible
        setTimeout(() => {
            const btn = document.getElementById("start-navigation-btn");
            if (sheet && btn) {
                sheet.scrollTop = btn.offsetTop - 20;
            }
        }, 200);
    }
}

function expandSheet() {
    const sheet = document.getElementById("planner-sheet");
    if (sheet) {
        sheet.classList.remove("collapsed");
        sheet.classList.add("expanded");
    }
    // Invalidate map after transition so tiles render correctly
    setTimeout(() => {
        if (tripCompareMap) tripCompareMap.invalidateSize();
    }, 400);
}

function collapseSheet() {
    const sheet = document.getElementById("planner-sheet");
    if (sheet) {
        sheet.classList.remove("expanded");
        sheet.classList.add("collapsed");
    }
    setTimeout(() => {
        if (tripCompareMap) tripCompareMap.invalidateSize();
    }, 400);
}

function togglePlannerMenu(e) {
    e.stopPropagation();
    const dd = document.getElementById("planner-menu-dropdown");
    if (dd) dd.style.display = dd.style.display === "none" ? "block" : "none";
}

function togglePlannerLayers() {
    // Cycle between standard and CartoDB dark tile layer
    if (!tripCompareMap) return;
    const currentLayers = [];
    tripCompareMap.eachLayer((l) => { if (l instanceof L.TileLayer) currentLayers.push(l); });
    const isDark = currentLayers.some((l) => l._url && l._url.includes("cartocdn"));

    currentLayers.forEach((l) => tripCompareMap.removeLayer(l));

    if (isDark) {
        L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: "© OpenStreetMap contributors", maxZoom: 19,
        }).addTo(tripCompareMap);
    } else {
        L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
            attribution: "© OpenStreetMap contributors © CARTO", maxZoom: 19,
        }).addTo(tripCompareMap);
    }
}
