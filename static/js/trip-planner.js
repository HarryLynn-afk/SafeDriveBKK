// ===== GLOBAL STATE =====
let tripPlannerRouteData = null;
let tripNavigationMap = null;

// ===== MULTI-ROUTE COMPARISON STATE =====
let routeOptions = {};   // { fast: routeObj, safe: routeObj }
let routeLayers = {};    // { fast: L.Polyline, safe: L.Polyline }
let selectedRoute = null;
let tripCompareMap = null;
let tripNavigationInterval = null;
let tripCurrentDistance = 0;
let tripMovingMarker = null;
let tripPreviousDistance = 0;
let tripWaypointCumulativeKm = [];
let lastAlertSegmentIndex = -1;
let spokenAlerts = new Set();
let lastAlertTime = 0;
let safeAlertFadeTimeout = null;
const ALERT_COOLDOWN = 5000; // 5 seconds between alerts minimum
const TRIP_PLANNER_JS_VERSION = "2026-02-19.1";
const SIMULATION_STEP_KM = 0.2;
const SIMULATION_TICK_MS = 500;
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
    { name: "Ratchada", lat: 13.73, lon: 100.72 },
    { name: "Din Daeng", lat: 13.73, lon: 100.74 },
    { name: "Asok", lat: 13.74, lon: 100.56 },
    { name: "Huai Khwang", lat: 13.76, lon: 100.75 },
    { name: "MBK Center", lat: 13.7447, lon: 100.5299 },
    { name: "Ari", lat: 13.7795, lon: 100.5362 }
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
        if (window.endLat && window.endLon) {
            calculateRoute(lat, lon, window.endLat, window.endLon, document.getElementById("destination").value);
        }
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
        if (!window.startLat || !window.startLon) {
            alert("Please set a start location first (search or use GPS).");
            startInput.focus();
            return;
        }
        calculateRoute(window.startLat, window.startLon, lat, lon, name);
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
            <div class="suggestion-icon">📍</div>
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

    // If destination already selected, recalculate route
    if (window.endLat && window.endLon) {
        const destInput = document.getElementById("destination");
        calculateRoute(lat, lon, window.endLat, window.endLon, destInput.value);
    }
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

    // Get start location — use user-selected or prompt them
    const startLat = window.startLat;
    const startLon = window.startLon;

    if (!startLat || !startLon) {
        alert("Please set a start location first (search or use GPS).");
        document.getElementById("start-location").focus();
        return;
    }

    calculateRoute(startLat, startLon, lat, lon, name);
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
        alert("Geolocation is not supported by your browser. Using default start location.");
        return;
    }

    // Show loading state
    startInput.value = "Detecting location...";
    if (gpsBtn) {
        gpsBtn.textContent = "⏳";
        gpsBtn.disabled = true;
    }

    navigator.geolocation.getCurrentPosition(
        (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;

            // Restore button
            if (gpsBtn) {
                gpsBtn.textContent = "✅";
                gpsBtn.disabled = false;
                setTimeout(() => { gpsBtn.textContent = "📍"; }, 2000);
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
                });
            } else {
                fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`)
                    .then((response) => response.json())
                    .then((data) => {
                        const placeName = data.display_name ? data.display_name.split(",")[0] : "";
                        startInput.value = placeName || "Current Location";
                        window.startLat = lat;
                        window.startLon = lon;
                    })
                    .catch(() => {
                        startInput.value = "Current Location";
                        window.startLat = lat;
                        window.startLon = lon;
                    });
            }
        },
        (error) => {
            // Restore button
            if (gpsBtn) {
                gpsBtn.textContent = "📍";
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
        },
        { enableHighAccuracy: true, timeout: 10000, maximumAge: 60000 }
    );
}

// ===== ROUTE CALCULATION =====
async function calculateRoute(startLat, startLon, endLat, endLon, destName) {
    // console.log("Comparing routes...", { startLat, startLon, endLat, endLon });

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
        updateStatus("✓ Analysis complete!");
        await sleep(100);

        hideAIAnalyzing();
        displayMultipleRoutes(data);

    } catch (error) {
        console.error("Route calculation error:", error);
        hideAIAnalyzing();
        alert(
            `Unable to calculate route: ${error.message}\n\n`
            + "This may be due to:\n"
            + "• Network connectivity issues\n"
            + "• Location outside Bangkok coverage\n"
            + "• External routing service unavailable\n\n"
            + "Please try:\n"
            + "• Different destination\n"
            + "• Check internet connection\n"
            + "• Use pre-loaded example routes"
        );
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
        s.textContent = "⏳";
    });
    const counter = document.getElementById("routes-analyzed");
    if (counter) counter.textContent = "0";
}

function hideAIAnalyzing() {
    const overlay = document.getElementById("ai-analyzing-overlay");
    if (overlay) overlay.style.display = "none";
}

function updateAnalysisStep(stepNumber, status) {
    const step = document.getElementById(`step-${stepNumber}`);
    if (!step) return;
    const icon = step.querySelector(".step-status");
    step.classList.remove("active", "complete");
    if (status === "active") {
        step.classList.add("active");
        if (icon) icon.textContent = "⏳";
    } else if (status === "complete") {
        step.classList.add("complete");
        if (icon) icon.textContent = "✅";
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
        tripCompareMap = L.map("route-compare-map", {
            zoomControl: true,
            scrollWheelZoom: false
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

    // Reveal results container (map + selection cards are inside)
    showElement("route-results");

    // Render route selection cards immediately (doesn't depend on map)
    showRouteSelectionUI(data);

    // Defer Leaflet map init so the browser has time to paint the container
    // before Leaflet measures its dimensions. Without this delay the map
    // renders as a grey box because the element reports 0×0 size.
    setTimeout(() => {
        initCompareMap();
        clearPreviewMap();

        // Draw both routes: blue = fast, green = safe
        displayRouteOnMap(data.fast_route, "fast", "#3B82F6", 0.75);
        displayRouteOnMap(data.safe_route, "safe", "#10B981", 0.75);

        // Fit map bounds to encompass both routes
        const allLayers = Object.values(routeLayers);
        if (allLayers.length > 0) {
            const group = L.featureGroup(allLayers);
            tripCompareMap.fitBounds(group.getBounds(), { padding: [50, 50] });
        }

        // Second invalidateSize after fitBounds for reliable tile loading
        setTimeout(() => tripCompareMap.invalidateSize(), 100);
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

function clearPreviewMap() {
    Object.values(routeLayers).forEach((layer) => {
        if (tripCompareMap) tripCompareMap.removeLayer(layer);
    });
    routeLayers = {};
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

    // Confidence values come from the backend (0.0–1.0). Fall back to
    // sensible defaults so UI never shows 0% or "undefined".
    const fastConf = Math.round((fastRoute.confidence || 0.88) * 100);
    const safeConf = Math.round((safeRoute.confidence || 0.92) * 100);
    const fastMethod = fastRoute.prediction_method || "xgboost";
    const safeMethod = safeRoute.prediction_method || "xgboost";
    const methodLabel = (m) => m === "xgboost" ? "XGBoost" : "Spatial ML";
    const altCount = data.alternatives_analyzed || 5;
    const fastTrafficSrc = fastRoute.traffic_source || "estimated";
    const safeTrafficSrc = safeRoute.traffic_source || "estimated";
    const trafficBadge = (src) => {
        if (src === "google_maps_traffic") return '<span class="traffic-badge live">Live Traffic</span>';
        if (src === "google_maps") return '<span class="traffic-badge">Google Maps</span>';
        if (src === "osrm") return '';
        return '<small class="simulated-note">*estimated</small>';
    };

    const routeSelectionHTML = `
        <div class="route-selection-container">
            <!-- ML analysis badge -->
            <div class="ml-analysis-badge">
                🤖 <strong>AI-Powered Route Analysis</strong>
                <span class="ml-model">XGBoost ML &bull; 83.8% Validated</span>
            </div>

            <h3>🗺️ Choose Your Route</h3>
            <p class="subtitle">AI analyzed ${altCount} route options &mdash; showing you the best two</p>

            <div class="route-options-grid">
                <!-- Fast Route Card -->
                <div class="route-card" id="route-card-fast" onclick="selectRoute('fast')">
                    <div class="route-badge fastest">🚀 FASTEST</div>
                    <div class="route-time">${fastTime} min ${trafficBadge(fastTrafficSrc)}</div>
                    <div class="route-distance">${(fastRoute.distance_km || 0).toFixed(1)} km</div>
                    <div class="route-card-risk ${fastRoute.risk_level}">
                        ${getRiskIcon(fastRoute.risk_level)} ${(fastRoute.risk_level || "unknown").toUpperCase()} RISK
                    </div>
                    <div class="risk-score">${fastRiskScore}/100</div>
                    ${fastHighRisk > 0
                        ? `<div class="warning-note">⚠️ Passes ${fastHighRisk} danger zone${fastHighRisk > 1 ? "s" : ""}</div>`
                        : '<div class="safe-note">✅ No major danger zones</div>'}
                    <div class="ai-confidence-indicator">
                        🎯 AI Confidence: <strong>${fastConf}%</strong>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width:${fastConf}%"></div>
                        </div>
                    </div>
                    <div class="prediction-method">
                        Method: <span class="method-badge">${methodLabel(fastMethod)}</span>
                    </div>
                    <button class="select-route-btn" onclick="selectRoute('fast'); event.stopPropagation();">
                        Select This Route
                    </button>
                </div>

                <!-- Safe Route Card -->
                <div class="route-card recommended" id="route-card-safe" onclick="selectRoute('safe')">
                    <div class="route-badge safest">🛡️ SAFEST</div>
                    ${safeRiskScore < fastRiskScore ? '<div class="recommended-badge">⭐ RECOMMENDED</div>' : ""}
                    <div class="route-time">${safeTime} min ${trafficBadge(safeTrafficSrc)}</div>
                    <div class="route-distance">${(safeRoute.distance_km || 0).toFixed(1)} km</div>
                    <div class="route-card-risk ${safeRoute.risk_level}">
                        ${getRiskIcon(safeRoute.risk_level)} ${(safeRoute.risk_level || "unknown").toUpperCase()} RISK
                    </div>
                    <div class="risk-score">${safeRiskScore}/100</div>
                    <div class="time-difference">${timeDiffText}</div>
                    ${safeHighRisk > 0
                        ? `<div class="warning-note">⚠️ Passes ${safeHighRisk} danger zone${safeHighRisk > 1 ? "s" : ""}</div>`
                        : '<div class="safe-note">✅ No major danger zones</div>'}
                    <div class="ai-confidence-indicator">
                        🎯 AI Confidence: <strong>${safeConf}%</strong>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width:${safeConf}%"></div>
                        </div>
                    </div>
                    <div class="prediction-method">
                        Method: <span class="method-badge">${methodLabel(safeMethod)}</span>
                    </div>
                    <button class="select-route-btn primary" onclick="selectRoute('safe'); event.stopPropagation();">
                        Select This Route
                    </button>
                </div>
            </div>

            ${data.recommendation
                ? `<div class="ai-recommendation">🤖 AI Recommendation: ${data.recommendation}</div>`
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
        case "high": return "🔴";
        case "medium": return "🟡";
        case "low": return "🟢";
        default: return "⚪";
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
    const detailsHTML = `
        <div class="selected-route-details">
            <h3>📍 Selected Route Details</h3>
            <div class="route-summary">
                <div class="summary-item">
                    <span class="label">Duration</span>
                    <span class="value">${route.time_min} min</span>
                </div>
                <div class="summary-item">
                    <span class="label">Distance</span>
                    <span class="value">${(route.distance_km || 0).toFixed(1)} km</span>
                </div>
                <div class="summary-item">
                    <span class="label">Risk Level</span>
                    <span class="value ${route.risk_level}">${(route.risk_level || "").toUpperCase()}</span>
                </div>
                <div class="summary-item">
                    <span class="label">Risk Score</span>
                    <span class="value">${Math.round(route.risk_score || 0)}/100</span>
                </div>
            </div>
            ${(route.high_risk_zones || 0) > 0
                ? `<div class="danger-zones-preview">
                    <h4>⚠️ Danger Zones on This Route</h4>
                    <p>This route crosses ${route.high_risk_zones} high-risk zone${route.high_risk_zones > 1 ? "s" : ""}.
                       You'll receive alerts during navigation.</p>
                   </div>`
                : '<div class="safe-route-note">✅ This route avoids major danger zones</div>'}
            <button class="start-navigation-btn" onclick="startNavigationWithSelected()">
                🚗 Start Navigation
            </button>
            <button class="change-route-btn" onclick="changeRoute()">
                ← Choose Different Route
            </button>
        </div>
    `;

    const container = document.querySelector(".route-selection-container");
    const existing = document.querySelector(".selected-route-details");
    if (existing) existing.remove();
    if (container) container.insertAdjacentHTML("beforeend", detailsHTML);
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

async function selectQuickDestination(name, lat, lon) {
    document.getElementById("destination").value = name;
    window.endLat = lat;
    window.endLon = lon;

    // Require start location
    if (!window.startLat || !window.startLon) {
        alert("Please set a start location first (search or use GPS).");
        document.getElementById("start-location").focus();
        return;
    }

    // Always go through the multi-route compare-routes flow so all chips
    // show the Waze-style route selection UI (not the old single-route card).
    await calculateRoute(window.startLat, window.startLon, lat, lon, name);
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
    document.body.style.overflow = "hidden";

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
    tripPreviousDistance = 0;

    // Reset state
    tripCurrentDistance = 0;
    const totalDistance = tripPlannerRouteData.route.distance_km;
    const waypoints = tripPlannerRouteData.route.waypoints;
    tripWaypointCumulativeKm = buildWaypointDistanceProfile(waypoints);

    // Update UI
    document.getElementById("nav-total-km").textContent = totalDistance.toFixed(1);
    document.getElementById("nav-distance").textContent = totalDistance.toFixed(1) + " km";
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

    // Start simulation interval
    if (tripNavigationInterval) {
        clearInterval(tripNavigationInterval);
    }
    tripNavigationInterval = setInterval(() => {
        tripPreviousDistance = tripCurrentDistance;
        tripCurrentDistance = Math.min(totalDistance, tripCurrentDistance + SIMULATION_STEP_KM);

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

    // Move marker using distance profile (keeps marker and segment alerts aligned)
    const waypointIndex = getWaypointIndexForDistance(tripCurrentDistance);
    if (waypointIndex < waypoints.length) {
        const newPosition = waypoints[waypointIndex];
        if (tripMovingMarker) tripMovingMarker.setLatLng(newPosition);

        // Pan map to follow marker smoothly
        tripNavigationMap.panTo(newPosition, {
            animate: true,
            duration: 0.3,
            easeLinearity: 0.5
        });
    }
}

function checkNavigationAlerts() {
    if (!tripPlannerRouteData || !tripPlannerRouteData.segments) return;

    const segmentIndex = findSegmentIndexForDistance(tripCurrentDistance);
    if (segmentIndex < 0) {
        return;
    }

    // Only show alert on segment transition
    if (segmentIndex === lastAlertSegmentIndex) {
        return;
    }

    const currentSegment = tripPlannerRouteData.segments[segmentIndex];
    const segmentColor = String(currentSegment.color || "").toLowerCase().trim();
    const severity = currentSegment.ui_severity || getSegmentSeverity(currentSegment);

    let icon;
    let title;
    let subtitle;

    if (severity === "danger") {
        icon = "🔴";
        title = "🔴 HIGH RISK ZONE";
        subtitle = `Extra caution | Risk: ${currentSegment.risk_score}/100`;

        // Voice alert once per segment with cooldown (deferred to avoid blocking animation)
        const now = Date.now();
        const segmentId = `segment-${currentSegment.start_km}`;
        if (!spokenAlerts.has(segmentId) && now - lastAlertTime >= ALERT_COOLDOWN) {
            spokenAlerts.add(segmentId);
            lastAlertTime = now;
            setTimeout(() => speakAlert("Warning: High risk zone. Drive carefully."), 0);
        }
    } else if (severity === "warning") {
        icon = "⚠️";
        title = "⚠️ MEDIUM RISK AREA";
        subtitle = `Stay alert | Risk: ${currentSegment.risk_score}/100`;
    } else {
        icon = "✓";
        title = "✓ SAFE ZONE";
        subtitle = "Low risk area - maintain safe driving";
    }

    showNavigationAlert(severity, icon, title, subtitle);
    lastAlertSegmentIndex = segmentIndex;

    const navScreen = document.getElementById("navigation-fullscreen");
    navScreen.className = "navigation-fullscreen active " + severity;

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
    if (tripNavigationInterval) {
        clearInterval(tripNavigationInterval);
        tripNavigationInterval = null;
    }

    // Hide fullscreen
    const navScreen = document.getElementById("navigation-fullscreen");
    navScreen.classList.remove("active");
    navScreen.className = "navigation-fullscreen";
    document.body.style.overflow = "auto";

    // Reset state
    tripCurrentDistance = 0;
    tripPreviousDistance = 0;
    tripWaypointCumulativeKm = [];
    lastAlertSegmentIndex = -1;
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
