// ===== GLOBAL STATE =====
let tripPlannerRouteData = null;
let tripNavigationMap = null;
let tripNavigationInterval = null;
let tripCurrentDistance = 0;
let tripMovingMarker = null;
let tripPreviousDistance = 0;
let tripWaypointCumulativeKm = [];
let lastAlertSegmentIndex = -1;
let spokenAlerts = new Set();
let lastAlertTime = 0;
const ALERT_COOLDOWN = 5000; // 5 seconds between alerts minimum
const TRIP_PLANNER_JS_VERSION = "2026-02-09.3";
const SIMULATION_STEP_KM = 0.2;
const SIMULATION_TICK_MS = 500;
const DEMO_ROUTE_NAMES = {
    "Huai Khwang": "huai_khwang",
    "Phaya Thai": "phaya_thai",
};
let demoRoutesCache = null;

// Pre-defined backup destinations
const BACKUP_DESTINATIONS = [
    { name: "Siam Paragon", lat: 13.7467, lon: 100.5348 },
    { name: "Chatuchak Weekend Market", lat: 13.80, lon: 100.55 },
    { name: "Victory Monument", lat: 13.765, lon: 100.538 },
    { name: "Huai Khwang", lat: 13.76, lon: 100.75 },
    { name: "Phaya Thai", lat: 13.78, lon: 100.58 },
    { name: "Bang Sue Junction", lat: 13.82, lon: 100.68 },
    { name: "MBK Center", lat: 13.7447, lon: 100.5299 },
    { name: "Ari", lat: 13.7795, lon: 100.5362 }
];

// ===== ADDRESS AUTOCOMPLETE =====
let searchTimeout = null;

async function searchDestinations(query) {
    // Clear previous timeout
    if (searchTimeout) {
        clearTimeout(searchTimeout);
    }

    // Hide suggestions if query is empty
    if (!query || query.length < 2) {
        hideElement("suggestions-dropdown");
        return;
    }

    // Debounce search (wait 300ms after user stops typing)
    searchTimeout = setTimeout(async () => {
        try {
            // Try Nominatim API first
            const response = await fetch(
                `https://nominatim.openstreetmap.org/search?`
                + `q=${encodeURIComponent(query + " Bangkok Thailand")}&`
                + `format=json&`
                + `limit=5&`
                + `addressdetails=1`
            );

            if (response.ok) {
                const results = await response.json();
                displaySuggestions(results, "api");
            } else {
                throw new Error("API failed");
            }
        } catch (error) {
            console.log("API failed, using backup destinations");
            // Fallback to backup destinations
            const filtered = BACKUP_DESTINATIONS.filter((destination) =>
                destination.name.toLowerCase().includes(query.toLowerCase())
            );
            displaySuggestions(filtered, "backup");
        }
    }, 300);
}

function displaySuggestions(results, source) {
    const dropdown = document.getElementById("suggestions-dropdown");

    if (!results || results.length === 0) {
        hideElement("suggestions-dropdown");
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

        item.onclick = () => selectDestination(name, lat, lon);
        dropdown.appendChild(item);
    });

    showElement("suggestions-dropdown");
}

function selectDestination(name, lat, lon) {
    console.log("Selected destination:", name, lat, lon);

    // Update input
    document.getElementById("destination").value = name;

    // Hide suggestions
    hideElement("suggestions-dropdown");

    window.endLat = lat;
    window.endLon = lon;

    // Get start location (default Siam Paragon for now)
    const startLat = window.startLat || 13.7467;
    const startLon = window.startLon || 100.5348;

    // Calculate route
    calculateRoute(startLat, startLon, lat, lon, name);
}

// Close suggestions when clicking outside
document.addEventListener("click", (event) => {
    const dropdown = document.getElementById("suggestions-dropdown");
    const input = document.getElementById("destination");

    if (event.target !== input && !dropdown.contains(event.target)) {
        hideElement("suggestions-dropdown");
    }
});

// ===== GPS LOCATION DETECTION =====
function detectMyLocation() {
    const startInput = document.getElementById("start-location");

    if (!navigator.geolocation) {
        alert("Geolocation is not supported by your browser");
        return;
    }

    startInput.value = "Detecting location...";

    navigator.geolocation.getCurrentPosition(
        (position) => {
            const lat = position.coords.latitude;
            const lon = position.coords.longitude;

            // Reverse geocode to get place name
            fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lon}&format=json`)
                .then((response) => response.json())
                .then((data) => {
                    const placeName = data.display_name.split(",")[0];
                    startInput.value = placeName || "Current Location";

                    // Store coordinates
                    window.startLat = lat;
                    window.startLon = lon;
                })
                .catch(() => {
                    startInput.value = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
                    window.startLat = lat;
                    window.startLon = lon;
                });
        },
        (error) => {
            alert("Unable to detect location: " + error.message);
            startInput.value = "Siam Paragon";
        }
    );
}

// ===== ROUTE CALCULATION =====
async function calculateRoute(startLat, startLon, endLat, endLon, destName) {
    console.log("Calculating route...", {
        startLat, startLon, endLat, endLon
    });

    // Show preview card with loading state
    document.getElementById("preview-title").textContent = "Calculating route to " + destName + "...";
    showElement("route-preview");

    document.getElementById("preview-time").textContent = "--";
    document.getElementById("preview-distance").textContent = "--";
    document.getElementById("risk-text").textContent = "ANALYZING...";
    document.getElementById("risk-details").textContent = "Calculating route risk...";

    try {
        // Call your existing analyze-route API
        const response = await fetch("/api/analyze-route", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                start_lat: startLat,
                start_lon: startLon,
                end_lat: endLat,
                end_lon: endLon
            })
        });

        if (!response.ok) {
            throw new Error("Route calculation failed");
        }

        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || "Unknown error");
        }

        setTripPlannerRouteData(data);

        // Update preview with real data
        displayRoutePreview(data, destName);
    } catch (error) {
        console.error("Route calculation error:", error);

        // Show user-friendly error with fallback options
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

        hideElement("route-preview");
    }
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

    // Update risk indicator
    const riskIndicator = document.getElementById("risk-indicator");
    const riskText = document.getElementById("risk-text");
    const riskDetails = document.getElementById("risk-details");

    // Set risk level - check both risk_level and risk_score
    let riskIcon;
    let riskLabel;
    let riskClass;

    // Debug logging
    console.log("=== RISK CALCULATION DEBUG ===");
    console.log("Risk level:", route.risk_level);
    console.log("Risk score:", route.risk_score);
    console.log("High-risk segments:", summary.high_risk_segments);
    console.log("Medium-risk segments:", summary.medium_risk_segments);
    console.log("Full route data:", data);
    console.log("==============================");

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
    document.querySelector(".risk-icon").textContent = riskIcon;
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
        console.log("Loading pre-saved demo route:", routeName);
        return route;
    } catch (error) {
        console.error("Failed to load demo route:", error);
        return null;
    }
}

async function useDemoRoute(demoRoute, destName, startLat, startLon) {
    try {
        showElement("loading-route");
        document.getElementById("preview-title").textContent = "Loading demo route to " + destName + "...";
        showElement("route-preview");
        document.getElementById("preview-time").textContent = "--";
        document.getElementById("preview-distance").textContent = "--";
        document.getElementById("risk-text").textContent = "ANALYZING...";
        document.getElementById("risk-details").textContent = "Applying pre-saved demo route...";

        const payload = {
            start_lat: startLat,
            start_lon: startLon,
            end_lat: demoRoute.end.lat,
            end_lon: demoRoute.end.lon,
        };

        if (Array.isArray(demoRoute.waypoints) && demoRoute.waypoints.length > 1) {
            payload.waypoints = demoRoute.waypoints;
            if (demoRoute.distance !== undefined) {
                payload.distance_km = demoRoute.distance;
            }
            if (demoRoute.duration !== undefined) {
                payload.duration_min = demoRoute.duration;
            }
        }

        const response = await fetch("/api/analyze-route", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            throw new Error("Demo route analysis failed");
        }

        const data = await response.json();
        if (!data.success) {
            throw new Error(data.error || "Unknown demo route error");
        }

        setTripPlannerRouteData(data);
        displayRoutePreview(data, destName);
        hideElement("loading-route");
        return true;
    } catch (error) {
        console.error("Demo route error:", error);
        hideElement("loading-route");
        return false;
    }
}

// Update selectQuickDestination to use demo routes when available.
async function selectQuickDestination(name, lat, lon) {
    document.getElementById("destination").value = name;
    window.endLat = lat;
    window.endLon = lon;

    const demoRouteKey = DEMO_ROUTE_NAMES[name];
    if (demoRouteKey) {
        try {
            const demoRoute = await loadDemoRoute(demoRouteKey);
            if (demoRoute) {
                const startLat = Number(demoRoute.start?.lat ?? window.startLat ?? 13.7467);
                const startLon = Number(demoRoute.start?.lon ?? window.startLon ?? 100.5348);
                const ok = await useDemoRoute(demoRoute, name, startLat, startLon);
                if (ok) {
                    return;
                }
            }
        } catch (error) {
            console.warn("Demo route failed, using normal calculation:", error);
        }
    }

    // Fallback to normal calculation
    const startLat = window.startLat || 13.7467;
    const startLon = window.startLon || 100.5348;
    await calculateRoute(startLat, startLon, window.endLat, window.endLon, name);
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
    console.log("Starting navigation...");

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
            console.log("Drawing connector segment - gap detected:", gapDistance);
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
    console.log(
        `[TripPlanner ${TRIP_PLANNER_JS_VERSION}] simulation active: `
        + `${SIMULATION_STEP_KM.toFixed(2)} km per ${SIMULATION_TICK_MS}ms`
    );

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
        tripMovingMarker.setLatLng(newPosition);

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

    console.log("=== SEGMENT ALERT DEBUG ===");
    console.log("Version:", TRIP_PLANNER_JS_VERSION);
    console.log("Segment index:", segmentIndex);
    console.log("Current segment color:", segmentColor);
    console.log("Current segment severity:", severity);
    console.log("Current segment risk_level:", currentSegment.risk_level);
    console.log("Current segment risk_score:", currentSegment.risk_score);
    console.log("Prev distance:", tripPreviousDistance.toFixed(2), "km");
    console.log("Distance:", tripCurrentDistance.toFixed(2), "km");
    console.log("==========================");

    let icon;
    let title;
    let subtitle;

    if (severity === "danger") {
        icon = "🔴";
        title = "🔴 HIGH RISK ZONE";
        subtitle = `Extra caution | Risk: ${currentSegment.risk_score}/100`;

        // Voice alert once per segment with cooldown
        const now = Date.now();
        const segmentId = `segment-${currentSegment.start_km}`;
        if (!spokenAlerts.has(segmentId) && now - lastAlertTime >= ALERT_COOLDOWN) {
            speakAlert("Warning: High risk zone. Drive carefully.");
            spokenAlerts.add(segmentId);
            lastAlertTime = now;
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

    if (severity === "safe") {
        setTimeout(() => {
            fadeOutCurrentAlert();
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
            console.log("Voice alert:", message);
        } catch (error) {
            console.log("Voice synthesis failed:", error);
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

    alertBox.className = "nav-alert-box " + severity;
    document.getElementById("nav-alert-icon").textContent = icon;
    document.getElementById("nav-alert-title").textContent = title;
    document.getElementById("nav-alert-subtitle").textContent = subtitle;

    // Pulse animation
    alertBox.style.animation = "none";
    setTimeout(() => {
        alertBox.style.animation = "slideUp 0.4s ease";
    }, 10);
}

function exitNavigation() {
    console.log("Exiting navigation");

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

    // Remove moving marker
    if (tripMovingMarker && tripNavigationMap) {
        tripNavigationMap.removeLayer(tripMovingMarker);
        tripMovingMarker = null;
    }
}

// ===== DEMO MODE TOGGLE =====
function toggleDemoMode() {
    document.body.classList.toggle("demo-mode");
    console.log("Demo mode:", document.body.classList.contains("demo-mode") ? "ON" : "OFF");
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
    console.log(`Trip Planner loaded (version ${TRIP_PLANNER_JS_VERSION})`);
    console.log(
        `Simulation config: ${SIMULATION_STEP_KM.toFixed(2)} km per ${SIMULATION_TICK_MS}ms`
    );
    console.log("Ready for destination input");
});

// Check network status
window.addEventListener("load", () => {
    if (!navigator.onLine) {
        setNetworkStatusVisibility(true);
        console.warn("Offline mode - limited functionality");
    } else {
        setNetworkStatusVisibility(false);
    }
});

window.addEventListener("offline", () => {
    setNetworkStatusVisibility(true);
});

window.addEventListener("online", () => {
    setNetworkStatusVisibility(false);
});
