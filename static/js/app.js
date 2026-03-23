function getRiskPill(level) {
    const normalizedLevel = String(level || "").toLowerCase();
    const cssClass = normalizedLevel === "high" ? "risk-high" : (normalizedLevel === "medium" ? "risk-medium" : "risk-low");
    return `<span class="risk-pill ${cssClass}">${normalizedLevel || "unknown"}</span>`;
}

const THEME_STORAGE_KEY = "rr-theme";
let currentTheme = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";

function getStoredTheme() {
    try {
        return localStorage.getItem(THEME_STORAGE_KEY) === "dark" ? "dark" : "light";
    } catch (error) {
        return "light";
    }
}

function syncThemeToggleButtons(theme = currentTheme) {
    const isDark = theme === "dark";
    document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
        button.classList.toggle("is-dark", isDark);
        button.setAttribute("aria-pressed", String(isDark));
        button.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
        button.setAttribute("title", isDark ? "Switch to light mode" : "Switch to dark mode");

        const label = button.querySelector(".theme-toggle-label");
        if (label) {
            label.textContent = isDark ? "Light mode" : "Dark mode";
        }
    });
}

function applyTheme(theme, options = {}) {
    const normalizedTheme = theme === "dark" ? "dark" : "light";
    const emitEvent = options.emitEvent !== false;

    currentTheme = normalizedTheme;
    document.documentElement.setAttribute("data-theme", normalizedTheme);
    document.documentElement.style.colorScheme = normalizedTheme;
    if (document.body) {
        document.body.setAttribute("data-theme", normalizedTheme);
    }

    syncThemeToggleButtons(normalizedTheme);

    if (emitEvent) {
        window.dispatchEvent(new CustomEvent("rr-theme-change", {
            detail: { theme: normalizedTheme },
        }));
    }

    return normalizedTheme;
}

function toggleTheme() {
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    try {
        localStorage.setItem(THEME_STORAGE_KEY, nextTheme);
    } catch (error) {
        // Ignore storage issues and still update the current page state.
    }
    applyTheme(nextTheme);
}

function initThemeToggle() {
    applyTheme(getStoredTheme(), { emitEvent: false });
    document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
        button.addEventListener("click", toggleTheme);
    });
}

function renderError(message) {
    return `<div class="alert alert-danger mb-0" role="alert">${message}</div>`;
}

let currentRouteData = null;
let drivingInterval = null;
let currentDistance = 0;
let leafletLoadPromise = null;

function appendLeafletCss(url) {
    const existing = Array.from(document.querySelectorAll('link[data-leaflet="true"]'))
        .find((node) => node.getAttribute("href") === url);
    if (existing) return;

    const link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = url;
    link.setAttribute("data-leaflet", "true");
    document.head.appendChild(link);
}

function appendLeafletScript(url) {
    return new Promise((resolve, reject) => {
        const existing = Array.from(document.querySelectorAll('script[data-leaflet="true"]'))
            .find((node) => node.getAttribute("src") === url);
        if (existing) {
            if (typeof window.L !== "undefined") {
                resolve();
            } else {
                existing.addEventListener("load", () => resolve(), { once: true });
                existing.addEventListener("error", () => reject(new Error(`Failed to load ${url}`)), { once: true });
            }
            return;
        }

        const script = document.createElement("script");
        script.src = url;
        script.async = false;
        script.defer = false;
        script.setAttribute("data-leaflet", "true");
        script.onload = () => resolve();
        script.onerror = () => reject(new Error(`Failed to load ${url}`));
        document.body.appendChild(script);
    });
}

async function ensureLeafletLoaded() {
    if (typeof window.L !== "undefined") return;
    if (leafletLoadPromise) {
        await leafletLoadPromise;
        return;
    }

    leafletLoadPromise = (async () => {
        appendLeafletCss("https://unpkg.com/leaflet@1.9.4/dist/leaflet.css");
        appendLeafletCss("https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.css");

        try {
            await appendLeafletScript("https://unpkg.com/leaflet@1.9.4/dist/leaflet.js");
        } catch (primaryError) {
            await appendLeafletScript("https://cdn.jsdelivr.net/npm/leaflet@1.9.4/dist/leaflet.js");
        }

        if (typeof window.L === "undefined") {
            throw new Error("Leaflet failed to initialize");
        }
    })();

    try {
        await leafletLoadPromise;
    } catch (error) {
        leafletLoadPromise = null;
        throw error;
    }
}

function fillExample(startLat, startLon, endLat, endLon) {
    document.querySelector('[name="start_lat"]').value = startLat;
    document.querySelector('[name="start_lon"]').value = startLon;
    document.querySelector('[name="end_lat"]').value = endLat;
    document.querySelector('[name="end_lon"]').value = endLon;
}

async function analyzeRoute(evt) {
    const startLat = parseFloat(document.querySelector('[name="start_lat"]').value);
    const startLon = parseFloat(document.querySelector('[name="start_lon"]').value);
    const endLat = parseFloat(document.querySelector('[name="end_lat"]').value);
    const endLon = parseFloat(document.querySelector('[name="end_lon"]').value);

    if (
        !Number.isFinite(startLat)
        || !Number.isFinite(startLon)
        || !Number.isFinite(endLat)
        || !Number.isFinite(endLon)
    ) {
        alert("Please enter valid start/end coordinates before analyzing.");
        return;
    }

    const button = (evt && evt.target) ? evt.target : null;
    if (!button) return;

    button.disabled = true;
    button.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Analyzing...';

    try {
        const response = await fetch("/api/analyze-route", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                start_lat: startLat,
                start_lon: startLon,
                end_lat: endLat,
                end_lon: endLon,
            }),
        });

        const data = await response.json();

        if (!data.success) {
            alert(`Analysis failed: ${data.error || "Unknown error"}`);
            return;
        }

        currentRouteData = data;

        document.getElementById("route-distance").textContent = data.route.distance_km;
        document.getElementById("route-time").textContent = data.route.duration_min;
        document.getElementById("route-risk").textContent = data.route.risk_score;

        const levelBadge = document.getElementById("route-level");
        levelBadge.textContent = data.route.risk_level.toUpperCase();
        levelBadge.className = `badge bg-${
            data.route.risk_level === "high"
                ? "danger"
                : data.route.risk_level === "medium"
                    ? "warning"
                    : "success"
        }`;

        const alertsList = document.getElementById("alerts-list");
        alertsList.innerHTML = "";

        if (data.alerts.length === 0) {
            alertsList.innerHTML = '<p class="text-success">No high-risk segments on this route</p>';
        } else {
            data.alerts.forEach((alertItem) => {
                const alertDiv = document.createElement("div");
                alertDiv.className = `alert alert-${alertItem.severity} mb-2`;
                alertDiv.innerHTML = `
                    <strong>km ${alertItem.trigger_km}:</strong> ${alertItem.message}
                    ${alertItem.detail ? `<br><small>${alertItem.detail}</small>` : ""}
                `;
                alertsList.appendChild(alertDiv);
            });
        }

        const segmentsTable = document.getElementById("segments-table");
        segmentsTable.innerHTML = "";

        data.segments.forEach((segment) => {
            const row = segmentsTable.insertRow();
            row.innerHTML = `
                <td>${segment.start_km} - ${segment.end_km}</td>
                <td><span class="badge" style="background-color: ${segment.color}">${segment.risk_level.toUpperCase()}</span></td>
                <td>${segment.risk_score}/100</td>
            `;
        });

        try {
            await drawSegmentedRoute(data);
        } catch (mapError) {
            console.error("Map render error:", mapError);
            document.getElementById("map-card").style.display = "none";
        }

        // Show cards progressively
        document.getElementById("route-overview").style.display = "block";
        document.getElementById("alerts-card").style.display = "block";
        document.getElementById("map-card").style.display = "block";
        document.getElementById("driving-card").style.display = "block";
        const simpleTotalKm = document.getElementById("simple-total-km");
        if (simpleTotalKm) simpleTotalKm.textContent = data.route.distance_km.toFixed(1);

        // Smooth scroll to driving card
        setTimeout(() => {
            document.getElementById("driving-card").scrollIntoView({
                behavior: "smooth",
                block: "center",
            });
        }, 500);
    } catch (error) {
        alert(`Error analyzing route: ${error.message}`);
        console.error(error);
    } finally {
        button.disabled = false;
        button.innerHTML = "Analyze Route";
    }
}

async function drawSegmentedRoute(data) {
    await ensureLeafletLoaded();
    const leaflet = window.L;

    if (!window.routeMap) {
        window.routeMap = leaflet.map("route-map").setView([13.75, 100.53], 11);
        leaflet.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
            attribution: "© OpenStreetMap contributors",
        }).addTo(window.routeMap);
    }

    if (window.routeLayerGroup) {
        window.routeMap.removeLayer(window.routeLayerGroup);
    }
    window.routeLayerGroup = leaflet.layerGroup().addTo(window.routeMap);

    data.segments.forEach((segment) => {
        const line = leaflet.polyline(segment.waypoints, {
            color: segment.color,
            weight: 6,
            opacity: 0.8,
        }).addTo(window.routeLayerGroup);

        line.bindTooltip(
            `km ${segment.start_km}-${segment.end_km}: ${segment.risk_level.toUpperCase()} (${segment.risk_score})`,
            {
                permanent: false,
                direction: "top",
            }
        );
    });

    const waypoints = data.route.waypoints;
    leaflet.marker(waypoints[0]).addTo(window.routeLayerGroup).bindPopup("Start");
    leaflet.marker(waypoints[waypoints.length - 1]).addTo(window.routeLayerGroup).bindPopup("End");

    window.routeMap.fitBounds(leaflet.polyline(waypoints).getBounds());
}

function startDrivingMode() {
    if (!currentRouteData) {
        alert("Please analyze a route first");
        return;
    }

    document.getElementById("start-driving").style.display = "none";
    document.getElementById("stop-driving").style.display = "inline-block";
    document.getElementById("driving-status").style.display = "block";

    currentDistance = 0;
    const totalDistance = currentRouteData.route.distance_km;

    drivingInterval = setInterval(() => {
        currentDistance += 0.05;

        if (currentDistance >= totalDistance) {
            stopDrivingMode();
            showDrivingAlert({
                type: "success",
                message: "✓ Destination reached! Drive safely.",
                detail: "",
                severity: "success",
            });
            return;
        }

        const progress = (currentDistance / totalDistance) * 100;
        document.getElementById("driving-progress").style.width = `${progress}%`;
        document.getElementById("progress-text").textContent = `${progress.toFixed(0)}%`;
        document.getElementById("current-km").textContent = currentDistance.toFixed(2);

        checkAlertsAtDistance(currentDistance);
    }, 500);
}

function stopDrivingMode() {
    if (drivingInterval) {
        clearInterval(drivingInterval);
        drivingInterval = null;
    }
    document.getElementById("start-driving").style.display = "inline-block";
    document.getElementById("stop-driving").style.display = "none";
}

function checkAlertsAtDistance(distance) {
    if (!currentRouteData || !currentRouteData.alerts) return;

    const activeAlerts = currentRouteData.alerts.filter((alertItem) =>
        Math.abs(alertItem.trigger_km - distance) < 0.1
    );

    if (activeAlerts.length > 0) {
        const alertItem = activeAlerts[0];
        showDrivingAlert(alertItem);

        if (alertItem.severity === "danger" && "speechSynthesis" in window) {
            const utterance = new SpeechSynthesisUtterance("Warning: High risk zone ahead");
            speechSynthesis.speak(utterance);
        }
    }
}

function showDrivingAlert(alertItem) {
    const alertDiv = document.getElementById("current-alert");
    alertDiv.className = `alert alert-${alertItem.severity}`;
    alertDiv.innerHTML = `
        <strong>${alertItem.message}</strong>
        ${alertItem.detail ? `<br><small>${alertItem.detail}</small>` : ""}
    `;
    alertDiv.style.display = "block";

    if (alertItem.severity === "success") {
        setTimeout(() => {
            alertDiv.style.display = "none";
        }, 3000);
    }
}

let routeMap = null;

function drawRouteMap(fastWaypoints, safeWaypoints) {
    const mapContainer = document.getElementById("route-map");
    if (!mapContainer || typeof window.L === "undefined") {
        return;
    }

    if (!routeMap) {
        routeMap = window.L.map("route-map").setView([13.75, 100.56], 11);
        window.L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
            attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
        }).addTo(routeMap);
    }

    if (window.routeLayerGroup) {
        routeMap.removeLayer(window.routeLayerGroup);
    }
    window.routeLayerGroup = window.L.layerGroup().addTo(routeMap);

    const fastLine = window.L.polyline(fastWaypoints, { color: "red", weight: 4, opacity: 0.7 })
        .addTo(window.routeLayerGroup)
        .bindPopup("Route A (Fast)");
    const safeLine = window.L.polyline(safeWaypoints, { color: "green", weight: 4, opacity: 0.7 })
        .addTo(window.routeLayerGroup)
        .bindPopup("Route B (Safe)");

    const startPoint = fastWaypoints[0];
    const endPoint = fastWaypoints[fastWaypoints.length - 1];
    window.L.marker(startPoint).addTo(window.routeLayerGroup).bindPopup("Start");
    window.L.marker(endPoint).addTo(window.routeLayerGroup).bindPopup("End");

    const allPoints = [...fastWaypoints, ...safeWaypoints];
    if (allPoints.length > 0) {
        routeMap.fitBounds(window.L.latLngBounds(allPoints).pad(0.2));
    } else {
        routeMap.fitBounds(window.L.featureGroup([fastLine, safeLine]).getBounds().pad(0.2));
    }
}

function buildRecommendationMessage(fast, safe, fallbackRecommendation) {
    const riskDiff = Number(fast.risk_score) - Number(safe.risk_score);
    const timeAdd = Number(safe.time_min) - Number(fast.time_min);

    if (riskDiff > 30) {
        return {
            alertClass: "alert-danger",
            message: `Route B significantly safer. Reduces risk by ${riskDiff} points for only +${timeAdd} minutes. Strongly recommended.`,
        };
    }
    if (riskDiff > 15) {
        return {
            alertClass: "alert-warning",
            message: `Route B offers good safety improvement (${riskDiff} points less risk) for +${timeAdd} minutes.`,
        };
    }
    return {
        alertClass: "alert-info",
        message: fallbackRecommendation || "Routes have similar risk levels. Choose based on time preference.",
    };
}

function updateRouteComparisonUI(data) {
    const fast = data.fast_route || data.route_fast;
    const safe = data.safe_route || data.route_safe;
    if (!fast || !safe) {
        throw new Error("Route payload missing fast/safe route data.");
    }

    document.getElementById("fast-distance").textContent = `${Number(fast.distance_km).toFixed(2)} km`;
    document.getElementById("safe-distance").textContent = `${Number(safe.distance_km).toFixed(2)} km`;
    document.getElementById("fast-time").textContent = `${fast.time_min} min`;
    document.getElementById("safe-time").textContent = `${safe.time_min} min`;
    document.getElementById("fast-risk").textContent = `${fast.risk_score}`;
    document.getElementById("safe-risk").textContent = `${safe.risk_score}`;
    document.getElementById("fast-level").innerHTML = getRiskPill(fast.risk_level);
    document.getElementById("safe-level").innerHTML = getRiskPill(safe.risk_level);
    const computedDiff = Number(safe.time_min) - Number(fast.time_min);
    document.getElementById("safe-time-diff").textContent = safe.time_difference || `+${Math.max(0, computedDiff)} min`;

    const recommendationBox = document.getElementById("route-recommendation");
    const recommendation = buildRecommendationMessage(fast, safe, data.recommendation);
    recommendationBox.innerHTML = `<div class="alert ${recommendation.alertClass} mb-0">${recommendation.message}</div>`;
    recommendationBox.classList.add("has-data");

    const resultBox = document.getElementById("route-compare-result");
    resultBox.innerHTML = '<div class="alert alert-success mb-0">Route comparison completed.</div>';

    drawRouteMap(fast.waypoints, safe.waypoints);
    if (data.debug) {
        // console.log("Route compare debug:", data.debug);
    }
}

document.addEventListener("DOMContentLoaded", () => {
    initThemeToggle();

    const routeForm = document.getElementById("route-compare-form");
    const routeResultBox = document.getElementById("route-compare-result");
    const routeSubmitBtn = document.getElementById("route-compare-submit-btn");
    if (routeForm && routeResultBox && routeSubmitBtn) {
        const exampleButtons = document.querySelectorAll(".route-example-btn");
        exampleButtons.forEach((button) => {
            button.addEventListener("click", () => {
                const startLat = button.getAttribute("data-start-lat");
                const startLon = button.getAttribute("data-start-lon");
                const endLat = button.getAttribute("data-end-lat");
                const endLon = button.getAttribute("data-end-lon");

                const startLatInput = document.getElementById("start-lat");
                const startLonInput = document.getElementById("start-lon");
                const endLatInput = document.getElementById("end-lat");
                const endLonInput = document.getElementById("end-lon");
                if (startLatInput && startLonInput && endLatInput && endLonInput) {
                    startLatInput.value = startLat || "";
                    startLonInput.value = startLon || "";
                    endLatInput.value = endLat || "";
                    endLonInput.value = endLon || "";
                }
            });
        });

        routeForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            const formData = new FormData(routeForm);
            const payload = {
                start_lat: Number(formData.get("start_lat")),
                start_lon: Number(formData.get("start_lon")),
                end_lat: Number(formData.get("end_lat")),
                end_lon: Number(formData.get("end_lon")),
            };

            const hasInvalid = Object.values(payload).some((value) => !Number.isFinite(value));
            if (hasInvalid) {
                routeResultBox.innerHTML = renderError("Please provide valid numeric coordinates for all fields.");
                return;
            }

            routeSubmitBtn.disabled = true;
            routeSubmitBtn.textContent = "Comparing...";
            routeResultBox.innerHTML = '<div class="alert alert-info mb-0">Comparing routes...</div>';

            try {
                const response = await fetch("/api/compare-routes", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(payload),
                });
                const data = await response.json();
                if (!response.ok) {
                    routeResultBox.innerHTML = renderError(data.error || "Unable to compare routes.");
                    return;
                }
                updateRouteComparisonUI(data);
            } catch (error) {
                routeResultBox.innerHTML = renderError("Request failed. Please try again.");
            } finally {
                routeSubmitBtn.disabled = false;
                routeSubmitBtn.textContent = "Compare Routes";
            }
        });
    }
});

// ===== SIMPLIFIED DRIVING SIMULATION =====
let simpleDrivingMap = null;
let simpleDrivingInterval = null;
let simpleCurrentDistance = 0;
let simpleMovingMarker = null;

async function startSimpleDriving() {
    if (!currentRouteData) {
        alert("Please analyze a route first");
        return;
    }

    try {
        await ensureLeafletLoaded();
    } catch (error) {
        alert("Unable to load map library. Please try again.");
        return;
    }

    // Show fullscreen
    document.getElementById("simple-driving-fullscreen").classList.add("active");
    document.body.style.overflow = "hidden";

    // Initialize map
    setTimeout(() => {
        if (!simpleDrivingMap) {
            simpleDrivingMap = L.map("simple-driving-map").setView([13.75, 100.53], 12);
            L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
                attribution: "© OpenStreetMap",
            }).addTo(simpleDrivingMap);
        }
        simpleDrivingMap.invalidateSize();

        // Draw route
        drawSimpleRoute();

        // Start simulation
        runSimpleSimulation();
    }, 200);
}

function drawSimpleRoute() {
    // Clear existing layers
    simpleDrivingMap.eachLayer((layer) => {
        if (layer instanceof L.Polyline || (layer instanceof L.Marker && layer !== simpleMovingMarker)) {
            simpleDrivingMap.removeLayer(layer);
        }
    });

    // Draw colored segments
    currentRouteData.segments.forEach((segment) => {
        L.polyline(segment.waypoints, {
            color: segment.color,
            weight: 8,
            opacity: 0.8,
        }).addTo(simpleDrivingMap);
    });

    // Add start/end markers
    const waypoints = currentRouteData.route.waypoints;
    L.marker(waypoints[0]).addTo(simpleDrivingMap).bindPopup("Start");
    L.marker(waypoints[waypoints.length - 1]).addTo(simpleDrivingMap).bindPopup("End");

    // Fit bounds
    simpleDrivingMap.fitBounds(L.polyline(waypoints).getBounds());
}

function runSimpleSimulation() {
    simpleCurrentDistance = 0;
    const totalDistance = currentRouteData.route.distance_km;
    const waypoints = currentRouteData.route.waypoints;

    document.getElementById("simple-total-km").textContent = totalDistance.toFixed(1);

    // Add moving car marker
    if (simpleMovingMarker) {
        simpleDrivingMap.removeLayer(simpleMovingMarker);
    }

    const carIcon = L.divIcon({
        className: "car-marker",
        html: '<div style="font-size: 30px; text-shadow: 2px 2px 4px rgba(0,0,0,0.5);">🚗</div>',
        iconSize: [30, 30],
        iconAnchor: [15, 15],
    });

    simpleMovingMarker = L.marker(waypoints[0], { icon: carIcon }).addTo(simpleDrivingMap);

    // Update time
    updateSimpleTime();

    // Start animation
    simpleDrivingInterval = setInterval(() => {
        simpleCurrentDistance += 0.05; // 50m per tick

        if (simpleCurrentDistance >= totalDistance) {
            clearInterval(simpleDrivingInterval);
            showSimpleAlert("success", "✓ Destination Reached!", "You have arrived safely");
            setTimeout(() => exitSimpleDriving(), 3000);
            return;
        }

        // Update progress
        const progress = (simpleCurrentDistance / totalDistance) * 100;
        document.getElementById("simple-progress-fill").style.width = progress + "%";
        document.getElementById("simple-progress-text").textContent = Math.round(progress) + "%";
        document.getElementById("simple-current-km").textContent = simpleCurrentDistance.toFixed(2);

        // Update speed (simulate variation)
        const speed = 35 + Math.random() * 15;
        document.getElementById("simple-speed").textContent = Math.round(speed);

        // Move marker
        const waypointIndex = Math.floor((simpleCurrentDistance / totalDistance) * waypoints.length);
        if (waypointIndex < waypoints.length) {
            simpleMovingMarker.setLatLng(waypoints[waypointIndex]);
            simpleDrivingMap.panTo(waypoints[waypointIndex], { animate: true, duration: 0.3 });
        }

        // Check for alerts
        checkSimpleAlerts(simpleCurrentDistance);
    }, 500);
}

function checkSimpleAlerts(distance) {
    if (!currentRouteData || !currentRouteData.alerts) return;

    const activeAlert = currentRouteData.alerts.find((alertItem) =>
        Math.abs(alertItem.trigger_km - distance) < 0.15
    );

    if (activeAlert) {
        showSimpleAlert(activeAlert.severity, activeAlert.message, activeAlert.detail);

        // Change background
        const fullscreen = document.getElementById("simple-driving-fullscreen");
        fullscreen.className = "driving-mode-fullscreen active " + activeAlert.severity;

        // Voice alert for danger
        if (activeAlert.severity === "danger" && "speechSynthesis" in window) {
            const utterance = new SpeechSynthesisUtterance("Warning: High risk zone ahead");
            utterance.rate = 0.9;
            speechSynthesis.speak(utterance);
        }
    }
}

function showSimpleAlert(severity, message, detail) {
    const alertBox = document.getElementById("simple-alert-box");
    alertBox.className = "driving-alert-box " + severity;

    document.getElementById("simple-alert-message").textContent = message;
    document.getElementById("simple-alert-detail").textContent = detail || "";

    // Pulse animation
    alertBox.style.animation = "none";
    setTimeout(() => {
        alertBox.style.animation = "pulse 1s ease";
    }, 10);

    // Auto-hide success alerts
    if (severity === "success") {
        setTimeout(() => {
            alertBox.className = "driving-alert-box safe";
            document.getElementById("simple-alert-message").textContent = "Drive safely";
            document.getElementById("simple-alert-detail").textContent = "";
        }, 3000);
    }
}

function pauseSimpleDriving() {
    if (simpleDrivingInterval) {
        clearInterval(simpleDrivingInterval);
        simpleDrivingInterval = null;
        document.getElementById("simple-pause-btn").textContent = "▶️ Resume";
        document.getElementById("simple-pause-btn").setAttribute("onclick", "resumeSimpleDriving()");
    }
}

function resumeSimpleDriving() {
    runSimpleSimulation();
    document.getElementById("simple-pause-btn").textContent = "⏸️ Pause";
    document.getElementById("simple-pause-btn").setAttribute("onclick", "pauseSimpleDriving()");
}

function exitSimpleDriving() {
    if (simpleDrivingInterval) {
        clearInterval(simpleDrivingInterval);
    }
    document.getElementById("simple-driving-fullscreen").classList.remove("active");
    document.getElementById("simple-driving-fullscreen").className = "driving-mode-fullscreen";
    document.body.style.overflow = "auto";
    simpleCurrentDistance = 0;
}

function updateSimpleTime() {
    const now = new Date();
    document.getElementById("simple-time").textContent = now.toLocaleTimeString("en-US", {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
    });

    // Update every minute
    setTimeout(updateSimpleTime, 60000);
}
