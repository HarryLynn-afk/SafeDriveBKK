const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

function element() {
    return {
        value: '', style: {}, children: [], textContent: '',
        classList: { add() {}, remove() {}, contains() { return false; } },
        appendChild(child) { this.children.push(child); },
        set innerHTML(value) { this.children = []; },
        setAttribute() {}, removeAttribute() {},
        contains(target) { return this.children.includes(target); },
    };
}

function browser() {
    const elements = Object.fromEntries([
        'start-location', 'destination', 'start-suggestions-dropdown', 'suggestions-dropdown',
    ].map(id => [id, element()]));
    const timers = new Map();
    let timerId = 0;
    const context = {
        console: { log() {}, warn() {}, error() {} },
        AbortController,
        setTimeout(fn, delay) { timers.set(++timerId, { fn, delay }); return timerId; },
        clearTimeout(id) { timers.delete(id); },
        addEventListener() {},
        document: {
            addEventListener() {},
            querySelector() { return null; },
            getElementById(id) { return elements[id] || null; },
            createElement: element,
            documentElement: { getAttribute() { return 'light'; } },
        },
    };
    context.window = context;
    vm.createContext(context);
    vm.runInContext(readFileSync(join(__dirname, '../../static/js/trip-planner.js'), 'utf8'), context);
    return {
        context, elements,
        async type(query, start = true) {
            elements[start ? 'start-location' : 'destination'].value = query;
            context[start ? 'searchStartLocations' : 'searchDestinations'](query);
            for (const [id, timer] of [...timers]) {
                if (timer.delay <= 300) { timers.delete(id); await timer.fn(); }
            }
        },
    };
}

test('Places New suggestions resolve selected coordinates with a session token', async () => {
    const { context, elements, type } = browser();
    const requests = [];
    const fields = [];
    context.google = { maps: { places: {
        AutocompleteSessionToken: class {},
        AutocompleteSuggestion: {
            async fetchAutocompleteSuggestions(request) {
                requests.push(request);
                return { suggestions: [{ placePrediction: {
                    mainText: { text: 'Siam Paragon' }, secondaryText: { text: 'Bangkok' },
                    toPlace() { return {
                        location: { lat: () => 13.7467, lng: () => 100.5348 },
                        async fetchFields(request) { fields.push(request.fields); },
                    }; },
                } }] };
            },
        },
    } } };
    context.initGooglePlaces();
    await type('Siam');
    assert.equal(requests.length, 1);
    assert.equal(requests[0].input, 'Siam');
    assert.deepEqual(Array.from(requests[0].includedRegionCodes), ['th']);
    assert.ok(requests[0].sessionToken);
    await elements['start-suggestions-dropdown'].children[0].onclick();
    assert.equal(elements['start-location'].value, 'Siam Paragon');
    assert.equal(context.startLat, 13.7467);
    assert.equal(context.startLon, 100.5348);
    assert.deepEqual(Array.from(fields[0]), ['location']);
});

function enablePlaces(context, fetchAutocompleteSuggestions) {
    context.google = { maps: { places: {
        AutocompleteSessionToken: class {},
        AutocompleteSuggestion: { fetchAutocompleteSuggestions },
    } } };
    context.initGooglePlaces();
}

function prediction(name) {
    return { placePrediction: { mainText: { text: name }, secondaryText: { text: 'Bangkok' } } };
}

test('older autocomplete responses cannot replace newer suggestions', async () => {
    const { context, elements, type } = browser();
    const pending = {};
    enablePlaces(context, request => new Promise(resolve => { pending[request.input] = resolve; }));
    const older = type('Older');
    const newer = type('Newer');
    pending.Newer({ suggestions: [prediction('Newer')] });
    await newer;
    pending.Older({ suggestions: [prediction('Older')] });
    await older;
    assert.equal(elements['start-suggestions-dropdown'].children[0].children[0].children[0].textContent, 'Newer');
});

test('editing or clearing a location invalidates the previous coordinates', async () => {
    const { context, type } = browser();
    context.startLat = context.endLat = 13.75;
    context.startLon = context.endLon = 100.5;
    await type('');
    assert.equal(context.startLat, null);
    assert.equal(context.startLon, null);
    await type('', false);
    assert.equal(context.endLat, null);
    assert.equal(context.endLon, null);
});

function previewBrowser() {
    const env = browser();
    const drawn = [];
    env.context.google = { maps: {
        Polyline: class {
            constructor(options) { drawn.push(options); }
            setMap() {}
            getPath() { return []; }
        },
        LatLngBounds: class { extend() {} isEmpty() { return false; } },
    } };
    env.context.testMap = { fitBounds() {} };
    env.context.testStart = { getPosition() { return { lat: () => 13.75, lng: () => 100.50 }; } };
    env.context.testEnd = { getPosition() { return { lat: () => 13.80, lng: () => 100.55 }; } };
    vm.runInContext('tripCompareMap = testMap; previewStartMarker = testStart; previewEndMarker = testEnd', env.context);
    return { ...env, drawn };
}

test('route preview calls the backend and draws its road geometry', async () => {
    const { context, drawn } = previewBrowser();
    const requests = [];
    context.fetch = async (url, options) => {
        requests.push({ url, options });
        return { ok: true, async json() { return { success: true, waypoints: [[13.75, 100.50], [13.8, 100.55]] }; } };
    };
    await context.fetchAndDrawPreviewRoute();
    assert.equal(requests.length, 1);
    assert.equal(requests[0].url, '/api/preview-route');
    assert.equal(requests[0].options.method, 'POST');
    assert.deepEqual(JSON.parse(requests[0].options.body), {
        start_lat: 13.75, start_lon: 100.50, end_lat: 13.80, end_lon: 100.55,
    });
    assert.equal(drawn.length, 1);
    assert.equal(drawn[0].path[0].lat, 13.75);
    assert.equal(drawn[0].path[1].lng, 100.55);
});

test('older preview responses cannot replace a newer route', async () => {
    const { context, drawn } = previewBrowser();
    const pending = [];
    context.fetch = (_url, options) => new Promise(resolve => pending.push({ resolve, options }));
    const older = context.fetchAndDrawPreviewRoute();
    context.newEnd = { getPosition() { return { lat: () => 13.9, lng: () => 100.6 }; } };
    vm.runInContext('previewEndMarker = newEnd', context);
    const newer = context.fetchAndDrawPreviewRoute();
    pending[1].resolve({ ok: true, async json() { return { success: true, waypoints: [[13.75, 100.5], [13.9, 100.6]] }; } });
    await newer;
    pending[0].resolve({ ok: true, async json() { return { success: true, waypoints: [[13.75, 100.5], [13.8, 100.55]] }; } });
    await older;
    assert.equal(drawn.length, 1);
    assert.equal(drawn[0].path[1].lat, 13.9);
    assert.equal(pending[0].options.signal.aborted, true);
});

test('autocomplete sessions are reused while typing and renewed after clearing', async () => {
    const { context, type } = browser();
    const tokens = [];
    enablePlaces(context, async request => { tokens.push(request.sessionToken); return { suggestions: [] }; });
    await type('Si');
    await type('Siam');
    await type('Siam', false);
    assert.equal(tokens[0], tokens[1]);
    assert.notEqual(tokens[0], tokens[2]);
    await type('');
    await type('Siam');
    assert.notEqual(tokens[0], tokens[3]);
});

test('a selected place cannot overwrite a newer typed query while details load', async () => {
    const { context, elements, type } = browser();
    let finish;
    enablePlaces(context, async () => ({ suggestions: [{ placePrediction: {
        mainText: { text: 'Old place' }, secondaryText: { text: 'Bangkok' },
        toPlace() { return {
            location: { lat: () => 13.75, lng: () => 100.5 },
            fetchFields() { return new Promise(resolve => { finish = resolve; }); },
        }; },
    } }] }));
    await type('Old');
    const selection = elements['start-suggestions-dropdown'].children[0].onclick();
    await type('Different');
    finish();
    await selection;
    assert.equal(elements['start-location'].value, 'Different');
    assert.equal(context.startLat, null);
});

test('address service failure keeps backup destinations usable', async () => {
    const { context, elements, type } = browser();
    enablePlaces(context, async () => { throw new Error('API denied'); });
    await type('Siam', false);
    const button = elements['suggestions-dropdown'].children[0];
    assert.equal(button.type, 'button');
    await button.onclick();
    assert.equal(context.endLat, 13.7467);
    assert.equal(context.endLon, 100.5348);
});

test('preview failure leaves the existing dashed line intact', async () => {
    const { context, drawn } = previewBrowser();
    let removed = false;
    context.dashed = { setMap() { removed = true; } };
    vm.runInContext('previewRouteLine = dashed', context);
    context.fetch = async () => ({ ok: false });
    await context.fetchAndDrawPreviewRoute();
    assert.equal(drawn.length, 0);
    assert.equal(removed, false);
});

test('clearing the preview prevents an in-flight response from redrawing it', async () => {
    const { context, drawn } = previewBrowser();
    context.testStart.setMap = context.testEnd.setMap = () => {};
    let finish;
    context.fetch = () => new Promise(resolve => { finish = resolve; });
    const request = context.fetchAndDrawPreviewRoute();
    context.clearPreviewMarkers();
    finish({ ok: true, async json() { return { success: true, waypoints: [[13.75, 100.5], [13.8, 100.55]] }; } });
    await request;
    assert.equal(drawn.length, 0);
});
