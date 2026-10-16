"use strict";

document.addEventListener("DOMContentLoaded", async () => {
  const svg = document.getElementById("grid-explorer-map");
  if (!svg) return;

  const ns = "http://www.w3.org/2000/svg";
  const layers = {
    weather: document.getElementById("layer-weather"),
    lines: document.getElementById("layer-lines"),
    shedding: document.getElementById("layer-shedding"),
  };
  const title = document.getElementById("explorer-detail-title");
  const description = document.getElementById("explorer-detail-description");
  const values = document.getElementById("explorer-detail-values");
  const picker = document.getElementById("explorer-line-picker");
  const extent = { west: -9.5, east: 3.25, south: 35.75, north: 43.75 };
  const base = { width: 1000, height: 630 };
  let view = { x: 0, y: 0, width: base.width, height: base.height };
  let activeLine = null;
  let dragging = null;

  function element(tag, attributes = {}) {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attributes)) {
      node.setAttribute(key, String(value));
    }
    return node;
  }

  function project(lon, lat) {
    return [
      (lon - extent.west) / (extent.east - extent.west) * base.width,
      (extent.north - lat) / (extent.north - extent.south) * base.height,
    ];
  }

  function color(value, min, max, stops) {
    const position = Math.max(0, Math.min(stops.length - 1,
      (value - min) / (max - min) * (stops.length - 1)));
    const lower = Math.min(stops.length - 2, Math.floor(position));
    const fraction = position - lower;
    const rgb = stops[lower].map((channel, i) =>
      Math.round(channel + fraction * (stops[lower + 1][i] - channel)));
    return `rgb(${rgb.join(",")})`;
  }

  const weatherStops = [[60, 95, 199], [168, 198, 244], [241, 239, 233],
    [238, 159, 136], [183, 44, 69]];
  const lineStops = [[72, 164, 95], [169, 198, 77], [230, 200, 76],
    [238, 114, 57], [182, 41, 97], [135, 46, 208]];

  function detail(heading, copy, entries) {
    title.textContent = heading;
    description.textContent = copy;
    values.replaceChildren();
    for (const [label, value] of entries) {
      const dt = document.createElement("dt");
      dt.textContent = label;
      const dd = document.createElement("dd");
      dd.textContent = value;
      values.append(dt, dd);
    }
  }

  function setView(x, y, width, height) {
    view = {
      x: Math.max(0, Math.min(base.width - width, x)),
      y: Math.max(0, Math.min(base.height - height, y)),
      width,
      height,
    };
    svg.setAttribute("viewBox",
      `${view.x} ${view.y} ${view.width} ${view.height}`);
  }

  function zoom(factor, focusX = 0.5, focusY = 0.5) {
    const width = Math.max(base.width / 5,
      Math.min(base.width, view.width / factor));
    const height = width * base.height / base.width;
    const anchorX = view.x + focusX * view.width;
    const anchorY = view.y + focusY * view.height;
    setView(anchorX - focusX * width, anchorY - focusY * height,
      width, height);
  }

  try {
    const response = await fetch("assets/data/spain_2030.json");
    if (!response.ok) throw new Error(`Map data: HTTP ${response.status}`);
    const data = await response.json();
    if (data.case.snapshot_id !== "ES-2030" || data.lines.length !== 442 ||
        data.buses.length !== 281) {
      throw new Error("Unexpected grid example data");
    }
    svg.replaceChildren();

    const weatherGroup = element("g", { "aria-label": "Air-temperature field" });
    const outlineGroup = element("g", { "aria-hidden": "true" });
    const lineGroup = element("g", { "aria-label": "Transmission lines" });
    const markerGroup = element("g", { "aria-label": "Load-shedding markers" });
    svg.append(weatherGroup, outlineGroup, lineGroup, markerGroup);

    const cellWidth = 0.25 / (extent.east - extent.west) * base.width;
    const cellHeight = 0.25 / (extent.north - extent.south) * base.height;
    for (const cell of data.weather) {
      const [x, y] = project(cell.lon, cell.lat);
      weatherGroup.append(element("rect", {
        x: x - cellWidth / 2, y: y - cellHeight / 2,
        width: cellWidth + 0.5, height: cellHeight + 0.5,
        fill: color(cell.temperature_c, 20, 50, weatherStops),
        opacity: 0.82,
      }));
    }

    const polygons = data.region_outline.type === "MultiPolygon"
      ? data.region_outline.coordinates : [data.region_outline.coordinates];
    for (const polygon of polygons) {
      const path = polygon.map(ring => ring.map(([lon, lat], index) => {
        const [x, y] = project(lon, lat);
        return `${index ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(" ") + " Z").join(" ");
      outlineGroup.append(element("path", {
        d: path, fill: "none", stroke: "#4d6262", "stroke-width": 1.6,
        "vector-effect": "non-scaling-stroke", opacity: 0.9,
      }));
    }

    const lineElements = new Map();
    for (const line of data.lines) {
      const [x0, y0] = project(line.lon0, line.lat0);
      const [x1, y1] = project(line.lon1, line.lat1);
      const d = `M${x0},${y0} L${x1},${y1}`;
      const painted = element("path", {
        d, fill: "none", "stroke-width": 2.2,
        "vector-effect": "non-scaling-stroke",
        stroke: color(line.temperature_c, 30, 90, lineStops),
      });
      const hit = element("path", {
        d, fill: "none", stroke: "transparent", "stroke-width": 11,
        "vector-effect": "non-scaling-stroke", class: "map-hit",
      });
      hit.append(element("title"));
      hit.firstChild.textContent = `Line ${line.id}: ${line.temperature_c.toFixed(1)} C`;
      const showLine = () => {
        if (activeLine) activeLine.setAttribute("stroke-width", 2.2);
        activeLine = painted;
        painted.setAttribute("stroke-width", 4.2);
        picker.value = line.id;
        detail(`Line ${line.id}`, "Solved conductor temperature at this hour.", [
          ["From", line.bus0], ["To", line.bus1],
          ["Temperature", `${line.temperature_c.toFixed(1)} C`],
        ]);
      };
      hit.addEventListener("pointerenter", showLine);
      hit.addEventListener("click", showLine);
      lineGroup.append(painted, hit);
      lineElements.set(line.id, showLine);
    }

    const markedBuses = data.buses.filter(bus => bus.display_marker);
    for (const bus of markedBuses) {
      const [x, y] = project(bus.lon, bus.lat);
      const marker = element("path", {
        d: `M${x},${y - 9} L${x + 8},${y + 7} L${x - 8},${y + 7} Z`,
        class: "map-marker", fill: "#f2d34b", stroke: "#1d292d",
        "stroke-width": 1.8, tabindex: 0, role: "button",
        "aria-label": `Bus ${bus.id}, load shedding ${bus.load_shedding_mw.toFixed(3)} MW`,
      });
      const showBus = () => {
        if (activeLine) activeLine.setAttribute("stroke-width", 2.2);
        activeLine = null;
        picker.value = "";
        detail(`Bus ${bus.id}`, "Modelled demand and unmet demand at this bus.", [
          ["Demand", `${bus.demand_mw.toFixed(2)} MW`],
          ["Load shedding", `${bus.load_shedding_mw.toFixed(3)} MW`],
        ]);
      };
      marker.addEventListener("pointerenter", showBus);
      marker.addEventListener("click", showBus);
      marker.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          showBus();
        }
      });
      markerGroup.append(marker);
    }

    for (const line of [...data.lines].sort((a, b) =>
      b.temperature_c - a.temperature_c)) {
      const option = document.createElement("option");
      option.value = line.id;
      option.textContent = `Line ${line.id} (${line.temperature_c.toFixed(1)} C)`;
      picker.append(option);
    }
    picker.addEventListener("change", () => {
      if (lineElements.has(picker.value)) lineElements.get(picker.value)();
    });

    function updateLayers() {
      weatherGroup.style.display = layers.weather.checked ? "" : "none";
      lineGroup.style.display = layers.lines.checked ? "" : "none";
      markerGroup.style.display = layers.shedding.checked ? "" : "none";
      document.getElementById("weather-legend").style.display =
        layers.weather.checked ? "" : "none";
      document.getElementById("line-legend").style.display =
        layers.lines.checked ? "" : "none";
    }
    Object.values(layers).forEach(input =>
      input.addEventListener("change", updateLayers));
    updateLayers();

    detail("24 July 2030, 13:00", "Select a line or the yellow marker to inspect the saved result.", [
      ["Represented lines", String(data.lines.length)],
      ["Represented buses", String(data.buses.length)],
      ["Marked buses", String(markedBuses.length)],
    ]);
    document.getElementById("map-zoom-in").addEventListener("click", () => zoom(1.4));
    document.getElementById("map-zoom-out").addEventListener("click", () => zoom(1 / 1.4));
    document.getElementById("map-zoom-reset").addEventListener("click", () =>
      setView(0, 0, base.width, base.height));
    svg.addEventListener("wheel", event => {
      event.preventDefault();
      const rect = svg.getBoundingClientRect();
      zoom(event.deltaY < 0 ? 1.25 : 0.8,
        (event.clientX - rect.left) / rect.width,
        (event.clientY - rect.top) / rect.height);
    }, { passive: false });
    svg.addEventListener("pointerdown", event => {
      if (event.button !== 0) return;
      if (event.target.closest(".map-hit, .map-marker")) return;
      dragging = { x: event.clientX, y: event.clientY, start: { ...view } };
      svg.setPointerCapture(event.pointerId);
      svg.classList.add("is-dragging");
    });
    svg.addEventListener("pointermove", event => {
      if (!dragging) return;
      const rect = svg.getBoundingClientRect();
      setView(dragging.start.x - (event.clientX - dragging.x) / rect.width * view.width,
        dragging.start.y - (event.clientY - dragging.y) / rect.height * view.height,
        view.width, view.height);
    });
    const stopDrag = () => {
      dragging = null;
      svg.classList.remove("is-dragging");
    };
    svg.addEventListener("pointerup", stopDrag);
    svg.addEventListener("pointercancel", stopDrag);
  } catch (error) {
    svg.replaceChildren(element("text", {
      x: 500, y: 315, "text-anchor": "middle", class: "map-loading",
    }));
    svg.firstChild.textContent = "Map data could not be loaded";
    detail("Map unavailable", "The example data could not be loaded.", []);
    console.error(error);
  }
});
