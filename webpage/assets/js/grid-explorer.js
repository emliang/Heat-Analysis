"use strict";

document.addEventListener("DOMContentLoaded", async () => {
  const svg = document.getElementById("grid-explorer-map");
  if (!svg) return;

  const ns = "http://www.w3.org/2000/svg";
  const layers = {
    weather: document.getElementById("layer-weather"),
    lines: document.getElementById("layer-lines"),
    shedding: document.getElementById("layer-shedding"),
    nodes: document.getElementById("layer-nodes"),
  };
  const title = document.getElementById("explorer-detail-title");
  const description = document.getElementById("explorer-detail-description");
  const values = document.getElementById("explorer-detail-values");
  const picker = document.getElementById("explorer-line-picker");
  const nodePicker = document.getElementById("explorer-node-picker");
  const weatherPicker = document.getElementById("weather-field");
  let mode = "solved";
  let selected = null;
  const extent = { west: -9.75, east: 3.75, south: 35.5, north: 44.35 };
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
  const weatherFields = {
    temperature_c: { title: "Air temperature", min: 20, max: 50, unit: "°C", stops: weatherStops },
    wind_speed_m_s: { title: "10-m wind speed", min: 0, max: 8, unit: "m/s", stops: [[247,252,240],[168,221,181],[67,162,202],[8,64,129]] },
    solar_irradiance_w_m2: { title: "Solar irradiance", min: 0, max: 1000, unit: "W/m²", stops: [[255,255,204],[254,217,118],[253,141,60],[189,0,38]] },
  };
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
    const configResponse = await fetch("assets/data/spain_grid_configuration.json");
    if (!configResponse.ok) throw new Error(`Grid configuration: HTTP ${configResponse.status}`);
    const config = await configResponse.json();
    const weatherResponse = await fetch("assets/data/spain_2030_weather.json");
    if (!weatherResponse.ok) throw new Error(`Weather: HTTP ${weatherResponse.status}`);
    const weatherData = await weatherResponse.json();
    if (weatherData.timestamp !== "2030-07-24T13:00:00" || weatherData.cells.length !== data.weather.length) {
      throw new Error("Unexpected weather snapshot");
    }
    if (data.case.snapshot_id !== "ES-2030" || data.lines.length !== 442 ||
        data.buses.length !== 281) {
      throw new Error("Unexpected grid example data");
    }
    svg.replaceChildren();

    const weatherGroup = element("g", { "aria-label": "Air-temperature field" });
    const outlineGroup = element("g", { "aria-hidden": "true", "pointer-events": "none" });
    const lineGroup = element("g", { "aria-label": "Transmission lines" });
    const markerGroup = element("g", { "aria-label": "Load-shedding markers" });
    const nodeGroup = element("g", { "aria-label": "Network nodes" });
    const dcGroup = element("g", { "aria-label": "DC connections" });
    const capacityGroup = element("g", { "aria-label": "Installed capacity mix" });
    svg.append(weatherGroup, outlineGroup, lineGroup, dcGroup, nodeGroup, capacityGroup, markerGroup);
    const linePaint = new Map();
    const nodePaint = new Map();
    const nodeActions = new Map();
    const maxDemand = Math.max(...data.buses.map(bus => bus.demand_mw));
    const degree = new Map(data.buses.map(bus => [bus.id, 0]));
    const capacityTotal = id => Object.values(config.capacity_by_bus[id] || {}).reduce((a, b) => a + b, 0);
    const maxCapacity = Math.max(...data.buses.map(bus => capacityTotal(bus.id)));
    const lineWidth = id => mode === "grid" ? 0.35 + 2.4 * Math.sqrt(config.ac_capacity_mva[id] / 5000) : 2.2;
    data.lines.forEach(line => {
      degree.set(line.bus0, (degree.get(line.bus0) || 0) + 1);
      degree.set(line.bus1, (degree.get(line.bus1) || 0) + 1);
    });
    function clearSelection() {
      if (activeLine && selected?.type === "line") activeLine.setAttribute("stroke-width", lineWidth(selected.id));
      activeLine = null;
      nodePaint.forEach(node => node.classList.remove("is-selected"));
      weatherGroup.querySelectorAll(".is-selected").forEach(node => node.classList.remove("is-selected"));
      capacityGroup.querySelectorAll(".is-selected").forEach(node => node.classList.remove("is-selected"));
      picker.value = "";
      nodePicker.value = "";
    }

    const cellWidth = 0.25 / (extent.east - extent.west) * base.width;
    const cellHeight = 0.25 / (extent.north - extent.south) * base.height;
    const weatherRects = [];
    const weatherActions = [];
    for (const [index, cell] of weatherData.cells.entries()) {
      const [x, y] = project(cell.lon, cell.lat);
      const rect = element("rect", {
        x: x - cellWidth / 2, y: y - cellHeight / 2,
        width: cellWidth + 0.5, height: cellHeight + 0.5,
        fill: color(cell.temperature_c, 20, 50, weatherStops),
        opacity: 0.82,
        class: "weather-cell",
        "data-weather-index": index,
      });
      const tooltip = element("title");
      tooltip.textContent = `${cell.temperature_c.toFixed(1)} °C; ${cell.wind_speed_m_s.toFixed(2)} m/s; ${cell.solar_irradiance_w_m2.toFixed(1)} W/m²`;
      rect.append(tooltip);
      const showWeather = () => {
        clearSelection();
        selected = { type: "weather", id: index };
        rect.classList.add("is-selected");
        detail("Weather grid cell", "24 July 2030, 13:00", [
          ["Longitude", cell.source_lon.toFixed(3)], ["Latitude", cell.source_lat.toFixed(3)],
          ["Air temperature", `${cell.temperature_c.toFixed(1)} °C`],
          ["10-m wind speed", `${cell.wind_speed_m_s.toFixed(2)} m/s`],
          ["Solar irradiance", `${cell.solar_irradiance_w_m2.toFixed(1)} W/m²`],
        ]);
      };
      rect.addEventListener("click", () => { if (!suppressWeatherClick) showWeather(); });
      weatherGroup.append(rect);
      weatherRects.push(rect);
      weatherActions.push(showWeather);
    }
    let suppressWeatherClick = false;
    function updateWeather() {
      const key = mode === "heat" ? weatherPicker.value : "temperature_c";
      const field = weatherFields[key];
      weatherRects.forEach((rect, i) => rect.setAttribute("fill",
        color(weatherData.cells[i][key], field.min, field.max, field.stops)));
      weatherGroup.setAttribute("aria-label", field.title + " field");
      document.getElementById("weather-legend-title").textContent = field.title;
      document.getElementById("weather-min").textContent = `${field.min} ${field.unit}`;
      document.getElementById("weather-max").textContent = `${field.max} ${field.unit}`;
      document.getElementById("weather-scale").style.background =
        `linear-gradient(90deg, ${field.stops.map(rgb => `rgb(${rgb.join(',')})`).join(',')})`;
    }
    weatherPicker.addEventListener("change", () => {
      layers.weather.checked = true;
      updateWeather(); updateLayers();
    });

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
        clearSelection();
        selected = { type: "line", id: line.id };
        activeLine = painted;
        painted.setAttribute("stroke-width", 4.2);
        picker.value = line.id;
        detail(`Line ${line.id}`, mode === "solved" ? "Solved conductor temperature at this hour." : "Connection in the represented network.", [
          ["From", line.bus0], ["To", line.bus1],
          ...(mode === "grid" ? [["Nominal capacity", `${config.ac_capacity_mva[line.id].toFixed(1)} MVA`]] : []),
          ...(mode === "solved" ? [["Temperature", `${line.temperature_c.toFixed(1)} °C`]] : []),
        ]);
      };
      hit.addEventListener("click", showLine);
      lineGroup.append(painted, hit);
      lineElements.set(line.id, showLine);
      linePaint.set(line.id, painted);
    }

    for (const bus of data.buses) {
      const [x, y] = project(bus.lon, bus.lat);
      const node = element("circle", { cx: x, cy: y, r: 3.5,
        fill: "#fff", stroke: "#236b70", "stroke-width": 1.3,
        "vector-effect": "non-scaling-stroke", class: "map-node" });
      const tooltip = element("title");
      tooltip.textContent = `Node ${bus.id}`;
      node.append(tooltip);
      const showBus = () => {
        clearSelection();
        selected = { type: "node", id: bus.id };
        node.classList.add("is-selected");
        nodePicker.value = bus.id;
        detail(`Node ${bus.id}`, mode === "grid" ? "Bus in the represented Spanish grid." : "Demand at this modelled bus for the selected hour.", [
          ["Connected lines", String(degree.get(bus.id))],
          ["Longitude", bus.lon.toFixed(3)], ["Latitude", bus.lat.toFixed(3)],
          ...(mode === "grid" ? config.categories.map(category => [category, `${(config.capacity_by_bus[bus.id]?.[category] || 0).toFixed(1)} MW`]) : []),
          ...(mode === "grid" ? [["Total installed", `${capacityTotal(bus.id).toFixed(1)} MW`]] : []),
          ...(mode !== "grid" ? [["Demand", `${bus.demand_mw.toFixed(2)} MW`]] : []),
          ...(mode === "solved" ? [["Load shedding", `${(Math.abs(bus.load_shedding_mw) < 0.0005 ? 0 : bus.load_shedding_mw).toFixed(3)} MW`]] : []),
        ]);
      };
      node.addEventListener("click", showBus);
      nodePaint.set(bus.id, node);
      nodeActions.set(bus.id, showBus);
      nodeGroup.append(node);
      const option = document.createElement("option");
      option.value = bus.id;
      option.textContent = bus.id;
      nodePicker.append(option);
      const total = capacityTotal(bus.id);
      if (total > 0) {
        const radius = 2.5 + 15 * Math.sqrt(total / maxCapacity);
        const pie = element("g", { class: "map-capacity", "data-bus": bus.id });
        let angle = -Math.PI / 2;
        for (const category of config.categories) {
          const value = config.capacity_by_bus[bus.id][category];
          if (value <= 0) continue;
          const end = angle + value / total * Math.PI * 2;
          const slice = value === total
            ? element("circle", { cx: x, cy: y, r: radius })
            : element("path", { d: `M${x},${y} L${x + radius * Math.cos(angle)},${y + radius * Math.sin(angle)} A${radius},${radius} 0 ${end - angle > Math.PI ? 1 : 0},1 ${x + radius * Math.cos(end)},${y + radius * Math.sin(end)} Z` });
          slice.setAttribute("fill", config.palette[category]);
          pie.append(slice);
          angle = end;
        }
        const ring = element("circle", { cx: x, cy: y, r: radius, fill: "none", stroke: "#555", "stroke-width": 0.7, "vector-effect": "non-scaling-stroke" });
        pie.append(ring);
        pie.addEventListener("click", showBus);
        capacityGroup.append(pie);
        const originalAction = showBus;
        nodeActions.set(bus.id, () => { originalAction(); pie.classList.add("is-selected"); });
        pie.addEventListener("click", () => pie.classList.add("is-selected"));
      }
    }
    const busesById = new Map(data.buses.map(bus => [bus.id, bus]));
    for (const link of config.dc_links) {
      const from = busesById.get(link.bus0), to = busesById.get(link.bus1);
      const [x1, y1] = project(from.lon, from.lat), [x2, y2] = project(to.lon, to.lat);
      const path = element("line", { x1, y1, x2, y2, stroke: "#a0008b", "stroke-width": 2, "vector-effect": "non-scaling-stroke", class: "map-hit" });
      const tooltip = element("title");
      tooltip.textContent = `DC link: ${link.capacity_mw} MW`;
      path.append(tooltip);
      path.addEventListener("click", () => {
        clearSelection(); selected = null;
        detail("DC connection", "Represented direct-current link.", [["From", link.bus0], ["To", link.bus1], ["Nominal capacity", `${link.capacity_mw} MW`]]);
      });
      dcGroup.append(path);
    }
    nodePicker.addEventListener("change", () => nodeActions.get(nodePicker.value)?.());
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
        nodeActions.get(bus.id)();
      };
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
      nodeGroup.style.display = layers.nodes.checked ? "" : "none";
      capacityGroup.style.display = mode === "grid" && layers.nodes.checked ? "" : "none";
      dcGroup.style.display = mode === "grid" && layers.lines.checked ? "" : "none";
      document.getElementById("capacity-legend").hidden = mode !== "grid";
      document.getElementById("weather-legend").style.display =
        layers.weather.checked ? "" : "none";
      document.getElementById("line-legend").style.display =
        layers.lines.checked && mode === "solved" ? "" : "none";
      document.getElementById("network-legend").style.display =
        mode === "heat" && layers.lines.checked ? "" : "none";
      document.getElementById("demand-legend").style.display =
        mode === "heat" && layers.nodes.checked ? "" : "none";
      picker.hidden = mode === "heat" && !layers.lines.checked;
      document.querySelector('label[for="explorer-line-picker"]').hidden = picker.hidden;
      nodePicker.hidden = mode === "heat" && !layers.nodes.checked;
      document.querySelector('label[for="explorer-node-picker"]').hidden = nodePicker.hidden;
      document.querySelector(".explorer-detail-note").hidden = mode === "heat";
    }
    Object.values(layers).forEach(input =>
      input.addEventListener("change", updateLayers));
    updateLayers();

    function overview() {
      clearSelection();
      selected = null;
      detail("24 July 2030, 13:00", "One saved Spain 2030 scenario.", [
      ["AC lines", String(data.lines.length)],
      ...(mode === "grid" ? [["DC links", String(config.dc_links.length)]] : []),
      ["Represented buses", String(data.buses.length)],
      ["Annual base-demand growth", `${data.case.load_growth_rate * 100}%`],
      ["Initial storage charge", `${data.case.initial_storage_soc * 100}%`],
      ]);
    }
    function setMode(next) {
      mode = next;
      document.querySelectorAll("[data-grid-view]").forEach(button =>
        button.setAttribute("aria-pressed", String(button.dataset.gridView === mode)));
      layers.weather.checked = mode !== "grid";
      layers.lines.checked = mode !== "heat";
      layers.nodes.checked = mode !== "heat";
      document.getElementById("weather-field-control").hidden = mode !== "heat";
      updateWeather();
      layers.shedding.checked = mode === "solved";
      layers.shedding.disabled = mode !== "solved";
      data.lines.forEach(line => {
        linePaint.get(line.id).setAttribute("stroke", mode === "solved" ? color(line.temperature_c, 30, 90, lineStops) : mode === "grid" ? "#059aa6" : "#65767c");
        linePaint.get(line.id).setAttribute("stroke-width", lineWidth(line.id));
      });
      const linesById = new Map(data.lines.map(line => [String(line.id), line]));
      for (const option of picker.options) {
        const line = linesById.get(option.value);
        if (line) option.textContent = mode === "solved"
          ? `Line ${line.id} (${line.temperature_c.toFixed(1)} °C)` : `Line ${line.id}`;
      }
      data.buses.forEach(bus => {
        const node = nodePaint.get(bus.id);
        node.setAttribute("r", mode === "heat" ? 12 * Math.sqrt(Math.max(0, bus.demand_mw) / maxDemand) : 3.5);
        node.setAttribute("fill", mode === "heat" ? "#236b70" : "#fff");
        node.setAttribute("fill-opacity", mode === "heat" ? 0.65 : 1);
      });
      updateLayers();
      if (selected?.type === "line") lineElements.get(selected.id)();
      else if (selected?.type === "node") nodeActions.get(selected.id)();
      else if (selected?.type === "weather" && mode !== "grid") weatherActions[selected.id]();
      else overview();
    }
    document.querySelectorAll("[data-grid-view]").forEach(button =>
      button.addEventListener("click", () => setMode(button.dataset.gridView)));
    document.getElementById("explorer-overview").addEventListener("click", overview);
    setMode("solved");
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
      suppressWeatherClick = false;
      if (event.button !== 0) return;
      if (event.target.closest(".map-hit, .map-marker, .map-node, .map-capacity")) return;
      dragging = { x: event.clientX, y: event.clientY, start: { ...view }, cell: event.target.closest(".weather-cell") };
      svg.setPointerCapture(event.pointerId);
      svg.classList.add("is-dragging");
    });
    svg.addEventListener("pointermove", event => {
      if (!dragging) return;
      if (Math.hypot(event.clientX - dragging.x, event.clientY - dragging.y) > 4) suppressWeatherClick = true;
      const rect = svg.getBoundingClientRect();
      setView(dragging.start.x - (event.clientX - dragging.x) / rect.width * view.width,
        dragging.start.y - (event.clientY - dragging.y) / rect.height * view.height,
        view.width, view.height);
    });
    const stopDrag = event => {
      if (event.type === "pointerup" && dragging?.cell && !suppressWeatherClick) {
        weatherActions[Number(dragging.cell.dataset.weatherIndex)]();
      }
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
