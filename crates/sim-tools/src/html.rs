//! Self-contained HTML replay page (docs/MILESTONE_7 "HTML page").

use serde_json::{json, Map, Value};
use sim_core::score::score;
use sim_core::state::{Outcome, State};

use crate::render::{
    action_label, improvement_name, phase_name, resource_name, snake, terrain_name,
};
use crate::replay::{replay_each, ReplayError, ReplayLog};

/// One tile. Keys whose value is null or false are left out to keep long replays small (Q8); the
/// page reads a missing key as null/false.
fn tile_json(state: &State, i: usize) -> Value {
    let t = &state.tiles[i];
    let mut m = Map::new();
    m.insert("terrain".into(), terrain_name(t.terrain).into());
    if let Some(r) = t.resource {
        m.insert("resource".into(), resource_name(r).into());
    }
    if let Some(imp) = t.improvement {
        m.insert("improvement".into(), improvement_name(imp).into());
    }
    if t.village {
        m.insert("village".into(), true.into());
    }
    if let Some(c) = t.city {
        m.insert("city".into(), c.0.into());
    }
    if let Some(c) = t.owner_city {
        m.insert("owner".into(), state.city(c).owner.into());
    }
    if let Some(u) = t.unit {
        m.insert("unit".into(), u.0.into());
    }
    Value::Object(m)
}

fn frame_json(state: &State) -> Value {
    let outcome = match state.outcome {
        None => Value::Null,
        Some(Outcome::Win(p)) => format!("p{p}").into(),
        Some(Outcome::Draw) => "draw".into(),
    };
    let players: Vec<Value> = (0..2u8)
        .map(|p| {
            json!({
                "stars": state.players[p as usize].stars,
                "score": score(state, p),
                "cities": state.cities.iter().filter(|c| c.owner == p).count(),
                "units": state.units.iter().filter(|u| u.owner == p).count(),
            })
        })
        .collect();
    let units: Vec<Value> = state
        .units
        .iter()
        .map(|u| {
            json!({
                "id": u.id.0,
                "kind": snake(&u.kind),
                "owner": u.owner,
                "row": u.pos.row,
                "col": u.pos.col,
                "health": u.health,
                "max_health": u.max_health,
                "phase": phase_name(u.phase),
                "veteran": u.veteran,
            })
        })
        .collect();
    let cities: Vec<Value> = state
        .cities
        .iter()
        .map(|c| {
            json!({
                "id": c.id.0,
                "owner": c.owner,
                "row": c.pos.row,
                "col": c.pos.col,
                "level": c.level,
                "capital": c.is_capital,
            })
        })
        .collect();
    let tiles: Vec<Value> = (0..state.tiles.len())
        .map(|i| tile_json(state, i))
        .collect();
    json!({
        "turn": state.turn,
        "current": state.current,
        "outcome": outcome,
        "players": players,
        "tiles": tiles,
        "units": units,
        "cities": cities,
    })
}

/// Verifies the log (same errors as `replay`) and returns the page data.
pub fn frames_json(log: &ReplayLog) -> Result<String, ReplayError> {
    let mut frames = Vec::with_capacity(log.actions.len() + 1);
    replay_each(log, &mut |_, s| frames.push(frame_json(s)))?;
    let labels: Vec<String> = log.actions.iter().map(action_label).collect();
    let data = json!({
        "size": log.map_params.size,
        "labels": labels,
        "frames": frames,
    });
    Ok(data.to_string())
}

pub fn html_page(log: &ReplayLog) -> Result<String, ReplayError> {
    let frames = frames_json(log)?;
    Ok(PAGE.replace("/*REPLAY*/", &format!("const REPLAY = {frames};")))
}

const PAGE: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Polytopia replay</title>
<style>
  body { font-family: system-ui, sans-serif; background: #1d2026; color: #e6e6e6; margin: 16px; }
  #wrap { display: flex; flex-wrap: wrap; gap: 20px; align-items: flex-start; }
  #board { display: grid; gap: 2px; background: #111; padding: 2px; }
  .cell { width: 52px; height: 52px; position: relative; box-sizing: border-box; font-size: 12px; }
  .field { background: #a9c46c; }
  .forest { background: #4f7a3a; }
  .mountain { background: #8b8b8b; }
  .fog { background: #2b2e35; }
  .own0 { box-shadow: inset 0 0 0 3px #3a7bd5; }
  .own1 { box-shadow: inset 0 0 0 3px #d5433a; }
  .res { position: absolute; left: 3px; top: 2px; color: #222; font-weight: bold; }
  .site { position: absolute; right: 3px; top: 2px; color: #222; font-weight: bold; }
  .unit { position: absolute; left: 50%; top: 56%; transform: translate(-50%, -50%);
          border-radius: 50%; width: 26px; height: 26px; line-height: 26px; text-align: center;
          color: #fff; font-weight: bold; border: 2px solid #fff; }
  .u0 { background: #3a7bd5; }
  .u1 { background: #d5433a; }
  .vet { border-color: #f2c94c; }
  .hp { position: absolute; bottom: 1px; width: 100%; text-align: center; font-size: 10px; color: #111; }
  #controls { margin: 12px 0; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  #slider { width: 360px; max-width: 100%; }
  button { background: #343944; color: #e6e6e6; border: 1px solid #555; padding: 4px 10px; cursor: pointer; }
  #panel div { margin: 4px 0; }
  .p0 { color: #7fb0ff; }
  .p1 { color: #ff8f86; }
  #label { font-family: monospace; }
  #legend { font-size: 12px; color: #aaa; max-width: 280px; }
</style>
</head>
<body>
<h1>Polytopia replay</h1>
<div id="controls">
  <button id="prev">&#9664; prev</button>
  <button id="play">play</button>
  <button id="next">next &#9654;</button>
  <input type="range" id="slider" min="0" value="0">
  <span id="step"></span>
</div>
<div id="wrap">
  <div id="board"></div>
  <div>
    <div id="panel"></div>
    <p>Last action: <span id="label"></span></p>
    <p id="legend">Letters: top left resource or improvement (o fruit, + crop, a animal, $ metal,
      H lumber hut, F farm, M mine); top right site (V village, C city, K capital, with level).
      Circles are units (W R A D S) in the owner's color, gold rim = veteran. Inner border =
      territory. Arrow keys step through the game.</p>
  </div>
</div>
<script>
/*REPLAY*/
(function () {
  const n = REPLAY.size;
  const last = REPLAY.frames.length - 1;
  const board = document.getElementById("board");
  const slider = document.getElementById("slider");
  const playBtn = document.getElementById("play");
  board.style.gridTemplateColumns = "repeat(" + n + ", 52px)";
  slider.max = String(last);
  const RES = { fruit: "o", crop: "+", animal: "a", metal: "$" };
  const IMP = { lumber_hut: "H", farm: "F", mine: "M" };
  const KIND = { warrior: "W", rider: "R", archer: "A", defender: "D", super_unit: "S" };
  let current = 0;
  let timer = null;

  function draw(k) {
    current = Math.max(0, Math.min(last, k));
    const f = REPLAY.frames[current];
    slider.value = String(current);
    document.getElementById("step").textContent = "step " + current + " / " + last;
    document.getElementById("label").textContent = current === 0 ? "start" : REPLAY.labels[current - 1];
    const units = {};
    for (const u of f.units) units[u.id] = u;
    const cities = {};
    for (const c of f.cities) cities[c.id] = c;
    board.textContent = "";
    f.tiles.forEach(function (t) {
      const cell = document.createElement("div");
      cell.className = "cell " + t.terrain;
      if (t.owner !== undefined && t.owner !== null) cell.classList.add("own" + t.owner);
      const mark = t.improvement ? IMP[t.improvement] : t.resource ? RES[t.resource] : "";
      if (mark) {
        const e = document.createElement("span");
        e.className = "res";
        e.textContent = mark;
        cell.appendChild(e);
      }
      let site = "";
      if (t.city !== undefined && t.city !== null) {
        const c = cities[t.city];
        site = (c.capital ? "K" : "C") + c.level;
      } else if (t.village) {
        site = "V";
      }
      if (site) {
        const e = document.createElement("span");
        e.className = "site";
        e.textContent = site;
        cell.appendChild(e);
      }
      if (t.unit !== undefined && t.unit !== null) {
        const u = units[t.unit];
        const e = document.createElement("div");
        e.className = "unit u" + u.owner + (u.veteran ? " vet" : "");
        e.textContent = KIND[u.kind] || "?";
        e.title = u.kind + " #" + u.id + " " + u.health + "/" + u.max_health + " " + u.phase;
        cell.appendChild(e);
        const hp = document.createElement("span");
        hp.className = "hp";
        hp.textContent = u.health + "/" + u.max_health;
        cell.appendChild(hp);
      }
      board.appendChild(cell);
    });
    const panel = document.getElementById("panel");
    panel.textContent = "";
    let status = "turn " + f.turn + " | P" + f.current + " to move";
    if (f.outcome === "draw") status += " | game over: draw";
    else if (f.outcome) status += " | game over: " + f.outcome.toUpperCase() + " wins";
    const head = document.createElement("div");
    head.textContent = status;
    panel.appendChild(head);
    f.players.forEach(function (p, i) {
      const d = document.createElement("div");
      d.className = "p" + i;
      d.textContent = "P" + i + ": stars " + p.stars + " | score " + p.score +
        " | cities " + p.cities + " | units " + p.units;
      panel.appendChild(d);
    });
  }

  function stop() {
    if (timer !== null) clearInterval(timer);
    timer = null;
    playBtn.textContent = "play";
  }

  function play() {
    if (current >= last) draw(0);
    playBtn.textContent = "pause";
    timer = setInterval(function () {
      if (current >= last) stop();
      else draw(current + 1);
    }, 300);
  }

  playBtn.addEventListener("click", function () { if (timer === null) play(); else stop(); });
  document.getElementById("prev").addEventListener("click", function () { stop(); draw(current - 1); });
  document.getElementById("next").addEventListener("click", function () { stop(); draw(current + 1); });
  slider.addEventListener("input", function () { stop(); draw(Number(slider.value)); });
  document.addEventListener("keydown", function (e) {
    if (e.key === "ArrowLeft") { stop(); draw(current - 1); e.preventDefault(); }
    if (e.key === "ArrowRight") { stop(); draw(current + 1); e.preventDefault(); }
  });
  draw(0);
})();
</script>
</body>
</html>
"#;
