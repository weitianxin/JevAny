"use strict";
const $ = id => document.getElementById(id);
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
const label = name => {
  const move = /^([xyz])_(plus|minus)_(1|5)cm$/.exec(name);
  return move ? `${move[1].toUpperCase()} ${move[2] === "plus" ? "+" : "−"}${move[3]} cm` :
    ({do:"Interact", noop:"Wait"}[name] || name.replaceAll("_", " "));
};
let config, current = "doom", mode = "replay", replay, state, index = 0;
let playing = false, automatic = false, busy = false, generation = 0, loop = 0, actions = {};

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}
async function request(path, body) {
  const response = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)
  });
  const result = await response.json();
  if (!response.ok) throw Error(result.error || `Request failed (${response.status})`);
  return result;
}
function error(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
  if (message) { playing = false; automatic = false; }
}
function controls() {
  $("primary").textContent = mode === "replay" ? (playing ? "Pause replay" : "Play replay") :
    mode === "model" ? (automatic ? "Pause after this step" : "Run automatically") : "New run";
  $("step").textContent = mode === "model" ? "One decision" : "Next step";
  $("step").hidden = mode === "manual";
  const ready = state && (mode === "replay" || (config.installed[current] && Number.isInteger(state.revision)));
  $("primary").disabled = !ready || (busy && !automatic) || (mode === "model" && (!config.model || state.done));
  $("step").disabled = !ready || busy || (mode === "replay" ? index >= replay.steps.length - 1 : (!config.model || state.done));
  $("reset").disabled = busy;
  $("scrub").hidden = mode !== "replay";
  $("seed-label").hidden = mode === "replay";
  $("seed").disabled = busy;
  document.querySelectorAll("[data-mode],.case").forEach(button => button.disabled = busy);
  document.querySelectorAll(".action").forEach(button => button.disabled = !ready || mode !== "manual" || busy || state?.done);
}
function updateMetrics(observation) {
  let values;
  if (current === "doom") {
    values = [["Health", Math.round(observation.health ?? 0)], ["Ammo", observation.ammo2 ?? 0],
      ["Kills", observation.killcount ?? 0], ["Game ticks", observation.game_ticks ?? 0]];
  } else if (current === "crafter") {
    const inventory = observation.inventory || {}, achieved = observation.achievements || {};
    const milestones = ["collect_wood","place_table","make_wood_pickaxe","collect_stone"].filter(key => achieved[key] > 0).length;
    values = [["Health", `${inventory.health ?? 9} / 9`], ["Wood", inventory.wood ?? 0],
      ["Stone", inventory.stone ?? 0], ["Goal milestones", `${milestones} / 4`]];
  } else {
    const checks = observation.checks || {};
    values = [["Peg held", observation.object_between_both_fingers ? "Yes" : "No"],
      ["Gripper", (observation.gripper_open ?? observation.gripper === "open") ? "Open" : "Closed"],
      ["Height above peg", `${Math.round((observation.gripper_height_above_peg_metres ?? observation.gripper_height_above_target_metres ?? 0)*1000)} mm`],
      ["Checks passed", `${Object.values(checks).filter(Boolean).length} / 6`]];
  }
  $("metrics").replaceChildren(...values.map(([name, value]) => {
    const box = element("div", undefined, "metric");
    box.append(element("span", name), element("b", String(value))); return box;
  }));
}
function drawActions(decision) {
  const probabilities = decision?.probabilities;
  $("action-count").textContent = `${Object.keys(actions).length} controls`;
  $("actions-label").textContent = mode === "manual" ? "CHOOSE AN ACTION" : "ACTION CHOICES";
  $("probability-note").textContent = probabilities ? "Original per-option probabilities from this decision." :
    mode === "manual" ? "Each click executes a real environment action." :
    mode === "model" ? "The model's probabilities appear after its first decision." :
    "This preview uses scripted controls. No model probabilities are shown.";
  if (mode === "replay" && replay.controller === "model" && !probabilities)
    $("probability-note").textContent = "Recorded JevAny-27B-SFT probabilities appear at each step.";
  $("actions").replaceChildren(...Object.entries(actions).map(([key, description]) => {
    const button = element("button", undefined, "action" + (decision?.action === key ? " selected" : ""));
    button.dataset.action = key; button.title = description;
    if (probabilities && probabilities[key] !== undefined) {
      const bar = element("span", undefined, "bar"); bar.style.width = `${probabilities[key]*100}%`; button.append(bar);
    }
    const actionName = element("span", label(key), "name");
    if (/^[xyz]_(plus|minus)_(1|5)cm$/.test(key)) actionName.style.textTransform = "none";
    button.append(actionName);
    if (probabilities?.[key] !== undefined) button.append(element("span", `${(probabilities[key]*100).toFixed(1)}%`, "prob"));
    button.onclick = () => liveStep(key);
    return button;
  }));
}
async function show(snapshot, animate = false) {
  state = snapshot;
  if (Object.keys(state.actions || {}).length) actions = state.actions;
  const observation = state.observation || {};
  const frames = state.frames || [];
  const stamp = generation;
  if (animate && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
    for (const frame of frames) {
      if (stamp !== generation) return;
      $("scene").src = frame; await delay(65);
    }
  } else if (frames.length) $("scene").src = frames.at(-1);
  if (stamp !== generation) return;
  $("step-count").textContent = `STEP ${state.step} / ${mode === "replay" ? replay.steps.length - 1 : config.cases[current].limit}`;
  $("feedback").textContent = [state.decision?.subgoal, state.feedback].filter(Boolean).join(" ");
  $("feedback-label").textContent = state.done ? (state.success ? "GOAL COMPLETED" : "EPISODE ENDED") : "ENVIRONMENT FEEDBACK";
  $("result-dot").className = state.done ? (state.success ? "success" : "failure") : "";
  $("state-json").textContent = JSON.stringify(observation, null, 2);
  $("scrub").value = index;
  updateMetrics(observation); drawActions(state.decision); controls();
}
function modeLabels() {
  document.querySelectorAll("[data-mode]").forEach(button => button.setAttribute("aria-pressed", String(button.dataset.mode === mode)));
  $("provenance").textContent = mode === "replay" ? (replay.controller === "model" ? "RECORDED MODEL RUN" : "SCRIPTED ENVIRONMENT PREVIEW") :
    mode === "manual" ? "LIVE · YOUR CONTROLS" : "LIVE · MODEL DECISIONS";
  $("recording-note").textContent = mode === "replay" ? replay.note :
    "Real environment execution. Simulation time advances only when an action runs." +
    (current === "arm" ? " Motion primitives drive joint motors; contacts and gravity determine the result." : "");
  $("connection").replaceChildren(element("i"), document.createTextNode(mode === "model" && config.model ? config.model : mode === "replay" ? "Local playback" : "Local CPU environment"));
  $("footer-note").textContent = mode === "replay" ? "Playback uses packaged assets only." :
    mode === "manual" ? "Manual play makes no model requests." :
    (config.images ? "The model receives the current image and measured state." : "Text-only: the model receives measured state.");
}
async function selectCase(key) {
  generation++; playing = false; automatic = false; mode = "replay"; current = key; actions = {}; state = null;
  error(""); controls();
  document.querySelectorAll(".case").forEach(button => button.classList.toggle("selected", button.dataset.case === key));
  $("category").textContent = config.cases[key].category;
  $("title").textContent = config.cases[key].title;
  $("goal").textContent = config.cases[key].goal;
  $("scene").className = key === "crafter" ? "pixelated" : "";
  $("scene").alt = config.cases[key].title + " environment view";
  const stamp = generation;
  try {
    const data = await request(`/recordings/${key}/replay.json`);
    if (stamp !== generation) return;
    replay = data; index = 0; $("scrub").max = replay.steps.length - 1;
    $("setup").hidden = true; modeLabels(); await show(replay.steps[0]); startPlayback();
  } catch (e) { error(e.message); }
}
async function startPlayback() {
  if (playing || !replay) return;
  if (index === replay.steps.length - 1) { index = 0; await show(replay.steps[0]); }
  playing = true; controls(); const stamp = generation, playback = ++loop;
  while (playing && playback === loop && mode === "replay" && stamp === generation && index < replay.steps.length - 1) {
    await delay(950);
    if (!playing || playback !== loop || mode !== "replay" || stamp !== generation) break;
    index++; await show(replay.steps[index], true);
  }
  if (stamp === generation && playback === loop) { playing = false; controls(); }
}
function setupMessage(message, command) {
  $("setup").replaceChildren(element("div", message));
  if (command) $("setup").append(element("code", command));
  $("setup").hidden = false;
  $("provenance").textContent = "REPLAY · LIVE SETUP REQUIRED";
  state = null;
}
async function changeMode(next) {
  generation++; playing = false; automatic = false; mode = next; error(""); modeLabels();
  $("setup").hidden = true;
  if (mode === "replay") { index = 0; await show(replay.steps[0]); return; }
  if (!config.installed[current]) {
    setupMessage("Install the optional CPU environments, then restart the playground. Replay remains available.",
      `python -m pip install -e '.[${config.cases[current].extra}]'`);
    controls(); return;
  }
  if (mode === "model" && !config.model) {
    setupMessage("Start the model server with JEVANY_MEDIA_ROOT=/tmp/jevany-media, then connect the playground.",
      "jevany demo --base-url http://127.0.0.1:8008 --media-root /tmp/jevany-media");
    controls(); return;
  }
  await newRun();
}
async function newRun() {
  automatic = false; playing = false; generation++; error("");
  busy = true; $("waiting-label").textContent = "Starting the environment"; $("waiting").hidden = false; controls();
  try {
    const snapshot = await request("/api/start", {case: current, seed: Number($("seed").value)});
    actions = snapshot.actions; await show(snapshot);
  } catch (e) { state = null; error(e.message); }
  finally { busy = false; $("waiting").hidden = true; controls(); }
}
async function liveStep(action) {
  if (busy || !state || state.done) return;
  busy = true; error("");
  $("waiting-label").textContent = mode === "model" ? "Waiting for the model" : "Executing your action";
  $("waiting").hidden = false; controls();
  try {
    const body = {revision: state.revision, ...(mode === "model" ? {model: true} : {action})};
    const snapshot = await request("/api/step", body);
    $("waiting").hidden = true; await show(snapshot, true);
    if (snapshot.done) automatic = false;
  } catch (e) { error(e.message); }
  finally { busy = false; $("waiting").hidden = true; controls(); }
}
$("primary").onclick = async () => {
  if (mode === "replay") { if (playing) {playing = false; loop++; controls();} else startPlayback(); }
  else if (mode === "manual") await newRun();
  else {
    automatic = !automatic; const run = ++loop; controls();
    while (automatic && run === loop && mode === "model" && state && !state.done) {
      await liveStep(); if (automatic) await delay(250);
    }
    controls();
  }
};
$("step").onclick = async () => {
  playing = false; automatic = false; loop++;
  if (mode === "replay" && index < replay.steps.length - 1) {index++; await show(replay.steps[index]);}
  else if (mode === "model") await liveStep();
};
$("reset").onclick = async () => {
  if (mode === "replay") {generation++; playing = false; index = 0; await show(replay.steps[0]);}
  else await newRun();
};
$("scrub").oninput = async () => {generation++; playing = false; index = Number($("scrub").value); await show(replay.steps[index]);};
document.querySelectorAll("[data-mode]").forEach(button => button.onclick = () => changeMode(button.dataset.mode));
$("download").onclick = async () => {
  try {
    const data = mode === "replay" ? replay : await request("/api/trace");
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type:"application/json"}));
    const a = element("a"); a.href = url; a.download = `jevany-${current}-${mode}.json`; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch (e) { error(e.message); }
};
(async () => {
  try {
    config = await request("/api/config");
    for (const [key, meta] of Object.entries(config.cases)) {
      const button = element("button", undefined, "case"); button.dataset.case = key;
      const img = element("img"); img.src = `/recordings/${key}/000.jpg`; img.alt = "";
      const text = element("div"); text.append(element("small", meta.category), element("b", meta.title),
        element("span", key === "crafter" ? "17 native controls" : key === "doom" ? "Freedoom · ViZDoom" : "Franka Panda · PyBullet"));
      button.append(img, text); button.onclick = () => selectCase(key); $("cases").append(button);
    }
    await selectCase(current);
  } catch (e) {error("Could not load the playground: " + e.message);}
})();
