// Page logic: inputs, lineups, validation and the result. The math lives in model.js.
(function () {
  "use strict";
  const M = LoLModel;
  const ROLE_NAMES = { top: "Top", jng: "Jng", mid: "Mid", bot: "Bot", sup: "Sup" };
  const GROUPS = ["Team rating", "Player ratings", "Experience", "Champion comfort", "Champion meta",
                  "Draft (solo queue)", "Blue side & baseline"];
  let SNAP = null;
  const teamByLabel = {}, teamLabel = {}, playerByLabel = {}, playerLabel = {}, playersByName = {};
  const champByKey = {};
  const ui = {};

  const norm = (s) => String(s || "").toLowerCase().replace(/[^a-z0-9]/g, "");
  const $ = (sel, root = document) => root.querySelector(sel);
  const today = () => Date.now() / 86400000;

  fetch("data/snapshot.json")
    .then((r) => { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(init)
    .catch((e) => { $("#snapshot-info").textContent = "Couldn't load the model (" + e.message + ")."; });

  function option(list, value) {
    const o = document.createElement("option");
    o.value = value;
    list.appendChild(o);
  }

  function init(snap) {
    SNAP = snap;
    const teams = Object.keys(snap.teams).sort((a, b) => {
      const A = snap.teams[a], B = snap.teams[b];
      return (B.worlds - A.worlds) || A.league.localeCompare(B.league) || A.name.localeCompare(B.name);
    });
    for (const id of teams) {
      const t = snap.teams[id];
      let label = `${t.name} · ${t.league}`;
      if (teamByLabel[label]) label += ` (${t.last_game})`;
      teamByLabel[label] = id;
      teamLabel[id] = label;
      option($("#dl-teams"), label);
    }
    const players = Object.keys(snap.players).sort((a, b) => snap.players[a].name.localeCompare(snap.players[b].name));
    for (const id of players) {
      const p = snap.players[id];
      let label = `${p.name} · ${p.team}`;
      if (playerByLabel[label]) label += ` #${id.slice(-4)}`;
      playerByLabel[label] = id;
      playerLabel[id] = label;
      (playersByName[norm(p.name)] = playersByName[norm(p.name)] || []).push(id);
      option($("#dl-players"), label);
    }
    for (const c of snap.champions) {
      champByKey[norm(c)] = c;
      option($("#dl-champs"), c);
    }

    ui.blue = buildSide("blue", "Blue side");
    ui.red = buildSide("red", "Red side");
    $("#swap").addEventListener("click", swap);
    $("#clear-champs").addEventListener("click", () => {
      for (const s of [ui.blue, ui.red]) for (const c of s.champs) c.value = "";
      update();
    });

    $("#snapshot-info").textContent =
      `Results through ${snap.data_until_date} · patch ${snap.patch} · draft from solo queue ` +
      `${snap.draft.patches.join(", ")} (${snap.draft.matches.toLocaleString()} games) · updated ${snap.generated}`;
    selfTest();
    update();
  }

  function buildSide(key, title) {
    const root = $("#side-" + key);
    root.appendChild($("#side-template").content.cloneNode(true));
    $(".side-title", root).textContent = title;
    const side = { root, team: $(".team", root), note: $(".team-note", root), players: [], champs: [] };
    const slots = $(".slots", root);
    for (const role of M.ROLES) {
      const row = document.createElement("div");
      row.className = "slot";
      row.innerHTML = `<span class="role">${ROLE_NAMES[role]}</span>
        <input class="player" list="dl-players" autocomplete="off" spellcheck="false" aria-label="${title} ${role} player">
        <input class="champ" list="dl-champs" autocomplete="off" spellcheck="false" aria-label="${title} ${role} champion" placeholder="Champion">`;
      slots.appendChild(row);
      side.players.push($(".player", row));
      side.champs.push($(".champ", row));
    }
    side.team.addEventListener("input", () => onTeam(side, false));
    side.team.addEventListener("change", () => onTeam(side, true));
    for (const p of side.players) p.addEventListener("change", () => { resolvePlayer(p, true); update(); });
    for (const c of side.champs) {
      c.addEventListener("input", update);
      c.addEventListener("change", () => { resolveChamp(c, true); update(); });
    }
    return side;
  }

  // --- resolving what was typed

  function resolveTeam(input, loose) {
    if (teamByLabel[input.value]) return teamByLabel[input.value];
    if (!loose || !input.value.trim()) return null;
    const q = norm(input.value);
    const hits = Object.keys(teamByLabel).filter((l) => norm(l).startsWith(q) || norm(l.split(" · ")[0]) === q);
    const exact = hits.filter((l) => norm(l.split(" · ")[0]) === q);
    const pick = exact.length === 1 ? exact[0] : hits.length === 1 ? hits[0] : null;
    if (pick) input.value = pick;
    return pick ? teamByLabel[pick] : null;
  }

  function onTeam(side, loose) {
    const id = resolveTeam(side.team, loose);
    side.team.classList.toggle("bad", loose && !!side.team.value.trim() && !id);
    if (id && id !== side.teamId) {
      side.teamId = id;
      SNAP.teams[id].roster.forEach((pid, r) => { side.players[r].value = playerLabel[pid] || ""; });
    } else if (!id) {
      side.teamId = null;
    }
    update();
  }

  // a player id, "new:<name>" for someone without history, or null when empty
  function resolvePlayer(input, loose) {
    const v = input.value.trim();
    if (!v) return null;
    if (playerByLabel[v]) return playerByLabel[v];
    const ids = playersByName[norm(v.split(" · ")[0])] || [];
    if (ids.length === 1) {
      if (loose) input.value = playerLabel[ids[0]];
      return ids[0];
    }
    return "new:" + v;
  }

  function resolveChamp(input, loose) {
    const q = norm(input.value);
    if (!q) return null;
    if (champByKey[q]) {
      if (loose) input.value = champByKey[q];
      return champByKey[q];
    }
    const hits = SNAP.champions.filter((c) => norm(c).startsWith(q));
    if (loose && hits.length === 1) {
      input.value = hits[0];
      return hits[0];
    }
    return null;
  }

  function swap() {
    const fields = (s) => [s.team, ...s.players, ...s.champs];
    const b = fields(ui.blue), r = fields(ui.red);
    b.forEach((input, i) => { const v = input.value; input.value = r[i].value; r[i].value = v; });
    [ui.blue.teamId, ui.red.teamId] = [ui.red.teamId, ui.blue.teamId];
    update();
  }

  // --- the result

  function readSide(side) {
    const players = side.players.map((p) => resolvePlayer(p, false));
    const champs = side.champs.map((c) => resolveChamp(c, false));
    side.players.forEach((p, i) => {
      p.classList.toggle("new", !!players[i] && players[i].startsWith("new:"));
      p.title = players[i] && players[i].startsWith("new:") ? "No match history: rated as a newcomer" : "";
    });
    return { team: side.teamId, players: players.map((p) => p || "new:"), champs };
  }

  function update() {
    if (!SNAP) return;
    const blue = readSide(ui.blue), red = readSide(ui.red);
    for (const [side, s] of [[ui.blue, blue], [ui.red, red]]) {
      const t = s.team && SNAP.teams[s.team];
      side.note.textContent = t ? `${t.league} · ${t.games} games · last played ${t.last_game}` : "";
    }
    const all = [...blue.champs, ...red.champs];
    const counts = {};
    all.forEach((c) => { if (c) counts[c] = (counts[c] || 0) + 1; });
    [...ui.blue.champs, ...ui.red.champs].forEach((input, i) => {
      const typed = input.value.trim() !== "";
      input.classList.toggle("bad", (typed && !all[i] && document.activeElement !== input) || (all[i] && counts[all[i]] > 1));
    });

    const out = $("#result");
    if (!blue.team || !red.team) {
      out.innerHTML = `<p class="hint">Pick two teams to see the pre-draft probability. Add all ten champions for the final one.</p>`;
      return;
    }
    if (blue.team === red.team) {
      out.innerHTML = `<p class="warn">Both sides are the same team.</p>`;
      return;
    }
    const nChamps = all.filter(Boolean).length;
    const draftDone = nChamps === 10 && Object.values(counts).every((n) => n === 1);
    const now = today();
    const res = M.predict(SNAP, blue.team, red.team, blue.players, red.players,
                          draftDone ? blue.champs : null, draftDone ? red.champs : null, now);
    const p = draftDone ? res.final : res.pre;
    render(out, blue, red, p, res, draftDone, nChamps, now);
  }

  function group(key) {
    if (key === "team_elo" || key === "league_offset") return "Team rating";
    if (key === "roster_elo" || key.endsWith("_player_elo")) return "Player ratings";
    if (key === "log_team_games" || key.endsWith("_log_player_games")) return "Experience";
    if (key.includes("_log_pc_")) return "Champion comfort";
    return "Champion meta";
  }

  function factors(blue, red, res, draftDone, now) {
    const m = SNAP.models;
    const sums = Object.fromEntries(GROUPS.map((g) => [g, 0]));
    if (draftDone) {
      const k = m.combiner;
      const bf = M.sideFeatures(SNAP, blue.team, blue.players, blue.champs, now);
      const rf = M.sideFeatures(SNAP, red.team, red.players, red.champs, now);
      for (const c of M.contributions(m.post, bf, rf)) sums[group(c.feature)] += k.b * c.value;
      sums["Draft (solo queue)"] = k.w * res.draft;
      sums["Blue side & baseline"] = k.a + k.b * m.post.intercept;
    } else {
      const bf = M.sideFeatures(SNAP, blue.team, blue.players, null, now);
      const rf = M.sideFeatures(SNAP, red.team, red.players, null, now);
      for (const c of M.contributions(m.pre, bf, rf)) sums[group(c.feature)] += c.value;
      sums["Blue side & baseline"] = m.pre.intercept;
      delete sums["Champion comfort"];
      delete sums["Champion meta"];
      delete sums["Draft (solo queue)"];
    }
    return Object.entries(sums);
  }

  function render(out, blue, red, p, res, draftDone, nChamps, now) {
    const bName = SNAP.teams[blue.team].name, rName = SNAP.teams[red.team].name;
    const pct = (x) => (100 * x).toFixed(1) + "%";
    const odds = (x) => (1 / x).toFixed(2);
    const fs = factors(blue, red, res, draftDone, now);
    const max = Math.max(...fs.map(([, v]) => Math.abs(v)), 0.05);
    const slope = p * (1 - p) * 100; // logit -> percentage points near the current probability
    const rows = fs
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .map(([name, v]) => {
        const w = (50 * Math.abs(v) / max).toFixed(1);
        const pp = v * slope;
        return `<div class="factor"><span>${name}</span>
          <span class="track"><span class="fill ${v >= 0 ? "to-blue" : "to-red"}" style="width:${w}%"></span></span>
          <span class="val">${pp >= 0 ? "+" : "−"}${Math.abs(pp).toFixed(1)} pp</span></div>`;
      }).join("");
    const newcomers = [...blue.players, ...red.players].filter((x) => x.startsWith("new:")).length;
    out.innerHTML = `
      <p class="stage">${draftDone ? "After the draft" : `Before the draft · ${nChamps}/10 champions entered`}</p>
      <div class="score">
        <div class="blue"><div class="team">${esc(bName)}</div><div class="pct">${pct(p)}</div>
          <div class="odds">fair odds ${odds(p)}</div></div>
        <div class="red"><div class="team">${esc(rName)}</div><div class="pct">${pct(1 - p)}</div>
          <div class="odds">fair odds ${odds(1 - p)}</div></div>
      </div>
      <div class="bar" role="img" aria-label="${esc(bName)} ${pct(p)}, ${esc(rName)} ${pct(1 - p)}"><span style="width:${(100 * p).toFixed(1)}%"></span></div>
      ${draftDone ? `<p class="pre-note">Before the draft: ${pct(res.pre)} · teams and champions without the draft score: ${pct(res.post)}</p>` : ""}
      ${newcomers ? `<p class="warn">${newcomers} player${newcomers > 1 ? "s" : ""} without match history, rated as newcomers.</p>` : ""}
      <div class="breakdown"><h3>What moves it: bars to the left favor ${esc(bName)}, to the right ${esc(rName)}</h3>${rows}</div>`;
  }

  function esc(s) {
    return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  // The snapshot carries Python's answers for random matchups; this page must reproduce them.
  function selfTest() {
    const el = $("#selftest");
    let ok = 0;
    for (const t of SNAP.tests || []) {
      const [bt, bp, bc] = t.blue, [rt, rp, rc] = t.red;
      const got = M.predict(SNAP, bt, rt, bp, rp, bc, rc, t.now);
      if (Object.keys(t.expected).every((k) => Math.abs(got[k] - t.expected[k]) < 1e-9)) ok++;
    }
    const n = (SNAP.tests || []).length;
    el.textContent = ` Model check: ${ok}/${n} test cases match the Python model.`;
    el.className = ok === n ? "ok" : "fail";
  }
})();
