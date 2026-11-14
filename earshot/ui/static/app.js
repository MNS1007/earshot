// Earshot live screen. Receives the full state over /ws (falls back to polling /api/state)
// and redraws only the parts whose data changed. All text goes in through textContent.
(function () {
  "use strict";

  var MODES = {
    idle:          { label: "Ready",                 sub: "Say “watch me” to teach a task" },
    teaching:      { label: "Watching you",          sub: "Do the task once and say what you are doing" },
    understanding: { label: "Working out the steps", sub: "Matching what the camera saw with what you said" },
    helping:       { label: "Helping you",           sub: "Handing you the next item" },
    doing:         { label: "Doing it myself",       sub: "Correct me by voice at any time" },
    paused:        { label: "Paused",                sub: "Arm stopped" }
  };
  var WAS = { teaching: "watching you", understanding: "working out the steps", helping: "helping you", doing: "doing it myself" };
  var STAGES = ["teaching", "understanding", "helping", "doing"];

  // What a spoken sentence led to, in plain words.
  var LED_TO = {
    start_teaching: "started watching", stop_teaching: "stopped watching",
    help_me: "started helping", your_turn: "started doing it myself",
    pause: "paused", resume: "resumed", load_task: "loaded a saved task",
    swap_steps: "swapped two steps", move_step: "changed the order",
    change_destination: "changed where it goes", set_grip: "changed the grip", set_speed: "changed the speed",
    retry_last: "trying the last step again", skip_step: "skipped a step",
    tap: "band tapped", double_tap: "band tapped twice"
  };
  var SILENT = { transcript: 1, nothing: 1 };
  var CORRECTING = { swap_steps: 1, move_step: 1, change_destination: 1, set_grip: 1, set_speed: 1, retry_last: 1, skip_step: 1 };
  var KIND_GLYPH = { heard: "g-ear", correction: "g-ear", seen: "g-eye", check: "g-eye", robot: "g-arm", error: "g-warn" };
  var MAX_EVENTS = 5;
  var FROM_RULE = "Not in the demonstration";
  var DISAGREE = "Picture and words disagreed. Following the picture.";
  var SVG = "http://www.w3.org/2000/svg";

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }
  function glyph(id, cls) {
    var s = document.createElementNS(SVG, "svg");
    s.setAttribute("class", cls || "glyph");
    s.setAttribute("aria-hidden", "true");
    var u = document.createElementNS(SVG, "use");
    u.setAttribute("href", "#" + id);
    s.appendChild(u);
    return s;
  }
  function two(n) { return (n < 10 ? "0" : "") + n; }
  function clock(t, seconds) {
    var d = new Date(t * 1000);
    return two(d.getHours()) + ":" + two(d.getMinutes()) + (seconds ? ":" + two(d.getSeconds()) : "");
  }
  function cap(s) { s = String(s || "").trim(); return s ? s.charAt(0).toUpperCase() + s.slice(1) : s; }
  function stop(s) { s = String(s || "").trim(); return /[.!?]$/.test(s) ? s : s + "."; }
  function setText(node, text) { if (node.textContent !== text) node.textContent = text; }
  function setHidden(node, hide) { if (node.hidden !== hide) node.hidden = hide; }

  // ---------------------------------------------------------------- reading the event log

  // "“Watch me.” → start_teaching", "→ tap", "[button] Help me"
  function parseHeard(text) {
    var m = /^[“"](.*)[”"]\s*(?:→|->)\s*(\S+)\s*$/.exec(text);
    if (m) return { said: m[1], action: m[2] };
    m = /^(?:→|->)\s*(\S+)\s*$/.exec(text);
    if (m) return { said: "", action: m[1] };
    m = /^\[button\]\s*(.*)$/.exec(text);
    if (m) return { button: m[1] };
    return null;
  }
  function parseCorrection(text) {
    var m = /^[“"](.*)[”"]\s*(?:→|->)\s*(.*)$/.exec(text);
    return m ? { said: m[1], effect: m[2] } : null;
  }
  function same(a, b) {
    var norm = function (x) { return String(x).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim(); };
    return norm(a) === norm(b);
  }

  // The short list under the sheet: newest first, each spoken sentence once.
  function recentEvents(evs) {
    var out = [];
    for (var i = evs.length - 1; i >= 0 && out.length < MAX_EVENTS; i--) {
      var e = evs[i], text = String(e.text || ""), row = { t: e.t, kind: e.kind, text: text, led: "" };
      if (e.kind === "heard") {
        var h = parseHeard(text);
        if (h && h.button !== undefined) {
          row.text = h.button + " button pressed"; row.kind = "button";
        } else if (h) {
          if (SILENT[h.action]) continue;
          if (CORRECTING[h.action] && h.said && laterCorrection(evs, i, h.said)) continue;
          row.led = LED_TO[h.action] || h.action.replace(/_/g, " ");
          row.text = h.said ? "“" + h.said + "”" : cap(row.led);
          if (!h.said) row.led = "";
        } else if (/(?:→|->)\s*(transcript|nothing)\s*$/.test(text)) {
          continue;
        }
      } else if (e.kind === "correction") {
        var c = parseCorrection(text);
        if (c) {
          if (c.said && c.said !== "(button)") { row.text = "“" + c.said + "”"; row.led = c.effect; }
          else row.text = cap(c.effect);
        }
      } else if (e.kind === "check") {
        if (/not right|cannot see|still on the table|closed on nothing|disagree/i.test(text)) row.miss = true;
      }
      out.push(row);
    }
    return out;
  }
  function laterCorrection(evs, i, said) {
    for (var j = i + 1; j < evs.length && j <= i + 8; j++) {
      if (evs[j].kind !== "correction") continue;
      var c = parseCorrection(String(evs[j].text || ""));
      if (c && same(c.said, said)) return true;
    }
    return false;
  }

  // Why step n last missed its camera check, read from the log (the step itself carries no reason).
  // Stops at the start of the current run, and at a passed check, so an old miss is never shown.
  function missReason(step, evs) {
    var n = String(step.n);
    for (var i = evs.length - 1; i >= 0 && i >= evs.length - 80; i--) {
      var e = evs[i], text = String(e.text || "");
      if (e.kind === "heard") {
        var h = parseHeard(text);
        var act = h && (h.action || String(h.button || "").toLowerCase().replace(/ /g, "_"));
        if (act === "help_me" || act === "your_turn") return null;
        continue;
      }
      if (e.kind === "correction") return null;      // steps may have been renumbered since

      if (e.kind !== "check") continue;
      var m = /^Step (\d+) (checked|not right): (.*)$/.exec(text);
      if (m) {
        if (m[1] !== n) continue;
        return m[2] === "checked" ? null : m[3];
      }
      m = /^Step (\d+): (.*)$/.exec(text);
      if (m) {
        if (m[1] !== n || /disagree/i.test(m[2])) continue;
        return /\bOK\.?$/.test(m[2]) ? null : m[2];
      }
      if (step.status === "failed" && /^Cannot see /i.test(text) &&
          (text.indexOf(step.item) >= 0 || text.indexOf(step.destination) >= 0)) return text;
    }
    return null;
  }

  // ---------------------------------------------------------------- drawing

  var lastVersion = -1;
  var keys = {};                       // last drawn data per region, so nothing is rebuilt needlessly
  function changed(name, value) {
    var k = typeof value === "string" ? value : JSON.stringify(value);
    if (keys[name] === k) return false;
    keys[name] = k;
    return true;
  }

  function draw(s) {
    if (!s || typeof s !== "object") return;
    if (typeof s.version === "number") {
      if (s.version < lastVersion && lastVersion - s.version < 1000) return;  // late poll after a newer push
      lastVersion = s.version;
    }
    var mode = MODES[s.mode] ? s.mode : "idle";
    var task = s.task || null;
    var evs = s.events || [];
    drawMode(mode, s, task);
    drawHeader(task);
    drawSheet(mode, task, evs);
    drawHearing(s.transcript || {});
    drawScore(s.counters || {});
    drawEvents(evs);
  }

  function drawMode(mode, s, task) {
    var from = s.paused_from && MODES[s.paused_from] ? s.paused_from : null;
    if (changed("mode", [mode, from, !!task])) {
      document.body.dataset.mode = mode;
      document.body.dataset.task = task ? "yes" : "no";
      var sub = MODES[mode].sub;
      if (mode === "idle" && task) sub = "Say “help me” or “your turn”";
      if (mode === "paused" && WAS[from]) sub = "Arm stopped. I was " + WAS[from];
      setText($("mode-label"), MODES[mode].label);
      setText($("mode-sub"), sub);

      var stage = mode === "paused" ? from : mode;
      var at = STAGES.indexOf(stage);
      if (at < 0 && task) at = 2;                    // a task is known: stages 1 and 2 are behind us
      var live = STAGES.indexOf(stage) >= 0;
      var items = $("stages").children;
      for (var i = 0; i < items.length; i++) {
        var cls = i < at ? "past" : (i === at && live ? (mode === "paused" ? "now held" : "now") : "");
        if (items[i].className !== cls) items[i].className = cls;
        if (i === at && live) items[i].setAttribute("aria-current", "step"); else items[i].removeAttribute("aria-current");
      }

      // the buttons that make sense now are large; the rest stay available in the small row
      var main = mode === "teaching" ? ["stop_teaching"]
        : mode === "understanding" ? []
        : mode === "helping" || mode === "doing" ? ["pause"]
        : mode === "paused" ? ["resume"]
        : task ? ["help_me", "your_turn"] : ["start_teaching"];
      eachButton("main-buttons", function (b) { setHidden(b, main.indexOf(b.dataset.action) < 0); });
      eachButton("other-buttons", function (b) { setHidden(b, main.indexOf(b.dataset.action) >= 0); });
      setHidden($("no-main"), main.length > 0);
    }
    setText($("robot-status"), s.robot_status || "");
  }
  function eachButton(id, fn) {
    var bs = $(id).querySelectorAll("button[data-action]");
    for (var i = 0; i < bs.length; i++) fn(bs[i]);
  }

  function drawHeader(task) {
    if (!changed("header", task ? [task.name, task.task_id, task.taught_at] : null)) return;
    var name = $("taskname");
    name.textContent = "";
    if (!task) { name.appendChild(el("span", "quiet", "No task yet")); return; }
    name.appendChild(el("span", "name", task.name || task.task_id || "Task"));
    if (task.taught_at) name.appendChild(el("span", "quiet", "taught " + clock(task.taught_at)));
  }

  // ---- the instruction sheet

  var stepNodes = [];                  // one per step, in order: { li, mark, fault, status, markKey }
  var density = 0;                     // 0 roomy ... 3 tightest
  var DENSITIES = ["steps", "steps snug", "steps tight open", "steps tight"];

  function drawSheet(mode, task, evs) {
    var steps = (task && task.steps) || [];
    var has = steps.length > 0;

    if (changed("sheetState", [mode, !!task, has])) {
      setHidden($("steps-wrap"), !has);
      setHidden($("invite"), has);
      var big = "Say “watch me”, then do the task once.";
      var small = "Talk while you do it. The steps will be written here, each with what the camera saw and what you said.";
      var working = false;
      if (task && !has) {
        big = "No steps found in that demonstration.";
        small = "Say “watch me” and show it again.";
      } else if (mode === "teaching") {
        big = "Watching. Do the task once and say what you are doing.";
        small = "Say “that’s it” when you have finished.";
      } else if (mode === "understanding") {
        big = "Working out the steps from the video and your words, about 30 seconds.";
        small = "";
        working = true;
      }
      setText($("invite-big"), big);
      setText($("invite-small"), small);
      setHidden($("invite-small"), !small);
      setHidden($("working"), !working);
    }

    var rs = (task && task.rules) || [];
    if (changed("rules", rs)) {
      var rules = $("rules");
      rules.textContent = "";
      setHidden(rules, rs.length === 0);
      rs.forEach(function (r) {
        var d = el("div", "rule-line");
        d.appendChild(el("span", "k", "Rule"));
        d.appendChild(el("span", "txt", cap(r.when)));
        d.appendChild(glyph("g-to", "to"));
        d.appendChild(el("span", "sr", " go to "));
        d.appendChild(el("span", "txt", r.destination));
        rules.appendChild(d);
      });
    }

    var cs = (task && task.corrections) || [];
    if (changed("corrections", cs)) {
      setHidden($("corrections"), cs.length === 0);
      var ul = $("corrections-list");
      ul.textContent = "";
      cs.slice(-2).forEach(function (c) {       // the last two fit; older ones are in the saved task
        var li = el("li");
        li.appendChild(el("span", "t", clock(c.at)));
        if (c.said && c.said !== "(button)") li.appendChild(el("span", "said", "“" + c.said + "”"));
        li.appendChild(el("span", "effect", c.effect));
        ul.appendChild(li);
      });
    }

    var rebuilt = false;
    if (changed("steps", steps.map(function (st) { return [st.n, st.item, st.destination, st.seen, st.heard, st.agrees]; }))) {
      buildSteps(steps);
      rebuilt = true;
    }

    var done = 0, current = null, touched = rebuilt;
    steps.forEach(function (st, i) {
      var node = stepNodes[i];
      var status = /^(todo|doing|done|failed)$/.test(st.status) ? st.status : "todo";
      if (status === "done") done++;
      var reason = status === "doing" || status === "failed" ? missReason(st, evs) : null;
      var markKey = status + "|" + (reason || "");
      if (node.markKey !== markKey) {
        var was = node.status;
        node.markKey = markKey;
        node.status = status;
        node.li.className = "step " + status + (status === "done" && was && was !== "done" && !rebuilt ? " just" : "");
        drawMark(node, status, reason);
        touched = true;
      }
      if (status === "doing") current = node;
    });
    setText($("progress"), has ? done + " of " + steps.length + " done" : "");

    var sized = changed("wrapSize", [window.innerWidth, window.innerHeight, $("steps-wrap").clientHeight]);
    if (touched || sized) layoutSteps(current, rebuilt || sized);
  }

  function buildSteps(steps) {
    var list = $("steps");
    list.textContent = "";
    stepNodes = steps.map(function (st) {
      var li = el("li", "step todo");
      li.appendChild(el("div", "num", String(st.n)));
      var what = el("div", "what");
      what.appendChild(el("span", "item", cap(st.item)));
      what.appendChild(glyph("g-to", "to"));
      what.appendChild(el("span", "sr", " goes to "));
      what.appendChild(el("span", "dest", st.destination));
      li.appendChild(what);
      var mark = el("div", "mark");
      li.appendChild(mark);

      var pair = el("div", "pair");
      if (String(st.seen || "").trim() === FROM_RULE) {
        var because = String(st.heard || "").replace(/^rule:\s*/i, "");
        var r = el("div", "ev rule");
        r.appendChild(el("span", "k", because ? "From the rule:" : "From the rule"));
        if (because) r.appendChild(el("span", "v", because));
        pair.appendChild(r);
      } else {
        pair.appendChild(evidence("seen", "g-eye", "Seen", st.seen, "nothing clear in the picture"));
        pair.appendChild(evidence("heard", "g-ear", "Heard", st.heard ? "“" + st.heard + "”" : "", "nothing said"));
      }
      li.appendChild(pair);

      if (st.agrees === false) {
        var d = el("div", "note");
        d.appendChild(glyph("g-warn"));
        d.appendChild(el("span", null, DISAGREE));
        li.appendChild(d);
      }
      var fault = el("div", "note");
      fault.hidden = true;
      fault.appendChild(glyph("g-cross"));
      fault.appendChild(el("span"));
      li.appendChild(fault);

      list.appendChild(li);
      return { li: li, mark: mark, fault: fault, status: null, markKey: null };
    });
    density = 0;
  }

  function evidence(kind, glyphId, label, text, missing) {
    var d = el("div", "ev " + kind + (text ? "" : " none"));
    d.appendChild(glyph(glyphId));
    d.appendChild(el("span", "k", label));
    d.appendChild(el("span", "v", text || missing));
    return d;
  }

  function drawMark(node, status, reason) {
    var mark = node.mark;
    mark.textContent = "";
    if (status === "done") {
      var s = document.createElementNS(SVG, "svg");
      s.setAttribute("viewBox", "0 0 24 24");
      s.setAttribute("aria-hidden", "true");
      var p = document.createElementNS(SVG, "path");
      p.setAttribute("class", "stroke");
      p.setAttribute("d", "M4 12.5l5.5 5.5L20 6.5");
      p.setAttribute("pathLength", "1");
      s.appendChild(p);
      mark.appendChild(s);
      mark.appendChild(el("span", "sr", "Done"));
    } else if (status === "failed") {
      mark.appendChild(el("span", null, "Did not work"));
    } else if (status === "doing") {
      mark.appendChild(el("span", null, reason ? "Second try" : "Now"));
    }
    var text = "";
    if (status === "failed") text = reason ? "Check failed: " + stop(reason) : "The camera check failed.";
    if (status === "doing" && reason) text = "First try missed: " + stop(reason) + " Trying again.";
    node.fault.lastChild.textContent = text;
    setHidden(node.fault, !text);
  }

  // Pick the roomiest density that fits, keep the current step in view, and lay the tape on it.
  function layoutSteps(current, fresh) {
    var wrap = $("steps-wrap"), list = $("steps"), tape = $("tape");
    if (wrap.hidden) { tape.hidden = true; return; }
    if (fresh) density = 0;
    list.className = DENSITIES[density];
    while (list.offsetHeight > wrap.clientHeight + 1 && density < DENSITIES.length - 1) {
      density++;
      list.className = DENSITIES[density];
    }
    if (current) {
      var top = current.li.offsetTop, h = current.li.offsetHeight;
      var room = wrap.clientHeight;
      var want = list.offsetHeight <= room ? 0 : Math.max(0, Math.min(list.offsetHeight - room, top - (room - h) / 2));
      list.style.transform = want ? "translateY(" + (-want) + "px)" : "";
      var jump = fresh || tape.hidden;
      tape.classList.toggle("jump", jump);
      tape.hidden = false;
      tape.style.height = h + "px";
      tape.style.transform = "translateY(" + (top - want) + "px)";
      if (jump) { void tape.offsetHeight; tape.classList.remove("jump"); }
    } else {
      tape.hidden = true;
      if (list.offsetHeight <= wrap.clientHeight) list.style.transform = "";
    }
  }

  // ---- the lines under the camera and under the sheet

  function drawHearing(tr) {
    var text = String(tr.text || "").trim();
    if (!changed("hearing", [text, !!tr.final])) return;
    $("hearing").className = "hearing" + (text ? (tr.final ? "" : " live") : " quiet");
    setText($("transcript"), text ? "“" + text + (tr.final ? "”" : "") : "Nothing yet");
  }

  function drawScore(c) {
    var a = c.attempted || 0, ok = c.succeeded || 0, r = c.retried || 0;
    if (!changed("score", [a, ok, r])) return;
    var p = $("score");
    p.textContent = "";
    if (!a) return;                                     // nothing tried yet: say nothing
    p.appendChild(el("span", "n", ok + " of " + a));
    p.appendChild(document.createTextNode(a === 1 ? " pick worked" : " picks worked"));
    if (r) p.appendChild(el("span", "retry", ", " + r + " needed a second try"));
  }

  function drawEvents(evs) {
    var rows = recentEvents(evs);
    if (!changed("events", rows)) return;
    var ul = $("events");
    ul.textContent = "";
    rows.forEach(function (row) {
      var li = el("li", row.kind + (row.miss ? " miss" : ""));
      li.appendChild(el("span", "t", clock(row.t, true)));
      li.appendChild(glyph(KIND_GLYPH[row.kind] || "g-arm"));
      li.appendChild(el("span", "txt", row.text));
      if (row.led) li.appendChild(el("span", "led", row.led));
      ul.appendChild(li);
    });
  }

  // ---------------------------------------------------------------- connection

  var socket = null, connected = false, lastState = null;

  function setConn(ok) {
    connected = ok;
    setHidden($("conn"), ok);
  }
  function show(s) { lastState = s; draw(s); }

  function connect() {
    try {
      socket = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
    } catch (err) { setTimeout(connect, 1000); return; }
    socket.onopen = function () { setConn(true); };
    socket.onmessage = function (m) { try { show(JSON.parse(m.data)); } catch (err) { /* ignore one bad frame */ } };
    socket.onclose = function () { setConn(false); lastVersion = -1; setTimeout(connect, 1000); };
    socket.onerror = function () { try { socket.close(); } catch (err) { /* already closed */ } };
  }

  function poll() {   // keeps the screen alive while the socket is down
    if (connected) return;
    fetch("/api/state", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (s) { if (!connected) { lastVersion = -1; show(s); } })
      .catch(function () { /* server is down; the header already says Reconnecting */ });
  }

  // ---------------------------------------------------------------- camera, about 5 frames a second

  function nextFrame() {
    var img = $("frame");
    var started = Date.now();
    var again = function () { setTimeout(nextFrame, Math.max(0, 200 - (Date.now() - started))); };
    var loader = new Image();
    loader.onload = function () { img.src = loader.src; again(); };
    loader.onerror = function () { setTimeout(nextFrame, 1000); };
    loader.src = "/frame.jpg?t=" + started;
  }

  // ---------------------------------------------------------------- buttons

  function send(action, args, button) {
    fetch("/api/command", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: action, args: args || {} })
    }).then(function (r) { flash(button, r.ok ? "sent" : "failed"); })
      .catch(function () { flash(button, "failed"); });
  }
  function flash(button, cls) {
    if (!button) return;
    button.classList.add(cls);
    setTimeout(function () { button.classList.remove(cls); }, 600);
  }

  $("controls").addEventListener("click", function (ev) {
    var b = ev.target.closest("button[data-action]");
    if (b) send(b.dataset.action, {}, b);
  });
  $("load-form").addEventListener("submit", function (ev) {
    ev.preventDefault();
    var q = $("load-query").value.trim();
    send("load_task", { query: q || "the last one" }, ev.target.querySelector("button"));
  });

  window.addEventListener("resize", function () { if (lastState) { lastVersion = -1; draw(lastState); } });

  connect();
  poll();
  setInterval(poll, 1000);
  nextFrame();
})();
