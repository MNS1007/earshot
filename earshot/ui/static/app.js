// Earshot live screen. Receives the full state over /ws (falls back to polling /api/state)
// and redraws. All text goes in through textContent, never innerHTML.
(function () {
  "use strict";

  var MODES = {
    idle:          { label: "READY",            sub: "Waiting for a task" },
    teaching:      { label: "WATCHING YOU",     sub: "Do the task once and talk it through" },
    understanding: { label: "UNDERSTANDING",    sub: "Matching what I saw with what I heard" },
    helping:       { label: "HELPING",          sub: "I hand you the next item" },
    doing:         { label: "DOING IT MYSELF",  sub: "Correct me by voice at any time" },
    paused:        { label: "PAUSED",           sub: "Arm stopped" }
  };
  var STAGES = ["teaching", "understanding", "helping", "doing"];
  var STATUS_WORDS = { todo: "TO DO", doing: "NOW", done: "DONE", failed: "FAILED" };
  var DISAGREE = "Picture and words disagree; using the picture";

  function $(id) { return document.getElementById(id); }
  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  }
  function two(n) { return (n < 10 ? "0" : "") + n; }
  function clock(t, seconds) {
    var d = new Date(t * 1000);
    return two(d.getHours()) + ":" + two(d.getMinutes()) + (seconds ? ":" + two(d.getSeconds()) : "");
  }

  // ---------------------------------------------------------------- drawing

  var lastVersion = -1, lastCurrent = null;

  function draw(s) {
    if (!s || typeof s !== "object") return;
    if (typeof s.version === "number") {
      if (s.version < lastVersion && lastVersion - s.version < 1000) return;  // late poll after a newer push
      lastVersion = s.version;
    }
    drawMode(s);
    drawTask(s.task);
    drawRight(s);
  }

  function drawMode(s) {
    var mode = MODES[s.mode] ? s.mode : "idle";
    document.body.dataset.mode = mode;
    $("mode-label").textContent = MODES[mode].label;
    var sub = MODES[mode].sub;
    if (mode === "paused" && s.paused_from && MODES[s.paused_from] && s.paused_from !== "idle") {
      sub = "Arm stopped. Was: " + MODES[s.paused_from].label;
    }
    $("mode-sub").textContent = sub;
    var stage = mode === "paused" ? s.paused_from : mode;
    var at = STAGES.indexOf(stage);
    var items = $("stages").children;
    for (var i = 0; i < items.length; i++) {
      var idx = STAGES.indexOf(items[i].dataset.stage);
      items[i].className = at < 0 ? "" : idx < at ? "past" : idx === at ? "now" : "";
    }
    $("robot-status").textContent = s.robot_status || " ";
  }

  function drawTask(task) {
    var name = $("taskname");
    name.textContent = "";
    if (!task) {
      name.appendChild(el("span", "dim", "No task yet"));
    } else {
      name.appendChild(document.createTextNode(task.name || task.task_id));
      var bits = [];
      if (task.kind) bits.push(task.kind);
      if (task.taught_at) bits.push("taught " + clock(task.taught_at));
      if (bits.length) name.appendChild(el("span", "dim", "  ·  " + bits.join("  ·  ")));
    }

    // rules
    var rules = $("rules");
    rules.textContent = "";
    var rs = (task && task.rules) || [];
    rules.hidden = rs.length === 0;
    rs.forEach(function (r) {
      var d = el("div", "rule");
      d.appendChild(el("span", "k", "RULE"));
      d.appendChild(document.createTextNode(r.when));
      d.appendChild(el("span", "arrow", "→"));
      d.appendChild(document.createTextNode(r.destination));
      rules.appendChild(d);
    });

    // steps
    var list = $("steps");
    var keepScroll = list.scrollTop;
    list.textContent = "";
    var steps = (task && task.steps) || [];
    var done = 0, current = null, currentNode = null;
    steps.forEach(function (st) {
      var status = STATUS_WORDS[st.status] ? st.status : "todo";
      if (status === "done") done++;
      var li = el("li", "step " + status + (st.agrees === false ? " disagree" : ""));
      li.appendChild(el("div", "num", status === "done" ? "✓" : status === "failed" ? "✕" : String(st.n)));
      var body = el("div", "body");
      var what = el("div", "what");
      what.appendChild(document.createTextNode((status === "done" || status === "failed" ? st.n + ". " : "") + st.item));
      what.appendChild(el("span", "arrow", "→"));
      what.appendChild(document.createTextNode(st.destination));
      body.appendChild(what);
      body.appendChild(evidence("seen", "Seen:", st.seen, "nothing clear in the picture"));
      body.appendChild(evidence("heard", "Heard:", st.heard, "nothing said"));
      li.appendChild(body);
      li.appendChild(el("div", "tag", STATUS_WORDS[status]));
      if (st.agrees === false) li.appendChild(el("div", "flag", DISAGREE));
      list.appendChild(li);
      if (status === "doing" || (status === "failed" && current === null)) { current = st.n; currentNode = li; }
    });
    if (!steps.length) {
      list.appendChild(el("li", "empty", task ? "No steps found yet." : "Show me a task and the steps will appear here."));
    }
    $("steps-label").textContent = steps.length ? "STEPS  ·  " + done + " of " + steps.length + " done" : "STEPS";
    list.scrollTop = keepScroll;
    if (currentNode && current !== lastCurrent && list.scrollHeight > list.clientHeight) {
      currentNode.scrollIntoView({ block: "nearest" });
    }
    lastCurrent = current;

    // corrections, oldest first so they read as a history; only the last three fit
    var cs = (task && task.corrections) || [];
    $("corrections").hidden = cs.length === 0;
    var ul = $("corrections-list");
    ul.textContent = "";
    cs.slice(-3).forEach(function (c) {
      var li = el("li");
      li.appendChild(el("span", "t", clock(c.at)));
      li.appendChild(document.createTextNode("You said "));
      li.appendChild(el("span", "said", "‘" + c.said + "’"));
      li.appendChild(el("span", "arrow", "→"));
      li.appendChild(el("span", "effect", c.effect));
      li.title = c.said + " -> " + c.effect;
      ul.appendChild(li);
    });
  }

  function evidence(kind, label, text, missing) {
    var d = el("div", "ev " + kind + (text ? "" : " none"));
    d.appendChild(el("span", "k", label));
    d.appendChild(document.createTextNode(text ? (kind === "heard" ? "“" + text + "”" : text) : missing));
    if (text) d.title = text;
    return d;
  }

  function drawRight(s) {
    var tr = s.transcript || {};
    var box = $("transcript");
    var text = (tr.text || "").trim();
    box.className = "transcript" + (text ? (tr.final ? "" : " live") : " quiet");
    box.textContent = text || "(nothing heard yet)";

    var c = s.counters || {};
    var a = c.attempted || 0;
    $("counters").textContent = (c.succeeded || 0) + " of " + a + (a === 1 ? " pick, " : " picks, ") + (c.retried || 0) + " retried";

    var log = $("log");
    var atTop = log.scrollTop < 4;
    log.textContent = "";
    var evs = s.events || [];
    for (var i = evs.length - 1; i >= 0 && i >= evs.length - 60; i--) {
      var e = evs[i];
      var li = el("li", e.kind);
      li.appendChild(el("span", "t", clock(e.t, true)));
      var txt = el("span", "txt");
      txt.appendChild(el("span", "k", e.kind));
      txt.appendChild(document.createTextNode(e.text));
      li.appendChild(txt);
      log.appendChild(li);
    }
    if (atTop) log.scrollTop = 0;
  }

  // ---------------------------------------------------------------- connection

  var socket = null, connected = false;

  function setConn(ok) {
    connected = ok;
    var c = $("conn");
    c.textContent = ok ? "LIVE" : "DISCONNECTED";
    c.className = "conn " + (ok ? "ok" : "bad");
  }

  function connect() {
    try {
      socket = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
    } catch (err) { setTimeout(connect, 1000); return; }
    socket.onopen = function () { setConn(true); };
    socket.onmessage = function (m) { try { draw(JSON.parse(m.data)); } catch (err) { /* ignore one bad frame */ } };
    socket.onclose = function () { setConn(false); lastVersion = -1; setTimeout(connect, 1000); };
    socket.onerror = function () { try { socket.close(); } catch (err) { /* already closed */ } };
  }

  function poll() {   // keeps the screen alive if the socket cannot connect
    if (connected) return;
    fetch("/api/state", { cache: "no-store" })
      .then(function (r) { return r.json(); })
      .then(function (s) { if (!connected) { lastVersion = -1; draw(s); } })
      .catch(function () { /* server is down; the marker already says so */ });
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

  $("buttons").addEventListener("click", function (ev) {
    var b = ev.target.closest("button[data-action]");
    if (b) send(b.dataset.action, {}, b);
  });
  $("load-form").addEventListener("submit", function (ev) {
    ev.preventDefault();
    var q = $("load-query").value.trim();
    send("load_task", { query: q || "the last one" }, ev.target.querySelector("button"));
  });

  connect();
  poll();
  setInterval(poll, 1000);
  nextFrame();
})();
