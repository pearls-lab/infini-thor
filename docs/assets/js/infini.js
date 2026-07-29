/* =============================================================================
   ∞-THOR project page
   Drives (a) the live trajectory readout, (b) the full-bleed -> docked handoff
   that keeps the trajectory playing for as long as the visitor keeps reading,
   and (c) the scroll reveals.

   window.TRAJ is injected by the Jekyll layout from _data/trajectories.json.
   ========================================================================== */
(function () {
  "use strict";

  var TRAJ = window.TRAJ || {};
  var hero = TRAJ.hero || { steps: 1000, subgoals: [], tokens: 0, scene: "FloorPlan230" };
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  /* ------------------------------------------------------------ formatting */
  function pad(n, w) {
    var s = String(n);
    while (s.length < w) s = "0" + s;
    return s;
  }
  function human(n) {
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 2) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e5 ? 0 : 1) + "K";
    return String(n);
  }

  /* ==========================================================================
     1. Live readout
     The <video> is a rendering of the whole trajectory, so playback position
     maps linearly onto the step index. Everything on screen is derived from it.
     ====================================================================== */
  var video = $("#trajVideo");
  var elStep = $("#roStep");
  var elSub = $("#roSub");
  var elTok = $("#roTok");
  var elTickIdx = $("#tickIdx");
  var elTickTxt = $("#tickTxt");
  var elFill = $("#barFill");
  var elBar = $("#bar");
  var elDockStep = $("#dockStep");

  var subgoals = hero.subgoals || [];
  var nSteps = hero.steps || 1;
  var perStepTokens = nSteps ? (hero.tokens || 0) / nSteps : 0;

  // Smoothed replays emit several in-between frames per environment step, so
  // playback position is not linear in step index -- step_frames[i] is the frame
  // each step starts on, and we binary-search it.
  var stepFrames = hero.step_frames && hero.step_frames.length ? hero.step_frames : null;
  var nFrames = hero.n_frames || nSteps + 1;

  function stepAtProgress(p) {
    if (!stepFrames) return Math.min(nSteps, Math.round(p * nSteps));
    var f = p * (nFrames - 1);
    var lo = 0, hi = stepFrames.length - 1;
    if (f < stepFrames[0]) return 0;
    while (lo < hi) {
      var mid = (lo + hi + 1) >> 1;
      if (stepFrames[mid] <= f) lo = mid; else hi = mid - 1;
    }
    return Math.min(nSteps, lo + 1);
  }

  // subgoal tick marks along the progress bar
  if (elBar && subgoals.length) {
    var frag = document.createDocumentFragment();
    subgoals.forEach(function (sg) {
      var t = document.createElement("i");
      t.className = "tick";
      t.style.left = (100 * sg.t / nSteps) + "%";
      frag.appendChild(t);
    });
    elBar.appendChild(frag);
  }

  function subgoalAt(step) {
    var lo = 0;
    for (var i = 0; i < subgoals.length; i++) {
      if (subgoals[i].t <= step) lo = i; else break;
    }
    return lo;
  }

  var lastStep = -1;
  function paint() {
    if (!video || !video.duration || !isFinite(video.duration)) return;
    var p = Math.min(1, video.currentTime / video.duration);
    var step = stepAtProgress(p);
    if (step === lastStep) return;
    lastStep = step;

    if (elStep) elStep.innerHTML = pad(step, String(nSteps).length) +
      '<span class="sub">/' + nSteps + "</span>";
    if (elDockStep) elDockStep.textContent = "t=" + pad(step, String(nSteps).length) + "/" + nSteps;
    if (elTok) elTok.textContent = "≈" + human(Math.round(step * perStepTokens));
    if (elFill) elFill.style.width = (p * 100) + "%";

    if (subgoals.length) {
      var i = subgoalAt(step);
      if (elSub) elSub.innerHTML = (i + 1) + '<span class="sub">/' + subgoals.length + "</span>";
      if (elTickIdx) elTickIdx.textContent = "subgoal " + pad(i + 1, 2);
      if (elTickTxt && subgoals[i].text) elTickTxt.textContent = "“" + subgoals[i].text + "”";
    }
  }

  if (video) {
    // rAF rather than `timeupdate`: the readout should tick smoothly, not 4x/second.
    var loop = function () { paint(); requestAnimationFrame(loop); };
    video.addEventListener("loadedmetadata", paint);
    requestAnimationFrame(loop);

    // Autoplay is best-effort: browsers block it until the visitor interacts, and
    // some data-saver modes never allow it. Retry on the first interaction.
    var tryPlay = function () {
      var pr = video.play();
      if (pr && pr.catch) pr.catch(function () {});
    };
    tryPlay();
    ["pointerdown", "keydown", "touchstart", "scroll"].forEach(function (ev) {
      window.addEventListener(ev, tryPlay, { once: true, passive: true });
    });
    // If the tab was backgrounded, resume rather than sit paused.
    document.addEventListener("visibilitychange", function () {
      if (!document.hidden && video.paused) tryPlay();
    });
  }

  /* ==========================================================================
     2. Full-bleed -> docked handoff
     While the visitor is inside the intro the trajectory owns the whole screen.
     When they read past it the same element shrinks into a corner card and
     keeps playing -- the trajectory outlasts the reading, which is the point.
     ====================================================================== */
  var stage = $("#trajStage");
  var introEnd = $("#introEnd");

  var MIN_W = 186, MIN_H = 34;

  function dockSize() {
    var cs = getComputedStyle(document.documentElement);
    return {
      w: parseInt(cs.getPropertyValue("--dock-w"), 10) || 264,
      h: parseInt(cs.getPropertyValue("--dock-h"), 10) || 149
    };
  }

  var docked = null;

  /* Anchor the stage bottom-right at whichever docked size is current, so
     collapsing to the pill does not leave it floating mid-air. */
  function layout() {
    if (!stage || !docked) return;
    var sz = stage.classList.contains("is-min") ? { w: MIN_W, h: MIN_H } : dockSize();
    var m = window.innerWidth < 780 ? 14 : 24;
    stage.style.left = (window.innerWidth - sz.w - m) + "px";
    stage.style.top = (window.innerHeight - sz.h - m) + "px";
    stage.style.width = sz.w + "px";
    stage.style.height = sz.h + "px";
  }

  function setStage(toDocked) {
    if (!stage || docked === toDocked) return;
    docked = toDocked;
    if (toDocked) {
      layout();
      stage.classList.add("is-docked");
      document.body.classList.add("is-docked-page");
      stage.setAttribute("aria-hidden", "false");
      stage.setAttribute("role", "button");
      stage.setAttribute("tabindex", "0");
      stage.setAttribute("title", "Back to the top — the trajectory is still running");
    } else {
      stage.style.left = "0px";
      stage.style.top = "0px";
      stage.style.width = "100vw";
      stage.style.height = "100vh";
      stage.classList.remove("is-docked");
      stage.classList.remove("is-min");
      document.body.classList.remove("is-docked-page");
      stage.removeAttribute("role");
      stage.removeAttribute("tabindex");
    }
  }

  if (stage && introEnd) {
    setStage(false);
    var onScroll = function () {
      // dock once the end-of-intro marker has travelled above the fold
      setStage(introEnd.getBoundingClientRect().top < window.innerHeight * 0.55);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", function () { if (docked) layout(); else setStage(false); });
    onScroll();

    var back = function () { window.scrollTo({ top: 0, behavior: reduced ? "auto" : "smooth" }); };
    stage.addEventListener("click", function () {
      if (!docked) return;
      if (stage.classList.contains("is-min")) { stage.classList.remove("is-min"); layout(); }
      else back();
    });
    stage.addEventListener("keydown", function (e) {
      if (docked && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); back(); }
    });

    // collapse the docked card to a pill so it never sits on top of the reading
    var minBtn = $("#dockMin");
    if (minBtn) {
      minBtn.addEventListener("click", function (e) {
        e.stopPropagation();
        stage.classList.add("is-min");
        layout();
      });
    }
  }

  /* ==========================================================================
     3. Slit-scan strip: playhead + scrub
     ====================================================================== */
  var strip = $("#strip");
  var stripHead = $("#stripHead");
  var stripTip = $("#stripTip");

  if (strip && video) {
    var seekTo = function (clientX) {
      var r = strip.getBoundingClientRect();
      var f = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
      if (video.duration && isFinite(video.duration)) video.currentTime = f * video.duration;
      return f;
    };
    var showTip = function (clientX) {
      var r = strip.getBoundingClientRect();
      var f = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
      var step = stepAtProgress(f);
      stripTip.style.left = (f * 100) + "%";
      stripTip.textContent = "t=" + step +
        (subgoals.length ? " · " + (subgoals[subgoalAt(step)].text || "subgoal " + (subgoalAt(step) + 1)) : "");
    };
    strip.addEventListener("pointermove", function (e) {
      showTip(e.clientX);
      if (e.buttons === 1) seekTo(e.clientX);
    });
    strip.addEventListener("pointerdown", function (e) { seekTo(e.clientX); });
    strip.addEventListener("keydown", function (e) {
      if (!video.duration) return;
      var d = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
      if (!d) return;
      e.preventDefault();
      video.currentTime = Math.max(0, Math.min(video.duration, video.currentTime + d * video.duration / 40));
    });

    // keep the playhead glued to playback
    var headLoop = function () {
      if (video.duration && isFinite(video.duration)) {
        var f = video.currentTime / video.duration;
        stripHead.style.left = (f * 100) + "%";
        strip.setAttribute("aria-valuenow", stepAtProgress(f));
      }
      requestAnimationFrame(headLoop);
    };
    requestAnimationFrame(headLoop);
  }

  /* ==========================================================================
     4. Reveals + the benchmark-length ruler
     ====================================================================== */
  var io = "IntersectionObserver" in window
    ? new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          e.target.classList.add("in");
          if (e.target.dataset.grow) {
            var f = $(".ruler-fill", e.target);
            if (f) f.style.width = e.target.dataset.grow + "%";
          }
          io.unobserve(e.target);
        });
      }, { rootMargin: "0px 0px -12% 0px", threshold: 0.15 })
    : null;

  $$(".reveal").forEach(function (el) {
    if (io) io.observe(el); else el.classList.add("in");
  });

  // count-up on the headline stat numbers
  var cio = "IntersectionObserver" in window
    ? new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (!e.isIntersecting) return;
          cio.unobserve(e.target);
          var el = e.target;
          var target = parseFloat(el.dataset.count);
          var suffix = el.dataset.suffix || "";
          if (reduced) { el.textContent = el.dataset.pretty || (target + suffix); return; }
          var t0 = null, dur = 1100;
          var tick = function (ts) {
            if (t0 === null) t0 = ts;
            var k = Math.min(1, (ts - t0) / dur);
            var eased = 1 - Math.pow(1 - k, 3);
            el.textContent = (el.dataset.human
              ? human(Math.round(target * eased))
              : Math.round(target * eased).toLocaleString()) + suffix;
            if (k < 1) requestAnimationFrame(tick);
            else el.textContent = el.dataset.pretty || el.textContent;
          };
          requestAnimationFrame(tick);
        });
      }, { threshold: 0.5 })
    : null;
  $$("[data-count]").forEach(function (el) {
    if (cio) cio.observe(el);
    else el.textContent = el.dataset.pretty || el.dataset.count;
  });

  /* --------------------------------------------------- gallery hover-to-play */
  $$(".gal-card video").forEach(function (v) {
    var card = v.closest(".gal-card");
    card.addEventListener("mouseenter", function () { var p = v.play(); if (p && p.catch) p.catch(function () {}); });
    card.addEventListener("mouseleave", function () { v.pause(); });
  });

  /* ------------------------------------------------------------ copy bibtex */
  var cite = $("#citeBtn");
  if (cite) {
    cite.addEventListener("click", function () {
      var txt = $("#bibtex").textContent;
      navigator.clipboard.writeText(txt).then(function () {
        var old = cite.textContent;
        cite.textContent = "copied ✓";
        setTimeout(function () { cite.textContent = old; }, 1600);
      });
    });
  }
})();
