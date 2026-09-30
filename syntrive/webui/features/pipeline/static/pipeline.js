(function () {
  "use strict";

  var dirty = false;

  function refresh(form) {
    var boxes = form.querySelectorAll('input[name="include"]');
    var kept = 0, changes = 0;
    boxes.forEach(function (box) {
      var changed = box.checked !== (box.dataset.initial === "1");
      if (box.checked) kept++;
      if (changed) changes++;
      box.closest(".sel-row").classList.toggle("is-changed", changed);
      box.closest(".sel-row").classList.toggle("is-excluded", !box.checked);
    });
    var count = form.querySelector("[data-count]");
    if (count) count.textContent = String(kept);
    var save = form.querySelector("[data-save]");
    if (save) {
      save.disabled = changes === 0;
      save.textContent = changes ? save.dataset.labelN.replace("{n}", changes) : save.dataset.label;
    }
    dirty = changes > 0;
  }

  document.addEventListener("change", function (event) {
    var form = event.target.closest("form[data-selection]");
    if (form) refresh(form);
  });

  document.addEventListener("click", function (event) {
    var bulk = event.target.closest("[data-bulk]");
    if (bulk) {
      var form = bulk.closest("form[data-selection]");
      var threshold = Number(bulk.dataset.threshold || 0);
      form.querySelectorAll('input[name="include"]').forEach(function (box) {
        if (bulk.dataset.bulk === "all") box.checked = true;
        if (bulk.dataset.bulk === "short" && Number(box.dataset.chars) < threshold) box.checked = false;
      });
      refresh(form);
    }
    if (event.target.closest("[data-close-drawer]")) {
      var drawer = document.getElementById("chapter-drawer");
      if (drawer) drawer.innerHTML = "";
    }
    var item = event.target.closest("[data-review-item]");
    if (item) {
      document.querySelectorAll("[data-review-item][aria-current]").forEach(function (el) { el.removeAttribute("aria-current"); });
      item.setAttribute("aria-current", "true");
    }
  });

  document.addEventListener("submit", function (event) {
    if (event.target.matches("form[data-selection]")) dirty = false;
  });
  document.addEventListener("htmx:beforeRequest", function (event) {
    if (event.detail.elt.matches && event.detail.elt.matches("form[data-selection]")) dirty = false;
  });

  var changedFields = new WeakMap();

  function markChanged(form, name) {
    if (!form || !name) return;
    var set = changedFields.get(form) || new Set();
    set.add(name);
    changedFields.set(form, set);
    var label = form.querySelector("[data-unsaved]");
    if (label) label.textContent = label.dataset.labelN.replace("{n}", set.size);
  }

  function anyChanged() {
    return Array.prototype.some.call(document.querySelectorAll("form[data-dirty-form]"), function (f) {
      var set = changedFields.get(f);
      return set && set.size > 0;
    });
  }

  document.addEventListener("change", function (event) {
    var form = event.target.closest("form[data-dirty-form]");
    if (form && event.target.name) markChanged(form, event.target.name);
    if (event.target.matches("[data-gate-source]")) {
      var gated = event.target.form && event.target.form.querySelector("[data-gated]");
      if (gated) gated.disabled = event.target.value !== "by-duration";
    }
  });
  document.addEventListener("htmx:beforeRequest", function (event) {
    var form = event.detail.elt.closest && event.detail.elt.closest("form[data-dirty-form]");
    if (form && event.detail.elt === form) changedFields.delete(form);
  });

  function activeRole(form) {
    return form.querySelector(".cast-role.is-active");
  }

  function refreshInUse(form) {
    var role = activeRole(form);
    if (!role) return;
    var bound = form.querySelector('input[name="voice_' + role.dataset.role + '"]').value;
    form.querySelectorAll(".voice-row").forEach(function (row) {
      row.classList.toggle("is-used", row.dataset.ref === bound);
    });
    var name = form.querySelector("[data-active-role-name]");
    if (name) name.textContent = role.dataset.roleName;
  }

  var player = null;
  function stopPlayer() {
    if (!player) return;
    player.audio.pause();
    player.button.classList.remove("is-playing");
    player = null;
  }

  document.addEventListener("click", function (event) {
    var tab = event.target.closest("[data-subtab]");
    if (tab) {
      var scope = tab.closest("form");
      scope.querySelectorAll("[data-subtab]").forEach(function (b) { b.setAttribute("aria-selected", String(b === tab)); });
      scope.querySelectorAll("[data-subpanel]").forEach(function (p) { p.hidden = p.dataset.subpanel !== tab.dataset.subtab; });
    }
    var pick = event.target.closest("[data-select-role]");
    if (pick) {
      var form = pick.closest("form");
      form.querySelectorAll(".cast-role").forEach(function (r) { r.classList.toggle("is-active", r.dataset.role === pick.dataset.selectRole); });
      refreshInUse(form);
    }
    var use = event.target.closest("[data-use]");
    if (use) {
      var f = use.closest("form");
      var role = activeRole(f);
      if (role) {
        var input = f.querySelector('input[name="voice_' + role.dataset.role + '"]');
        input.value = use.dataset.use;
        role.querySelector("[data-bound-label]").textContent = use.closest(".voice-row").dataset.label;
        markChanged(f, input.name);
        refreshInUse(f);
      }
    }
    var chip = event.target.closest("[data-gender]");
    if (chip && chip.classList.contains("chip-btn")) {
      var list = chip.closest(".voice-list");
      list.querySelectorAll(".chip-btn").forEach(function (c) { c.setAttribute("aria-pressed", String(c === chip)); });
      list.querySelectorAll(".voice-row").forEach(function (row) {
        row.hidden = Boolean(chip.dataset.gender) && row.dataset.gender !== chip.dataset.gender;
      });
    }
    var play = event.target.closest("[data-play]");
    if (play) {
      var same = player && player.button === play;
      stopPlayer();
      if (!same) {
        var audio = document.getElementById("voice-player");
        audio.src = play.dataset.play;
        audio.onended = stopPlayer;
        audio.play();
        play.classList.add("is-playing");
        player = { audio: audio, button: play };
      }
    }
  });

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("form.tts").forEach(refreshInUse);
  });

  window.addEventListener("beforeunload", function (event) {
    if (!dirty && !anyChanged()) return;
    event.preventDefault();
    event.returnValue = "";
  });
})();
