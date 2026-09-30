(function () {
  "use strict";

  function refresh(form) {
    var n = form.querySelectorAll('input[name="chapter"][data-action="enqueue"]:checked').length;
    var button = form.querySelector("[data-enqueue]");
    if (!button) return;
    button.textContent = n ? button.dataset.labelN.replace("{n}", n) : button.dataset.label;
    button.disabled = n === 0;
    form.querySelectorAll("[data-applies]").forEach(function (b) {
      b.disabled = !form.querySelector('input[name="chapter"][data-action="' + b.dataset.applies + '"]:checked');
    });
  }

  document.addEventListener("change", function (event) {
    var form = event.target.closest("form[data-queue-form]");
    if (form) refresh(form);
  });

  document.addEventListener("click", function (event) {
    var all = event.target.closest("[data-select-ready]");
    if (!all) return;
    var form = all.closest("form[data-queue-form]");
    form.querySelectorAll('input[name="chapter"][data-action="enqueue"]').forEach(function (box) { box.checked = true; });
    refresh(form);
  });

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll("form[data-queue-form]").forEach(refresh);
  });
})();
