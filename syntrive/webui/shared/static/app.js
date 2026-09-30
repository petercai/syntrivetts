(function () {
  "use strict";

  document.addEventListener("DOMContentLoaded", function () {
    if (!window.htmx) return;
    htmx.config.responseHandling = [
      { code: "204", swap: false },
      { code: "[23]..", swap: true },
      { code: "[45]..", swap: true, error: true },
    ];
  });

  document.addEventListener("htmx:afterSwap", function (event) {
    if (event.detail.target.id !== "dialog-host") return;
    var dialog = event.detail.target.querySelector("dialog");
    if (dialog && !dialog.open) dialog.showModal();
  });

  document.addEventListener("htmx:afterRequest", function (event) {
    if (event.detail.successful) return;
    var dialog = event.detail.elt.closest && event.detail.elt.closest("dialog");
    if (dialog && dialog.open) dialog.close();
  });

  document.addEventListener("click", function (event) {
    var closer = event.target.closest("[data-close-dialog]");
    if (closer) closer.closest("dialog").close();
    var dismiss = event.target.closest("[data-dismiss-flash]");
    if (dismiss) document.getElementById("flash").innerHTML = "";
  });

  document.addEventListener("input", function (event) {
    var input = event.target.closest("input[data-match]");
    if (!input) return;
    var button = document.getElementById(input.dataset.matchTarget);
    if (button) button.disabled = input.value.trim() !== input.dataset.match;
  });

  document.addEventListener("click", function (event) {
    document.querySelectorAll("details.menu[open]").forEach(function (menu) {
      if (!menu.contains(event.target)) menu.removeAttribute("open");
    });
  });
})();
