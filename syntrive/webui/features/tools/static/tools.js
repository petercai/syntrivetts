(function () {
  "use strict";

  function fill(template, values) {
    return template.replace(/\{(\w+)\}/g, function (_, key) { return values[key]; });
  }

  function refreshExport(form) {
    form.querySelectorAll(".pick-book").forEach(function (book) {
      var jobs = book.querySelectorAll('input[name="job"]');
      var on = book.querySelectorAll('input[name="job"]:checked').length;
      var head = book.querySelector("[data-book]");
      head.checked = on === jobs.length && on > 0;
      head.indeterminate = on > 0 && on < jobs.length;
    });
    var n = form.querySelectorAll('input[name="job"]:checked').length;
    var button = form.querySelector("[data-export-submit]");
    button.textContent = fill(button.dataset.labelN, { n: n });
    button.disabled = n === 0;
  }

  function refreshImport(form) {
    var counts = { add: 0, overwrite: 0, skip: 0 };
    form.querySelectorAll("[data-job-row]").forEach(function (row) {
      var chosen = row.querySelector('input[name="job"]').checked;
      row.classList.toggle("is-off", !chosen);
      row.querySelectorAll('input[type="radio"]').forEach(function (r) { r.disabled = !chosen; });
      if (!chosen) return;
      if (row.dataset.conflict !== "true") counts.add += 1;
      else if (row.querySelector('input[value="overwrite"]:checked')) counts.overwrite += 1;
      else counts.skip += 1;
    });
    var button = form.querySelector("[data-import-submit]");
    button.textContent = fill(button.dataset.label, counts);
    button.disabled = counts.add + counts.overwrite === 0;
  }

  document.addEventListener("change", function (event) {
    var exportForm = event.target.closest("form[data-export-form]");
    if (exportForm) {
      if (event.target.matches("[data-book]")) {
        event.target.closest(".pick-book").querySelectorAll('input[name="job"]').forEach(function (box) { box.checked = event.target.checked; });
      }
      refreshExport(exportForm);
    }
    var importForm = event.target.closest("form[data-import-form]");
    if (importForm) refreshImport(importForm);
  });

  document.addEventListener("htmx:afterSwap", function (event) {
    if (event.detail.target.id !== "dialog-host") return;
    event.detail.target.querySelectorAll("form[data-export-form]").forEach(refreshExport);
    event.detail.target.querySelectorAll("form[data-import-form]").forEach(refreshImport);
  });
})();
