(function () {
  "use strict";

  var player = null;
  function stopPlayer() {
    if (!player) return;
    player.audio.pause();
    player.button.classList.remove("is-playing");
    player = null;
  }

  function refreshImport(form) {
    var n = form.querySelectorAll('input[name="path"]:checked').length;
    var button = form.querySelector("[data-import-submit]");
    if (button) {
      button.textContent = button.dataset.labelN.replace("{n}", n);
      button.disabled = n === 0;
    }
    var all = form.querySelector("[data-check-all]");
    if (all) all.checked = n === form.querySelectorAll('input[name="path"]').length;
  }

  document.addEventListener("click", function (event) {
    var play = event.target.closest("[data-play]");
    if (!play) return;
    var same = player && player.button === play;
    stopPlayer();
    if (same) return;
    var audio = document.getElementById("voice-player");
    audio.src = play.dataset.play;
    audio.onended = stopPlayer;
    audio.play();
    play.classList.add("is-playing");
    player = { audio: audio, button: play };
  });

  document.addEventListener("change", function (event) {
    var form = event.target.closest("form[data-import-form]");
    if (!form) return;
    if (event.target.matches("[data-check-all]")) {
      form.querySelectorAll('input[name="path"]').forEach(function (box) { box.checked = event.target.checked; });
    }
    refreshImport(form);
  });
})();
