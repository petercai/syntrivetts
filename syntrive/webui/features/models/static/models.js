(function () {
  "use strict";
  document.addEventListener("click", function (event) {
    var hide = event.target.closest("[data-files-hide]");
    if (hide) hide.closest(".files-slot").innerHTML = "";
  });
})();
