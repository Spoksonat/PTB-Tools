"use strict";

const ptbEl = document.querySelector(".ptb");
const toolEl = document.querySelectorAll(".tool");
const titleToolEl = document.querySelectorAll(".title-tool");
const descriptionToolEl = document.querySelectorAll(".description-tool");
const noToolEl = document.querySelectorAll(".no-tool");

// Hover events
for (let i = 0; i < toolEl.length; i++) {
  toolEl[i].addEventListener("mouseover", function () {
    titleToolEl[i].classList.add("txtBlueHighlighted");
    descriptionToolEl[i].classList.add("txtHighlighted");
    toolEl[i].style.backgroundColor = "rgba(255,255,255,0.1)";
    toolEl[i].classList.add("zoom");
  });

  toolEl[i].addEventListener("mouseout", function () {
    titleToolEl[i].classList.remove("txtBlueHighlighted");
    descriptionToolEl[i].classList.remove("txtHighlighted");
    toolEl[i].style.backgroundColor = "#000000";
    toolEl[i].classList.remove("zoom");
  });
}

ptbEl.addEventListener("mouseover", function () {
  ptbEl.classList.add("txtBlueHighlighted");
});

ptbEl.addEventListener("mouseout", function () {
  ptbEl.classList.remove("txtBlueHighlighted");
});

// Alert for not implemented tools
for (let i = 0; i < noToolEl.length; i++) {
  noToolEl[i].addEventListener("click", function () {
    alert("This function is not yet implemented");
  });
}
