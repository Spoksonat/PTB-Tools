"use strict";

const subtitle = document.querySelector(".subtitle");
const toolEl = document.querySelectorAll(".tool");
const titleToolEl = document.querySelectorAll(".title-tool");
const descriptionToolEl = document.querySelectorAll(".description-tool");
const noToolEl = document.querySelectorAll(".no-tool");
const subsection1 = document.querySelector(".subsection-1");
const subsection2 = document.querySelector(".subsection-2");
const container1 = document.querySelector(".container-1");
const container2 = document.querySelector(".container-2");
const modalWindow1 = document.querySelector(".modal1");
const modalWindow2 = document.querySelector(".modal2");
const overlay = document.querySelector(".overlay");
const btnCloseModal1 = document.querySelector(".close-modal1");
const btnCloseModal2 = document.querySelector(".close-modal2");
const btnInput = document.querySelector(".input-button");
const inputText = document.querySelector(".input-text");
const btnSend = document.querySelector(".send-file");
const message1 = document.querySelector(".modal-message1");
const message2 = document.querySelector(".modal-message2");
const btnSendFilename = document.querySelector(".send-filename");
const btnShowB1p = document.querySelector(".b1p-button");
const btnShowB1m = document.querySelector(".b1m-button");
const btnShowNcm = document.querySelector(".ncm-button");

const closeModal1 = function () {
  modalWindow1.classList.add("hidden");
  overlay.classList.add("hidden");
  modalWindow1.classList.remove("center");
};

const closeModal2 = function () {
  modalWindow2.classList.add("hidden");
  overlay.classList.add("hidden");
  modalWindow2.classList.remove("center");
};

const openModal1 = function () {
  modalWindow1.classList.remove("hidden");
  overlay.classList.remove("hidden");
  modalWindow1.classList.add("center");
};

const openModal2 = function () {
  if (boolLoad === "false") {
    alert("Please load your data first!");
  } else {
    modalWindow2.classList.remove("hidden");
    overlay.classList.remove("hidden");
    modalWindow2.classList.add("center");
  }
};

const sendData = function () {
  let value = document.querySelector(".input-button").value;
  console.log(value);
  $.ajax({
    url: "/process",
    type: "POST",
    data: { data: value },
    success: function (response) {},
    error: function (error) {
      console.log(error);
    },
  });
};

const sendName = function () {
  let value = document.querySelector(".input-text").value;
  console.log(value);
  $.ajax({
    url: "/process2",
    type: "POST",
    data: { name_b1p: value },
    success: function (response) {},
    error: function (error) {
      console.log(error);
    },
  });
};

// Hover events
for (let i = 0; i < toolEl.length; i++) {
  toolEl[i].addEventListener("mouseover", function () {
    titleToolEl[i].classList.add("txtGreenHighlighted");
    descriptionToolEl[i].classList.add("txtHighlighted");
    toolEl[i].style.backgroundColor = "rgba(255,255,255,0.1)";
    toolEl[i].classList.add("zoom");
  });

  toolEl[i].addEventListener("mouseout", function () {
    titleToolEl[i].classList.remove("txtGreenHighlighted");
    descriptionToolEl[i].classList.remove("txtHighlighted");
    toolEl[i].style.backgroundColor = "#000000";
    toolEl[i].classList.remove("zoom");
  });
}

subsection1.addEventListener("mouseover", function () {
  subsection1.classList.add("txtGreenHighlighted");
});

subsection2.addEventListener("mouseover", function () {
  subsection2.classList.add("txtGreenHighlighted");
});

subsection1.addEventListener("mouseout", function () {
  subsection1.classList.remove("txtGreenHighlighted");
});

subsection2.addEventListener("mouseout", function () {
  subsection2.classList.remove("txtGreenHighlighted");
});

//Click events
subsection1.addEventListener("click", function () {
  container1.classList.toggle("hidden");
});

subsection2.addEventListener("click", function () {
  container2.classList.toggle("hidden");
});

btnCloseModal1.addEventListener("click", closeModal1);
btnCloseModal2.addEventListener("click", closeModal2);
overlay.addEventListener("click", closeModal1);
overlay.addEventListener("click", closeModal2);
btnInput.addEventListener("change", sendData);
btnSend.addEventListener("click", function () {
  message1.textContent =
    "Note: Your file is being uploaded. This window will close when the file is finished uploading";
});
inputText.addEventListener("change", sendName);
btnSendFilename.addEventListener("click", function () {
  message2.textContent = "Note: Your file is being saved";
});

btnShowB1p.addEventListener("click", function () {
  if (boolLoad === "false") {
    btnShowB1p.href = "/B1_Mapping_Toolbox";
    alert("Please load your data first!");
  } else {
    btnShowB1p.href = "/B1_Mapping_Toolbox/b1p/";
  }
});

btnShowB1m.addEventListener("click", function () {
  if (boolLoad === "false") {
    btnShowB1m.href = "/B1_Mapping_Toolbox";
    alert("Please load your data first!");
  } else {
    btnShowB1m.href = "/B1_Mapping_Toolbox/b1m/";
  }
});

btnShowNcm.addEventListener("click", function () {
  if (boolLoad === "false") {
    btnShowNcm.href = "/B1_Mapping_Toolbox";
    alert("Please load your data first!");
  } else {
    btnShowNcm.href = "/B1_Mapping_Toolbox/noise_correlation_maps/";
  }
});

// Alert for not implemented tools
for (let i = 0; i < noToolEl.length; i++) {
  noToolEl[i].addEventListener("click", function () {
    alert("This function is not yet implemented");
  });
}
