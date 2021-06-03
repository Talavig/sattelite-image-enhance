var divisor = document.getElementsByClassName("divisor")[0];
var handle = document.getElementsByClassName("handle")[0];
var slider = document.getElementsByClassName("slider")[0];
var divisor_canny = document.getElementsByClassName("divisor")[1];
var handle_canny = document.getElementsByClassName("handle")[1];
var slider_canny = document.getElementsByClassName("slider")[1];


function moveDivisor() {
  handle.style.left = slider.value+"%";
	divisor.style.width = slider.value+"%";
  handle_canny.style.left = slider_canny.value+"%";
	divisor_canny.style.width = slider_canny.value+"%";
}

window.onload = function() {
	moveDivisor();
};

