let btn = document.querySelector("#btn");
let sidebar = document.querySelector(".sidebar");
let searchBtn = document.querySelector(".bx-search");

btn.onclick = function () {
    sidebar.classList.toggle("active");
}

$(document).ready(function () {
    $('#clean').DataTable();
});

$(document).ready(function () {
    $("#clean").DataTable({
        scrollY: 700,
        scrollX: true,
    });
});