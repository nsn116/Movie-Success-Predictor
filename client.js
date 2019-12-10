function UserAction() {
    var xhttp_post = new XMLHttpRequest();
    xhttp_post.open("POST", "http://localhost:5000/post_movie_metrics", false);
    xhttp_post.setRequestHeader("Content-type", "application/json");
    xhttp_post.setRequestHeader("Access-Control-Allow-Origin", "*");
    var form_data = {
      "genres": document.getElementById("genres").value.split(","),
      "budget": document.getElementById("budget").value,
      "runtime": document.getElementById("runtime").value,
      "directors": document.getElementById("directors").value.split(","),
      "writers": document.getElementById("writers").value.split(","),
      "cast": document.getElementById("cast").value.split(","),
      "prod_companies": document.getElementById("prod_companies").value.split(",")
    };
    var body = JSON.stringify(form_data);
    xhttp_post.send(body);

    var xhttp_get = new XMLHttpRequest();
    xhttp_get.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            var response = this.responseText;
            document.getElementById("result").hidden = false;
            var split = response.split(";;");
            var revenue = split[0].trim();
            console.log(revenue);
            var rating = split[1].trim();
            console.log(rating);
            document.getElementById("revenue").value = revenue;
            document.getElementById("rating").value = rating;
        }
    };
    xhttp_get.open("GET", "http://localhost:5000/get_success_metrics", false);
    xhttp_get.setRequestHeader("Content-type", "application/json");
    xhttp_get.setRequestHeader("Access-Control-Allow-Origin", "*");
    xhttp_get.send(null);
}
