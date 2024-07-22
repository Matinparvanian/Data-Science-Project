Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Drop files here or click to upload",
        autoProcessQueue: false,
        init: function () {
            this.on("addedfile", function (file) {
                if (dz.files[1] != null) {
                    dz.removeFile(dz.files[0]);
                }
                let reader = new FileReader();
                reader.onload = function (event) {
                    file.dataURL = event.target.result;
                };
                reader.readAsDataURL(file);
            });

            this.on("complete", function (file) {
                let imageData = file.dataURL;

                var url = "http://127.0.0.1:5000/classify_image";

                $.post(url, {
                    image_data: imageData
                }, function (data, status) {
                    console.log(data);
                    if (!data || data.length == 0) {
                        $("#resultHolder").hide();
                        $("#divClassTable").hide();
                        $("#error").show();
                        return;
                    }
                    let players = ["elon musk", "jeff bezos", "mark zuckerberg", "maryam mirzakhani", "tim cook", "warren buffett"];

                    let match = null;
                    let bestScore = -1;
                    for (let i = 0; i < data.length; ++i) {
                        let maxScoreForThisClass = Math.max(...data[i].class_probability);
                        if (maxScoreForThisClass > bestScore) {
                            match = data[i];
                            bestScore = maxScoreForThisClass;
                        }
                    }
                    if (match) {
                        $("#error").hide();
                        $("#resultHolder").show();
                        $("#divClassTable").show();
                        $("#resultHolder").html($(`[data-player="${match.class.replace(" ", "_")}"`).html());
                        let classDictionary = match.class_dictionary;
                        for (let personName in classDictionary) {
                            let index = classDictionary[personName];
                            let probabilityScore = match.class_probability[index];
                            let elementName = "#score_" + personName.replace(" ", "_");
                            $(elementName).html(probabilityScore.toFixed(2) + "%"); // Display as a percentage with 2 decimal places
                        }
                    }
                });
            });
        }
    });

    $("#submitBtn").on('click', function (e) {
        dz.processQueue();
    });
}

$(document).ready(function () {
    console.log("ready!");
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();

    init();
});
