let startTime;

const textarea = document.getElementById("text");

textarea.addEventListener("focus", () => {
    startTime = new Date().getTime();
});

function predict() {
    let endTime = new Date().getTime();
    let timeTaken = (endTime - startTime) / 1000;

    let text = textarea.value;

    fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text: text,
            time: timeTaken
        })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById("result").innerHTML =
            `<b>Prediction:</b> ${data.label}<br>
             <b>Confidence:</b> ${data.confidence}<br>
             <b>Decision:</b> ${data.decision}<br>
             <b>Behavior:</b> ${data.behavior}`;
    });
}