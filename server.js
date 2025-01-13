const  express = require("express");
const path = require("path");
const PORT= 8080;
const app = express();
app.use(express.json())
app.use(express.static(__dirname + '/public'));
app.get("/",(req, res)=>{
    res.sendFile(path.join(__dirname, "./index.html"))
})
async function predict(text){
    send = JSON.stringify({"text":text})
    response = await fetch("http://127.0.0.1:8000/predict",{
        method : "POST", 
        headers: { "Content-Type": "application/json" },
        body: send})
    data = await response.json();
    return data;
}
app.post("/predict", async (req, res)=>{
    console.log(req.body.text);
    predictedData = await predict(req.body.text);
    console.log(predictedData)
    res.json(predictedData);
    return;
})
app.listen(PORT, ()=>{
    console.log("Js Server started on port: " + PORT )
})