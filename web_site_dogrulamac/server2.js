import { Client } from "@gradio/client";
const grad = await Client.connect("https://9ab2688b03d645943e.gradio.live/");

import express from "express";
import path from "path"
import { fileURLToPath } from 'url';
const __filename = fileURLToPath(import.meta.url); // get the resolved path to the file
const __dirname = path.dirname(__filename); // get the name of the directory
const PORT= 8080;
const app = express();
app.use(express.json())
app.use(express.static(__dirname + '/public'));
app.use(express.static(__dirname + '/public/favicon'));

app.get("/",(req, res)=>{
    res.sendFile(path.join(__dirname, "./index.html"))
})
app.get("/dene", (req, res)=>{
    res.sendFile(path.join(__dirname,"./public/Dene.html"))
})
const delay = (delayInms) => {
    return new Promise(resolve => setTimeout(resolve, delayInms));
  };
async function predict(text){
    const result = await grad.predict("/predict", [text]);
    
    console.log(result);
    return result.data[0];
    // let a = await delay(1000);
    // return "merhaba"
}
async function getSimilarNews(text){
    let send = JSON.stringify({"input_news":text})
    const result = await fetch("http://localhost:5000/similar",{
        headers: { "Content-Type": "application/json" },
        method: "POST",
        body: send

    })
    return result
}
app.post("/similarNews", async (req,res)=>{
    const haber = req.body.text;
    console.log(haber)
    let result = await getSimilarNews(haber);
    let data = await result.json();
    res.json(data)
    return 
})
app.post("/predict", async (req, res)=>{
    console.log(req.body.text);
    const predictedData = await predict(req.body.text);
    console.log(predictedData)
    res.json(predictedData);
    return;
})
app.listen(PORT, ()=>{
    console.log("Js Server started on port: " + PORT)
})