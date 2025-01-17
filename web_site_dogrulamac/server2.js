import { Client } from "@gradio/client";
const grad = await Client.connect("http://127.0.0.1:7860")

import express from "express";
import path from "path"
import { fileURLToPath } from 'url';
import cors from "cors"

const __filename = fileURLToPath(import.meta.url); // get the resolved path to the file
const __dirname = path.dirname(__filename); // get the name of the directory
const PORT= 8000;
const app = express();
// app.use(cors({
//     origin: "*"
// }));
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
    console.log(result);
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