const uiButton = document.querySelector("#ui")
const apiButton = document.querySelector("#api")

const apiContainer = document.querySelector("#api-container")

const uiContainer = document.querySelector("#giris-sonuc-textarea")

const sonuc = document.querySelector("#sonuc")
const beklemeEkrani = document.querySelector("#sonuc-bekleme")
const beklemeEkraniYazilar = document.querySelector("#sonuc-bekleme span")

const dogrulaButton = document.querySelector("#doğrula-button")
const girisText = document.querySelector("#giris")

const sonucDogru = document.querySelector("#sonuc-dogru")
const sonucYanlis = document.querySelector("#sonuc-yanlis")

const languages = document.querySelector("#languages")
const pythonO = document.querySelector("#python");
const javascriptO = document.querySelector("#javascript");

const jsCode = document.querySelector("#api-code-js")
const pyCode = document.querySelector("#api-code-py")
const benzerHaberler = document.querySelector("#checkbox");
const benzerHaberlerSonuc = document.querySelector("#benzerhaberlersonuc");

const sonucLogo = document.querySelector("#sonuc-logo");

const bitArtik = document.querySelector("#bit-artik");
addEventListener("DOMContentLoaded",()=>{
    uiButton.style.color = "var(--accent-color2)"
    uiButton.style.backgroundColor = "var(--pre-accent-color)"
    pythonO.style.color = "var(--accent-color2)";
    pythonO.style.backgroundColor = "var(--pre-accent-color)"
    jsCode.style.display = "none";
})
function uiclick(){
    uiButton.style.color = "var(--accent-color2)"
    uiButton.style.backgroundColor = "var(--pre-accent-color)"
    uiContainer.style.display = "flex"

    apiButton.style.color = null
    apiButton.style.backgroundColor = null
    apiContainer.style.display = "none"
    languages.style.display = "none"
}
function javascript(){
    javascriptO.style.color = "var(--accent-color2)";
    javascriptO.style.backgroundColor = "var(--pre-accent-color)"
    jsCode.style.display = "block"
    pythonO.style.color = null;
    pythonO.style.backgroundColor = null
    pyCode.style.display = "none";
}
function python(){
    pythonO.style.color = "var(--accent-color2)";
    pythonO.style.backgroundColor = "var(--pre-accent-color)"
    pyCode.style.display = "block";
    javascriptO.style.color = null;
    javascriptO.style.backgroundColor = null
    jsCode.style.display = "none";
}
function apiclick(){
    uiButton.style.color = null
    uiButton.style.backgroundColor = null
    uiContainer.style.display = "none";

    apiButton.style.color = "var(--accent-color2)"
    apiButton.style.backgroundColor = "var(--pre-accent-color)"
    apiContainer.style.display = "flex"
    languages.style.display = "flex"
}
async function getPrediction(text){
    let dataToSend = {
        text
    };
    dataToSend = JSON.stringify(dataToSend);
    console.log(dataToSend)
    let res = await fetch("http://dogrulamac.me/predict", {
        method: "POST", 
        headers: { "Content-Type": "application/json" },
        body: dataToSend,
        mode: "cors"
    });
    let data = await res.json();
    return data;
}
async function getSimilarNews(text) {
    let dataToSend = {
        text
    };
    dataToSend = JSON.stringify(dataToSend);
    let res = await fetch("http://dogrulamac.me/similarNews", {
        method: "POST", 
        headers: { "Content-Type": "application/json" },
        body: dataToSend
    });
    let data = await res.json();
    return data;
}
let i = 0;
const yazilar = ["Haberiniz LLM modeline gönderiliyor...","Haber işleme alınıyor...","Doğrulama modelden alınıyor...", "Neredeyse tamamlandı..."]
async function dogrula(){
    dogrulaButton.setAttribute("disabled","")
    girisText.setAttribute("disabled", "")

    sonuc.style.display = "none";
    sonuc.innerHTML = "";
    sonucLogo.style.display = "block"
    beklemeEkrani.style.display = "flex"
    beklemeEkraniYazilar.textContent = "";
    beklemeEkraniYazilar.textContent = yazilar[0];
    i++;
    const interval = setInterval(()=>{
        beklemeEkraniYazilar.textContent = yazilar[i]
        if(i<3)  i++;
      
    },2000)
    benzerHaberlerSonuc.innerHTML = "";
    sonucDogru.style.display = "none"
    sonucYanlis.style.display = "none"
    bitArtik.style.display = "none"
    const benzerHaberlerBulunsunMu = benzerHaberler.checked;
    console.log(benzerHaberlerBulunsunMu)
    let similarNews;
    if(benzerHaberlerBulunsunMu){
        similarNews = await getSimilarNews(girisText.value)
        console.log(similarNews)
        let results = similarNews.result
        if(results == "No trusted articles found"){
            benzerHaberlerSonuc.innerHTML += "Benzer haber bulunamadı.";
        }
        else{
            for(let i=0;i<similarNews.result.length;i++){
                console.log(results[i][0])
                benzerHaberlerSonuc.innerHTML += "<a>  Haber linki: " + results[i][0] + "</a><br>" +  "Benzerlik oranı: " + results[i][1] + "<br><br>";
    
            }
        }
        
    }
    

    let prediction = await getPrediction(girisText.value);
    console.log(prediction);
    bitArtik.style.display = "flex"
    clearInterval(interval);
    beklemeEkrani.style.display = "none"
    sonuc.style.display = "block"
    sonuc.textContent += prediction
    girisText.removeAttribute("disabled")
    dogrulaButton.removeAttribute("disabled")
    if(prediction[0]=="Y"){
        sonucYanlis.style.display = "flex"
    }
    else{
        sonucDogru.style.display = "flex"
    }
    i= 0;
}
function temizle(){
    
    benzerHaberlerSonuc.innerHTML = "";
    sonucDogru.style.display = "none"
    sonucYanlis.style.display = "none"
    sonuc.textContent = "";
    girisText.value = "";
    benzerHaberler.checked = false
}