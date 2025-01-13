const input = document.querySelector("#haber_input"); 
const button = document.querySelector("#haber_send")  
const sonuc = document.querySelector("#sonuc-text")
const loader = document.querySelector("#loader")
const haberInput = document.querySelector("#haber_input")
const sonucLogo = document.querySelector("#sonuc-logo")
const sonucDogru = document.querySelector("#sonuc-sonuc-dogru")
const sonucYanlis = document.querySelector("#sonuc-sonuc-yanlis")
const benzerHaberler = document.querySelector("#benzer-haberler")
const benzerHaberlerSonuc = document.querySelector("#benzer-haberler-sonuc")
async function getPrediction(text){
    let dataToSend = {
        text
    };
    dataToSend = JSON.stringify(dataToSend);
    console.log(dataToSend)
    let res = await fetch("http://localhost:8080/predict", {
        method: "POST", 
        headers: { "Content-Type": "application/json" },
        body: dataToSend
    });
    let data = await res.json();
    return data;
}
async function getSimilarNews(text) {
    let dataToSend = {
        text
    };
    dataToSend = JSON.stringify(dataToSend);
    let res = await fetch("http://localhost:8080/similarNews", {
        method: "POST", 
        headers: { "Content-Type": "application/json" },
        body: dataToSend
    });
    let data = await res.json();
    return data;
}
let i = 0;
const yazilar = ["Haberiniz LLM modeline gönderiliyor...","Haber işleme alınıyor...","Doğrulama modelden alınıyor...", "Neredeyse tamamlandı..."]
button.addEventListener("click",async ()=>{
    button.setAttribute("disabled","")
    haberInput.setAttribute("disabled", "")
    sonuc.innerHTML = "";
    sonucLogo.style.display = "block"
    sonuc.innerHTML = yazilar[0];
    i++;
    const interval = setInterval(()=>{
        sonuc.innerHTML = yazilar[i]
        if(i<3)  i++;
      
    },2000)
    
    const benzerHaberlerBulunsunMu = benzerHaberler.checked;
    console.log(benzerHaberlerBulunsunMu)
    let similarNews;
    if(benzerHaberlerBulunsunMu){
        similarNews = await getSimilarNews(input.value)
        console.log(similarNews)
        let results = similarNews.result
        for(let i=0;i<similarNews.result.length;i++){
            console.log(results[i][0])
            benzerHaberlerSonuc.innerHTML += "<a>  Haber linki: " + results[i][0] + "</a><br>" +  "Benzerlik oranı: " + results[i][1] + "<br><br>";

        }
    }

    sonucDogru.style.display = "none"
    sonucYanlis.style.display = "none"

    
    let prediction = await getPrediction(input.value);
    clearInterval(interval);
    sonucLogo.style.display = "none"
    console.log(prediction);
    if(prediction[6]=="Y"){
        sonucYanlis.style.display = "flex"
    }
    else{
        sonucDogru.style.display = "flex"
    }
    sonuc.textContent = prediction
    haberInput.removeAttribute("disabled")
    button.removeAttribute("disabled")

    i= 0;

})
function hemenDene(){
    window.location = '/Dene'
}