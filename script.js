window.onload = function () {
    console.log("✅ script.js loaded");
  
    const analyzeBtn = document.getElementById("analyzeBtn");
    const fileInput = document.getElementById("fileUpload");
    const resultText = document.getElementById("resultText");
    const previewImage = document.getElementById("previewImage");
  
    if (!analyzeBtn || !fileInput || !resultText) {
      console.error("❌ Missing required elements");
      return;
    }
  
    // Preview selected image
    fileInput.addEventListener("change", () => {
      if (!fileInput.files.length) return;
  
      const file = fileInput.files[0];
  
      const reader = new FileReader();
      reader.onload = () => {
        previewImage.src = reader.result;
        previewImage.style.display = "block";
      };
      reader.readAsDataURL(file);
  
      resultText.innerText = `Selected: ${file.name}. Ready to analyze.`;
    });
  
    analyzeBtn.onclick = async function () {
      if (!fileInput.files.length) {
        resultText.innerText = "Please select an image first.";
        return;
      }
  
      const formData = new FormData();
      formData.append("image", fileInput.files[0]);
  
      resultText.innerText = "Analyzing...";
  
      // Timeout protection (prevents infinite loading if backend hangs)
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 60000); // 60s
  
      try {
        const res = await fetch("/predict", {
          method: "POST",
          body: formData,
          signal: controller.signal
        });
  
        const raw = await res.text();
        let data = {};
        try { data = JSON.parse(raw); } catch {}
  
        if (!res.ok) {
          resultText.innerText = data.error || raw || "Prediction failed.";
          return;
        }
  
        resultText.innerText = `Prediction: ${data.prediction} (${data.confidence}%)`;
      } catch (err) {
        if (err.name === "AbortError") {
          resultText.innerText = "Request timed out. Backend may be stuck/crashing during prediction.";
        } else {
          resultText.innerText = "Network/Server error: " + (err.message || "Load failed");
        }
      } finally {
        clearTimeout(timeout);
      }
    };
  };
  