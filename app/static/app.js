document.addEventListener("DOMContentLoaded", () => {
  const textarea = document.getElementById("text-input");
  const charCounter = document.getElementById("char-count");
  const copyBtn = document.getElementById("copy-btn");
  const form = document.getElementById("correct-form");

  if (textarea && charCounter) {
    const updateCount = () => { charCounter.textContent = `${textarea.value.length} / 2000`; };
    textarea.addEventListener("input", updateCount);
    updateCount();
  }

  if (textarea && form) {
    textarea.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        form.submit();
      }
    });
  }

  if (copyBtn) {
    copyBtn.addEventListener("click", () => {
      const outputElem = document.getElementById("corrected-text");
      if (outputElem) {
        navigator.clipboard.writeText(outputElem.innerText.trim()).then(() => {
          const oldLabel = copyBtn.innerText;
          copyBtn.innerText = "Copied!";
          setTimeout(() => { copyBtn.innerText = oldLabel; }, 1800);
        });
      }
    });
  }
});
