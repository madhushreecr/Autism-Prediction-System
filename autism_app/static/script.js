// Theme toggle (persisted in localStorage)
(function () {
  const saved = localStorage.getItem("theme") || "light";
  document.documentElement.setAttribute("data-theme", saved);

  document.addEventListener("click", (e) => {
    if (e.target && e.target.matches("[data-theme-toggle]")) {
      const cur = document.documentElement.getAttribute("data-theme");
      const next = cur === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem("theme", next);
      e.target.textContent = next === "dark" ? "☀️ Light" : "🌙 Dark";
    }
  });

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.querySelector("[data-theme-toggle]");
    if (btn) btn.textContent = saved === "dark" ? "☀️ Light" : "🌙 Dark";
  });
})();

// Form validation + loading overlay
document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("predict-form");
  const overlay = document.getElementById("loader");
  if (!form) return;

  form.addEventListener("submit", (e) => {
    const age = +form.age.value;
    const anx = +form.anxiety_level.value;
    const com = +form.communication_score.value;

    if (isNaN(age) || age < 1 || age > 100)
      return alert("Age must be between 1 and 100"), e.preventDefault();
    if (isNaN(anx) || anx < 0 || anx > 10)
      return alert("Anxiety level must be 0–10"), e.preventDefault();
    if (isNaN(com) || com < 0 || com > 10)
      return alert("Communication score must be 0–10"), e.preventDefault();

    if (overlay) overlay.classList.add("active");
  });
});
