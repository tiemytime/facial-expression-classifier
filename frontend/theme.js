/**
 * Light/dark toggle, shared by every page. The initial theme is applied by
 * an inline script in <head> (before paint); this just wires the button.
 */
(() => {
  const toggle = document.getElementById("themeToggle");
  if (!toggle) return;

  toggle.addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "light" ? "dark" : "light";
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("theme", next);
    } catch (err) {}
  });
})();
