// Selecting a language switches every synchronised tab set on the page, which changes the height of the content above
// the clicked tab. Keep the clicked label at the same viewport position by scrolling by the distance it moved; browsers
// without native scroll anchoring (Safari) would otherwise move the page under the pointer.
//
// The label's position is recorded when the click starts, before sphinx-design switches the synchronised tab sets, and
// compared once the clicked tab's radio input reports the change, before the browser paints. The click the browser then
// forwards to that input must not clear the record. A click on the label of the checked tab fires no change and is not
// recorded.
//
// The bar under the active label of a language tab set is the set's ::before pseudo-element (docs/_static/custom.css),
// placed under the checked label through CSS variables. Switching a language checks the synchronised inputs of every
// set within the same event, so all sets are placed on the following frame, from where their bars slide to the new
// label.
const initLanguageTabs = () => {
  const sets = document.querySelectorAll(".sd-tab-set.code-tabs");
  if (sets.length === 0) return;

  let pending = null;

  document.addEventListener(
    "click",
    (event) => {
      const label = event.target.closest(".sd-tab-set > label[data-sync-group]");
      if (label && !label.previousElementSibling.checked) {
        pending = { label, top: label.getBoundingClientRect().top };
      }
    },
    true
  );

  document.addEventListener(
    "change",
    (event) => {
      if (!pending || event.target !== pending.label.previousElementSibling) return;

      const shift = pending.label.getBoundingClientRect().top - pending.top;
      pending = null;
      if (shift === 0) return;

      // jump rather than animate, whatever scroll-behavior the page sets
      const root = document.documentElement;
      const behavior = root.style.scrollBehavior;
      root.style.scrollBehavior = "auto";
      window.scrollBy(0, shift);
      root.style.scrollBehavior = behavior;
    },
    true
  );

  const placeIndicators = () => {
    const placements = [];

    for (const set of sets) {
      const label = set.querySelector(":scope > input:checked + label");
      if (!label) continue;

      const box = set.getBoundingClientRect();
      const rect = label.getBoundingClientRect();
      placements.push({ set, x: rect.left - box.left, width: rect.width, bottom: rect.bottom - box.top });
    }

    for (const { set, x, width, bottom } of placements) {
      set.style.setProperty("--code-tabs-indicator-x", `${x}px`);
      set.style.setProperty("--code-tabs-indicator-width", `${width}px`);
      set.style.setProperty("--code-tabs-indicator-bottom", `${bottom}px`);
    }
  };

  placeIndicators();

  // the initial placement is not animated
  requestAnimationFrame(() => {
    for (const set of sets) set.classList.add("code-tabs-animated");
  });

  let resizeFrame = 0;

  window.addEventListener("load", placeIndicators);
  window.addEventListener("resize", () => {
    if (resizeFrame) return;
    resizeFrame = requestAnimationFrame(() => {
      resizeFrame = 0;
      placeIndicators();
    });
  });
  document.addEventListener("change", () => requestAnimationFrame(placeIndicators));
};

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initLanguageTabs);
} else {
  initLanguageTabs();
}
