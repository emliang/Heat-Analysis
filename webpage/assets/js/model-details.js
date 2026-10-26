(() => {
  const container = document.getElementById('model-details');
  if (!container) return;
  const triggers = [...document.querySelectorAll('.model-open')];
  const tabs = [...container.querySelectorAll('[role="tab"]')];
  const panels = [...container.querySelectorAll('[role="tabpanel"]')];
  let opener = null;
  function select(name) {
    container.hidden = false;
    container.dataset.active = name;
    tabs.forEach(tab => {
      const active = tab.dataset.model === name;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
    });
    panels.forEach(panel => { panel.hidden = panel.id !== `model-${name}`; });
    triggers.forEach(button => button.setAttribute('aria-expanded', String(button.dataset.model === name)));
  }
  function close() {
    container.hidden = true;
    triggers.forEach(button => button.setAttribute('aria-expanded', 'false'));
    if (opener) opener.focus();
  }
  triggers.forEach(button => button.addEventListener('click', () => {
    opener = button;
    select(button.dataset.model);
    tabs.find(tab => tab.dataset.model === button.dataset.model).focus({preventScroll: true});
    container.scrollIntoView({block: 'start', behavior: 'instant'});
  }));
  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => {
      select(tab.dataset.model);
      container.scrollIntoView({block: 'start', behavior: 'instant'});
    });
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      select(tabs[next].dataset.model);
      tabs[next].focus({preventScroll: true});
      container.scrollIntoView({block: 'start', behavior: 'instant'});
    });
  });
  container.querySelector('.model-close').addEventListener('click', close);
  container.addEventListener('keydown', event => {
    if (event.key === 'Escape') { event.preventDefault(); close(); }
  });
})();
