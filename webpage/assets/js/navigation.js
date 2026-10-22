(() => {
  const header = document.querySelector('.site-header');
  const nav = document.getElementById('primary-navigation');
  const toggle = header.querySelector('.nav-toggle');
  const links = [...nav.querySelectorAll('a[href^="#"]')];
  const sections = links.map(link => document.querySelector(link.hash));
  const compact = window.matchMedia('(max-width: 1100px)');

  function setOpen(open) {
    nav.classList.toggle('is-open', open);
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close navigation' : 'Open navigation');
    toggle.title = open ? 'Close navigation' : 'Open navigation';
    toggle.firstElementChild.textContent = open ? '\u00d7' : '\u2630';
  }

  toggle.addEventListener('click', () => setOpen(toggle.getAttribute('aria-expanded') !== 'true'));
  nav.addEventListener('click', event => {
    if (event.target.closest('a')) {
      setOpen(false);
      if (compact.matches) toggle.focus({ preventScroll: true });
    }
  });
  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && toggle.getAttribute('aria-expanded') === 'true') {
      setOpen(false);
      toggle.focus();
    }
  });
  document.addEventListener('click', event => {
    if (!header.contains(event.target)) setOpen(false);
  });
  header.addEventListener('focusout', event => {
    if (!header.contains(event.relatedTarget)) setOpen(false);
  });
  compact.addEventListener('change', () => setOpen(false));

  function updateCurrent() {
    // Use one reading line so section boundaries do not produce competing highlights.
    const readingLine = Math.max(header.offsetHeight + 24, window.innerHeight * 0.25);
    let current = -1;
    sections.forEach((section, index) => {
      if (section.getBoundingClientRect().top <= readingLine) current = index;
    });
    links.forEach((link, index) => {
      if (index === current) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  }
  let scheduled = false;
  function scheduleUpdate() {
    if (scheduled) return;
    scheduled = true;
    requestAnimationFrame(() => { updateCurrent(); scheduled = false; });
  }
  window.addEventListener('scroll', scheduleUpdate, { passive: true });
  window.addEventListener('resize', scheduleUpdate);
  window.addEventListener('load', scheduleUpdate);
  updateCurrent();
})();
