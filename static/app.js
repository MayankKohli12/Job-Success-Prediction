/* SkillPulse: progressive enhancement only. Every page works without this file. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const X = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M6 6l12 12M18 6 6 18"/></svg>';

  // ---- navigation drawer (small screens)
  const rail = $('#rail'), scrim = $('#scrim'), navBtn = $('#nav-open');
  const setNav = (on) => { rail.classList.toggle('open', on); scrim.classList.toggle('on', on); navBtn.setAttribute('aria-expanded', on); };
  navBtn?.addEventListener('click', () => setNav(!rail.classList.contains('open')));

  // ---- filter drawer
  const filters = $('#filters');
  const setFilters = (on) => { filters?.classList.toggle('open', on); scrim.classList.toggle('on', on); };
  document.addEventListener('click', (e) => {
    if (e.target.closest('[data-open-filters]')) setFilters(true);
    if (e.target.closest('[data-close-filters]')) setFilters(false);
  });
  scrim.addEventListener('click', () => { setNav(false); setFilters(false); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { setNav(false); setFilters(false); } });

  // ---- token combobox (skills, companies, cities)
  function initCombo(box) {
    const name = box.dataset.name, kind = box.dataset.kind, input = $('input[type=text]', box), list = $('.sugg', box), wrap = $('.combo-box', box);
    const neg = !!$('.token.neg', box) || name === 'xskill';
    let items = [], idx = -1, timer, seq = 0;
    const have = () => $$('input[type=hidden]', box).map((i) => i.value);
    const close = () => { list.hidden = true; input.setAttribute('aria-expanded', 'false'); idx = -1; };
    const paint = () => {
      list.innerHTML = '';
      items.forEach((it, i) => {
        const li = document.createElement('li');
        li.setAttribute('role', 'option'); li.setAttribute('aria-selected', i === idx);
        li.innerHTML = '<b style="font-weight:500"></b><span class="num"></span>';
        li.firstChild.textContent = it.label; li.lastChild.textContent = it.n;
        li.addEventListener('mousedown', (e) => { e.preventDefault(); add(it); });
        list.append(li);
      });
      list.hidden = !items.length; input.setAttribute('aria-expanded', !!items.length);
    };
    const add = (it) => {
      if (have().includes(it.value)) return;
      const t = document.createElement('span');
      t.className = 'token' + (neg ? ' neg' : ''); t.dataset.value = it.value;
      t.append(document.createTextNode(it.label));
      const h = Object.assign(document.createElement('input'), { type: 'hidden', name, value: it.value });
      const b = Object.assign(document.createElement('button'), { type: 'button' });
      b.setAttribute('aria-label', 'Remove ' + it.label); b.innerHTML = X;
      t.append(h, b); wrap.insertBefore(t, input);
      input.value = ''; close(); input.focus();
      box.closest('form')?.dispatchEvent(new Event('change', { bubbles: true }));
    };
    const fetchSugg = async () => {
      const my = ++seq;
      const r = await fetch(`/api/suggest/${kind}?q=${encodeURIComponent(input.value)}`);
      if (my !== seq) return;
      const taken = have();
      items = (await r.json()).filter((x) => !taken.includes(x.value)); idx = -1; paint();
    };
    input.addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(fetchSugg, 120); });
    input.addEventListener('focus', fetchSugg);
    input.addEventListener('blur', () => setTimeout(close, 100));
    input.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault(); if (!items.length) return;
        idx = (idx + (e.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length; paint();
      } else if (e.key === 'Enter') {
        if (list.hidden && !input.value) return;
        e.preventDefault(); const it = items[idx >= 0 ? idx : 0]; if (it) add(it);
      } else if (e.key === 'Backspace' && !input.value) {
        $$('.token', box).pop()?.querySelector('button').click();
      } else if (e.key === 'Escape') close();
    });
    wrap.addEventListener('click', (e) => {
      const b = e.target.closest('.token button');
      if (b) { b.closest('.token').remove(); box.closest('form')?.dispatchEvent(new Event('change', { bubbles: true })); }
      else input.focus();
    });
  }
  $$('[data-combo]').forEach(initCombo);
  // ---- plain-text suggestions (job title, city): a short popular list when empty, prefix then substring matches while typing
  $$('[data-suggest]').forEach((wrap) => {
    let opts = []; try { opts = JSON.parse(wrap.dataset.suggest); } catch (e) {}
    const input = $('input', wrap), ul = document.createElement('ul');
    ul.className = 'sugg'; ul.setAttribute('role', 'listbox'); ul.hidden = true; wrap.appendChild(ul);
    let idx = -1, shown = [];
    const hide = () => { ul.hidden = true; idx = -1; };
    const paint = () => $$('li', ul).forEach((li, i) => li.setAttribute('aria-selected', i === idx));
    const show = () => {
      const q = input.value.trim().toLowerCase();
      shown = (q ? [...opts.filter((o) => o.toLowerCase().startsWith(q)), ...opts.filter((o) => !o.toLowerCase().startsWith(q) && o.toLowerCase().includes(q))] : opts).slice(0, 6);
      if (!shown.length || (shown.length === 1 && shown[0].toLowerCase() === q)) return hide();
      ul.innerHTML = ''; idx = -1;
      shown.forEach((o) => { const li = document.createElement('li'); li.setAttribute('role', 'option'); li.textContent = o;
        li.addEventListener('mousedown', (e) => { e.preventDefault(); pick(o); }); ul.appendChild(li); });
      ul.hidden = false;
    };
    const pick = (o) => { input.value = o; hide(); input.dispatchEvent(new Event('input', { bubbles: true })); input.dispatchEvent(new Event('change', { bubbles: true })); };
    input.addEventListener('focus', show); input.addEventListener('input', show); input.addEventListener('blur', () => setTimeout(hide, 120));
    input.addEventListener('keydown', (e) => {
      if (ul.hidden) return;
      if (e.key === 'ArrowDown') { e.preventDefault(); idx = (idx + 1) % shown.length; paint(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); idx = (idx - 1 + shown.length) % shown.length; paint(); }
      else if (e.key === 'Enter' && idx >= 0) { e.preventDefault(); pick(shown[idx]); }
      else if (e.key === 'Escape') hide();
    });
  });



  // ---- live filtering: swap #results without a full reload
  const form = $('form[data-live]');
  if (form) {
    const results = $('#results');
    $$('[data-nojs-show]', form).forEach((el) => (el.hidden = true));
    let t, ctl;
    const run = async (push = true) => {
      const p = new URLSearchParams(new FormData(form));
      for (const [k, v] of [...p]) if (v === '') p.delete(k, v);
      if (p.get('sort') === 'relevance') p.delete('sort');
      if (p.get('mode') === 'all') p.delete('mode');
      p.delete('page');
      const url = form.getAttribute('action') + (p.toString() ? '?' + p : '');
      ctl?.abort(); ctl = new AbortController();
      results.style.opacity = '.55';
      try {
        const html = await (await fetch(url, { signal: ctl.signal })).text();
        const doc = new DOMParser().parseFromString(html, 'text/html');
        const fresh = $('#results', doc);
        if (fresh) { results.innerHTML = fresh.innerHTML; if (push) history.replaceState(null, '', url); }
        $$('fieldset .check', form).forEach(() => {});
      } catch (e) { if (e.name !== 'AbortError') location.href = url; }
      results.style.opacity = '';
    };
    const queue = () => { clearTimeout(t); t = setTimeout(run, 280); };
    form.addEventListener('change', queue);
    form.addEventListener('input', (e) => { if (e.target.matches('input[type=search],input[type=number],input[type=text]')) queue(); });
    form.addEventListener('submit', (e) => { e.preventDefault(); run(); });
    results.addEventListener('change', (e) => { if (e.target.name === 'sort') queue(); });
  }

  // ---- range inputs show their value (re-bound after live swaps via delegation)
  document.addEventListener('input', (e) => {
    const r = e.target;
    if (r.matches?.('input[type=range]')) {
      const o = r.closest('.range-row')?.querySelector('output');
      if (o) o.textContent = r.value;
    }
  });

  // ---- dual-handle experience range
  $$('[data-dual]').forEach((d) => {
    const [a, b] = $$('input[type=range]', d), read = d.parentElement.querySelector('[data-dual-read]');
    const max = +a.max;
    const sync = (who) => {
      if (+a.value > +b.value) { if (who === a) b.value = a.value; else a.value = b.value; }
      d.style.setProperty('--lo', (a.value / max) * 100); d.style.setProperty('--hi', (b.value / max) * 100);
      if (read) read.textContent = `${a.value} to ${b.value} yrs`;
    };
    a.addEventListener('input', () => sync(a)); b.addEventListener('input', () => sync(b));
  });

  // ---- chip pickers: count, search, clear, presets
  $$('[data-picker]').forEach((p) => {
    const step = p.closest('.step'), count = $('[data-count]', step);
    const boxes = () => $$('input[type=checkbox]', p);
    const upd = () => {
      if (!count) return;
      const extra = $$('input[type=hidden]', step).length;
      const n = boxes().filter((c) => c.checked).length + extra;
      count.textContent = n + ' selected';
    };
    const fire = () => { upd(); p.closest('form')?.dispatchEvent(new Event('change', { bubbles: true })); };
    p.addEventListener('change', (e) => { if (e.target.type === 'checkbox') upd(); });
    $('[data-picker-search]', p)?.addEventListener('input', (e) => {
      const q = e.target.value.trim().toLowerCase();
      $$('.cc', p).forEach((c) => (c.hidden = q && !c.textContent.toLowerCase().includes(q)));
      $$('.pgroup', p).forEach((g) => (g.hidden = !$$('.cc:not([hidden])', g).length));
    });
    $('[data-picker-clear]', p)?.addEventListener('click', () => { boxes().forEach((c) => (c.checked = false)); fire(); });
    $$('[data-preset]', p).forEach((b) => b.addEventListener('click', () => {
      const set = new Set(b.dataset.preset.split(','));
      boxes().forEach((c) => (c.checked = set.has(c.value))); fire();
    }));
    p.closest('form')?.addEventListener('change', upd);
    upd();
  });

  // ---- career profiles
  $$('[data-ratings]').forEach((b) => b.addEventListener('click', () => {
    const vals = JSON.parse(b.dataset.ratings);
    vals.forEach((v, i) => { const r = $(`input[name=r${i}][value="${v}"]`); if (r) r.checked = true; });
    b.closest('form').dispatchEvent(new Event('change', { bubbles: true }));
  }));

  // ---- hero pulse: the sentence drives a live salary ribbon
  const hero = $('#hero-form');
  if (hero) {
    let t, ctl;
    const fig = $('#p-fig'), meta = $('#p-meta'), chart = $('#p-chart'), go = $('#p-go');
    const run = async () => {
      const p = new URLSearchParams(new FormData(hero));
      for (const [k, v] of [...p]) if (v === '') p.delete(k, v);
      ctl?.abort(); ctl = new AbortController(); chart.classList.add('busy');
      try {
        const d = await (await fetch('/api/pulse?' + p, { signal: ctl.signal })).json();
        if (d.n) {
          const g = (x) => +x.toFixed(x % 1 ? 1 : 0);
          fig.innerHTML = `₹${g(d.med)}<small>LPA median</small>`;
          meta.innerHTML = `<div><b>${d.n.toLocaleString('en-IN')}</b>postings</div><div><b>${Math.round(d.q1)}–${Math.round(d.q3)}</b>LPA, middle half</div><div><b>${d.companies.toLocaleString('en-IN')}</b>employers</div>`;
          chart.innerHTML = d.svg; go.disabled = false;
        } else {
          fig.textContent = '–'; meta.innerHTML = '';
          chart.innerHTML = '<p class="none">No postings match that combination. Try a broader title, another city or fewer skills.</p>'; go.disabled = true;
        }
        history.replaceState(null, '', '/?' + p);
      } catch (e) { if (e.name !== 'AbortError') console.error(e); }
      chart.classList.remove('busy');
    };
    const queue = () => { clearTimeout(t); t = setTimeout(run, 220); };
    hero.addEventListener('change', queue);
    hero.addEventListener('input', (e) => { if (e.target.matches('input[type=number],input[type=text]')) queue(); });
  }

  // ---- theme toggle
  const tbtn = $('#theme'), root = document.documentElement;
  const dark = () => root.dataset.theme === 'dark' || (!root.dataset.theme && matchMedia('(prefers-color-scheme: dark)').matches);
  const paint = () => tbtn?.setAttribute('aria-pressed', dark());
  paint();
  tbtn?.addEventListener('click', () => {
    const next = dark() ? 'light' : 'dark';
    root.dataset.theme = next;
    try { localStorage.setItem('sp-theme', next); } catch (e) {}
    paint();
  });
})();
