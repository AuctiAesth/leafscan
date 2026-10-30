/* LeafScan Second Labeller: a second team member labels exported study photos without seeing the first label
   (the code in the file name), the team's guess or LeafScan's answer. Photos come from a study ZIP exported by
   Study mode; problems and checks come from the same data/field_codes.json and data/kb.json as the app and the
   printed Field ID Key. An agriculturist uses the same page (role "agriculturist", letter X): the team's checks are
   hidden for them, and they can add a diagnosis in their own words and how sure they are. The labels are downloaded
   as a CSV that training/label_agreement.py compares with the first labels (Cohen's kappa). Nothing is uploaded. */
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const el = {
    rater: $('rater'), zipInput: $('zipInput'), startNote: $('startNote'), startCard: $('startCard'),
    labelCard: $('labelCard'), counter: $('counter'), doneCount: $('doneCount'), bar: $('bar'), photo: $('photo'),
    wholeBox: $('wholeBox'), crop: $('crop'), problems: $('problems'), check: $('check'),
    prev: $('prev'), next: $('next'), doneCard: $('doneCard'), summary: $('summary'), download: $('download'),
    role: $('role'), diag: $('diag'), sure: $('sure'),
  };

  let codes = null;
  let kb = null;
  let items = [];      // photos to label, in this labeller's random order
  let wholeFor = {};   // "CROP-SITE-Pnn" -> object URL of the L0 whole-plant photo
  let underFor = {};   // "CROP-SITE-Pnn-Ln" -> object URL of that leaf's underside photo (Un)
  let pos = 0;
  let labels = {};     // photo_file -> { code, text, sure }
  let storeKey = '';
  let zipName = '';

  const store = {
    get(k) { try { return JSON.parse(localStorage.getItem(k)) || {}; } catch { return {}; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch { /* the CSV download still works */ } },
  };

  // ---------- ZIP reading (stored entries from LeafScan, or deflated if someone re-zipped the export) ----------
  async function readZip(file) {
    const buf = new Uint8Array(await file.arrayBuffer());
    const dv = new DataView(buf.buffer);
    let eocd = -1;
    for (let i = buf.length - 22; i >= Math.max(0, buf.length - 65557); i--) {
      if (dv.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
    }
    if (eocd < 0) throw new Error('This file is not a ZIP.');
    const count = dv.getUint16(eocd + 10, true);
    let p = dv.getUint32(eocd + 16, true);
    const files = new Map();
    for (let n = 0; n < count; n++) {
      if (dv.getUint32(p, true) !== 0x02014b50) throw new Error('The ZIP is damaged.');
      const method = dv.getUint16(p + 10, true);
      const size = dv.getUint32(p + 20, true);
      const nameLen = dv.getUint16(p + 28, true);
      const extraLen = dv.getUint16(p + 30, true);
      const commentLen = dv.getUint16(p + 32, true);
      const local = dv.getUint32(p + 42, true);
      const name = new TextDecoder().decode(buf.subarray(p + 46, p + 46 + nameLen));
      const start = local + 30 + dv.getUint16(local + 26, true) + dv.getUint16(local + 28, true);
      files.set(name.split('/').pop(), { method, data: buf.subarray(start, start + size) });
      p += 46 + nameLen + extraLen + commentLen;
    }
    return files;
  }

  async function unzip(entry) {
    if (entry.method === 0) return entry.data;
    if (entry.method === 8 && 'DecompressionStream' in window) {
      const stream = new Blob([entry.data]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
      return new Uint8Array(await new Response(stream).arrayBuffer());
    }
    throw new Error('This ZIP uses a compression this browser cannot open. Send the original export.');
  }

  function parseCsv(text) {
    const rows = [];
    let row = [], cell = '', quoted = false;
    for (let i = 0; i < text.length; i++) {
      const c = text[i];
      if (quoted) {
        if (c === '"' && text[i + 1] === '"') { cell += '"'; i++; } else if (c === '"') quoted = false; else cell += c;
      } else if (c === '"') quoted = true;
      else if (c === ',') { row.push(cell); cell = ''; } else if (c === '\n') { row.push(cell.replace(/\r$/, '')); rows.push(row); row = []; cell = ''; } else cell += c;
    }
    if (cell || row.length) { row.push(cell); rows.push(row); }
    const head = rows.shift() || [];
    return rows.filter((r) => r.length > 1).map((r) => Object.fromEntries(head.map((h, i) => [h.replace(/^﻿/, ''), r[i] || ''])));
  }

  // Same order every time for this labeller and ZIP, different between labellers.
  function shuffle(list, seedText) {
    let h = 1779033703;
    for (const ch of seedText) h = Math.imul(h ^ ch.charCodeAt(0), 3432918353), h = (h << 13) | (h >>> 19);
    const rand = () => { h = Math.imul(h ^ (h >>> 16), 2246822507); h = Math.imul(h ^ (h >>> 13), 3266489909); return ((h ^= h >>> 16) >>> 0) / 4294967296; };
    const a = list.slice();
    for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; }
    return a;
  }

  // ---------- Problems and checks (same names and checks as Study mode) ----------
  const cropOf = (code) => codes.crops.find((c) => c.code === code) || null;
  const problemsOf = (crop) => (crop ? [...crop.problems, ...codes.common] : []);
  const recordOf = (id) => id && (kb.conditions.find((c) => c.id === id) || (kb.more || []).find((m) => m.id === id));
  function problemName(p) {
    if (p.id && p.id.endsWith('/healthy')) return 'Healthy';
    const r = recordOf(p.id);
    return r ? (r.name || r.en) : p.en;
  }

  async function open(file) {
    const rater = el.rater.value.trim().toUpperCase();
    if (!/^[A-Z]$/.test(rater)) { el.startNote.textContent = 'Type your member letter first.'; el.zipInput.value = ''; return; }
    el.startNote.textContent = 'Opening…';
    try {
      if (!codes) {
        [codes, kb] = await Promise.all([fetch('data/field_codes.json').then((r) => r.json()), fetch('data/kb.json').then((r) => r.json())]);
      }
      const files = await readZip(file);
      const log = files.get('photo_log.csv');
      if (!log) throw new Error('No photo_log.csv in this ZIP: is it a LeafScan study export?');
      const rows = parseCsv(new TextDecoder().decode(await unzip(log)));
      const url = async (name) => {
        const f = files.get(name);
        return f ? URL.createObjectURL(new Blob([await unzip(f)], { type: 'image/jpeg' })) : '';
      };
      wholeFor = {};
      underFor = {};
      const todo = [];
      let own = 0, uncoded = 0;
      for (const r of rows) {
        if (!r.code || !cropOf(r.crop_code)) { uncoded++; continue; }
        if (r.site_code[0] === rater) { own++; continue; }
        const plant = [r.crop_code, r.site_code, r.plant_no].join('-');
        if (r.leaf_no === 'L0') { wholeFor[plant] = r.photo_file; continue; }
        if (r.leaf_no[0] === 'U') { underFor[`${plant}-L${r.leaf_no.slice(1)}`] = r.photo_file; continue; }
        todo.push({ file: r.photo_file, crop: r.crop_code, plant, leaf: `${plant}-${r.leaf_no}` });
      }
      if (!todo.length) throw new Error(own ? 'Every coded photo in this ZIP is yours: label a teammate\'s ZIP.' : 'This ZIP has no coded photos of the study crops.');
      for (const k of Object.keys(wholeFor)) wholeFor[k] = await url(wholeFor[k]);
      for (const k of Object.keys(underFor)) underFor[k] = await url(underFor[k]);
      for (const it of todo) it.url = await url(it.file);
      zipName = file.name;
      items = shuffle(todo.filter((it) => it.url), `${rater}|${zipName}`);
      storeKey = `leafscan.label2.${rater}.${zipName}`;
      labels = store.get(storeKey);
      for (const [k, v] of Object.entries(labels)) if (typeof v === 'string') labels[k] = { code: v };
      pos = Math.max(0, items.findIndex((it) => !labelled(it)));
      if (items.every(labelled)) pos = 0;
      el.startNote.textContent = `${items.length} photos to label` +
        (own ? `; ${own} of your own skipped` : '') + (uncoded ? `; ${uncoded} without a study code skipped` : '') + '.';
      el.labelCard.hidden = false;
      el.doneCard.hidden = false;
      el.rater.disabled = true;
      el.role.disabled = true;
      show();
    } catch (err) {
      console.error(err);
      el.startNote.textContent = '';
      el.startNote.append(Object.assign(document.createElement('span'), { className: 'lb-error', textContent: err.message }));
    }
  }

  const isExpert = () => el.role.value === 'expert';
  const labelled = (x) => !!(labels[x.file] && (labels[x.file].code || labels[x.file].text));
  function save(file, change) {
    labels[file] = { ...(labels[file] || {}), ...change };
    store.set(storeKey, labels);
  }

  function show() {
    const it = items[pos];
    const cur = labels[it.file] || {};
    const done = items.filter(labelled).length;
    el.counter.textContent = `Photo ${pos + 1} of ${items.length}`;
    el.doneCount.textContent = `${done} labelled`;
    el.bar.style.width = `${(100 * done) / items.length}%`;
    // The leaf to label, plus the plant's whole-plant photo and this leaf's underside when they were taken.
    const views = [['This leaf', it.url], ['Whole plant', wholeFor[it.plant]], ['Underside', underFor[it.leaf]]].filter((v) => v[1]);
    const showView = (src) => {
      el.photo.src = src;
      for (const b of el.wholeBox.children) b.setAttribute('aria-pressed', String(b.dataset.src === src));
    };
    el.wholeBox.hidden = views.length < 2;
    el.wholeBox.replaceChildren(...views.map(([name, src]) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.dataset.src = src;
      b.append(Object.assign(document.createElement('img'), { src, alt: '' }), name);
      b.addEventListener('click', () => showView(src));
      return b;
    }));
    showView(it.url);
    const crop = cropOf(it.crop);
    const h = kb.hosts[crop.host] || {};
    el.crop.textContent = `${crop.code} · ${[h.fil, h.en].filter(Boolean).join(' / ')}: what problem do you see?`;
    el.problems.replaceChildren(...problemsOf(crop).map((p) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('aria-pressed', String(cur.code === p.code));
      const name = p.code === 'OTH' ? 'Other problem (write it below)' : problemName(p);
      b.append(Object.assign(document.createElement('b'), { textContent: p.code }), name);
      b.addEventListener('click', () => { save(it.file, { code: p.code }); show(); });
      return b;
    }));
    renderCheck(isExpert() ? null : problemsOf(crop).find((p) => p.code === cur.code));
    el.diag.value = cur.text || '';
    for (const b of el.sure.querySelectorAll('button')) b.setAttribute('aria-pressed', String(cur.sure === b.dataset.sure));
    el.prev.disabled = pos === 0;
    el.next.textContent = pos === items.length - 1 ? 'Finish' : 'Next';
    el.summary.textContent = `${done} of ${items.length} photos labelled from ${zipName}.` +
      (done < items.length ? ' Unlabelled photos are left out of the CSV.' : '');
  }

  function renderCheck(p) {
    const lines = [];
    if (p && p.tier) lines.push(`Tier ${p.tier}: ${codes.tiers[p.tier]}`);
    const r = p && recordOf(p.id);
    if (r && r.look_for && r.look_for.length) lines.push(`Check: ${r.look_for.join(' ')}`);
    else if (r && r.about) lines.push(r.about);
    el.check.hidden = !lines.length;
    el.check.replaceChildren(...lines.map((t) => Object.assign(document.createElement('p'), { textContent: t })));
  }

  el.zipInput.addEventListener('change', () => { if (el.zipInput.files[0]) open(el.zipInput.files[0]); });
  el.role.addEventListener('change', () => { if (isExpert() && !el.rater.value) el.rater.value = 'X'; });
  el.diag.addEventListener('input', () => { save(items[pos].file, { text: el.diag.value.trim() }); });
  el.diag.addEventListener('change', () => show());
  for (const b of el.sure.querySelectorAll('button')) {
    b.addEventListener('click', () => { save(items[pos].file, { sure: b.dataset.sure }); show(); });
  }
  el.prev.addEventListener('click', () => { if (pos > 0) { pos--; show(); } });
  el.next.addEventListener('click', () => {
    if (pos < items.length - 1) { pos++; show(); } else el.doneCard.scrollIntoView({ behavior: 'smooth' });
  });

  el.download.addEventListener('click', () => {
    const rater = el.rater.value.trim().toUpperCase();
    const rows = [['photo_file', 'second_labeller', 'role', 'second_problem_code', 'diagnosis', 'how_sure', 'zip']];
    for (const it of items) {
      const l = labels[it.file];
      if (labelled(it)) rows.push([it.file, rater, el.role.value, l.code || '', l.text || '', l.sure || '', zipName]);
    }
    const csv = rows.map((r) => r.map((v) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v)).join(',')).join('\r\n') + '\r\n';
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob(['﻿' + csv], { type: 'text/csv' }));
    a.download = `second_labels_${rater}_${zipName.replace(/\.zip$/i, '')}.csv`;
    document.body.append(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 30000);
  });
})();
