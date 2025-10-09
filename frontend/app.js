// frontend/app.js
document.getElementById('btnUpload')?.addEventListener('click', uploadFiles);

function escapeHtml(s) {
  return String(s).replace(/[&<>"'`]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;','`':'&#96;'}[c]));
}

async function uploadFiles() {
  const input = document.getElementById('files');
  const status = document.getElementById('status');
  const output = document.getElementById('output');
  output.innerHTML = '';
  if (!input || !input.files || input.files.length === 0) {
    status.textContent = 'Please choose at least one file.';
    return;
  }

  status.textContent = 'Uploading...';
  try {
    for (let i = 0; i < input.files.length; i++) {
      const file = input.files[i];
      status.textContent = `Uploading ${file.name} (${i+1}/${input.files.length})...`;
      const fd = new FormData();
      fd.append('file', file);

      const resp = await fetch('/extract', {
        method: 'POST',
        body: fd
      });

      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`Server ${resp.status}: ${txt}`);
      }

      const json = await resp.json();
      renderResult(json, output);
    }
    status.textContent = 'Done.';
  } catch (err) {
    console.error(err);
    status.textContent = 'Error: ' + err.message;
    output.innerHTML = `<pre>${escapeHtml(err.message)}</pre>`;
  }
}

function renderResult(result, container) {
  const wrapper = document.createElement('div');
  wrapper.className = 'result-card';

  const title = document.createElement('h3');
  title.textContent = result.filename || 'result';
  wrapper.appendChild(title);

  // extracted text
  const txtHead = document.createElement('h4');
  txtHead.textContent = 'Extracted Text:';
  wrapper.appendChild(txtHead);
  const pre = document.createElement('pre');
  pre.textContent = result.extracted_text || '';
  wrapper.appendChild(pre);

  // tables
  if (result.tables && result.tables.length > 0) {
    const tbHead = document.createElement('h4');
    tbHead.textContent = 'Detected Tables:';
    wrapper.appendChild(tbHead);

    result.tables.forEach((t, idx) => {
      const meta = document.createElement('div');
      meta.textContent = `Table ${idx+1} — page: ${t.page || '?'} — type: ${t.type || 'ocr'}`;
      wrapper.appendChild(meta);

      // build HTML table
      const htmlTable = document.createElement('table');
      htmlTable.className = 'extracted-table';
      if (t.data && t.data.length > 0) {
        t.data.forEach((row, rIdx) => {
          const tr = document.createElement('tr');
          // if row is an object (camelot may return dicts), convert to array
          const cells = Array.isArray(row) ? row : Object.values(row);
          cells.forEach(cell => {
            const td = document.createElement(rIdx === 0 ? 'th' : 'td');
            td.innerHTML = escapeHtml(cell === null ? '' : String(cell));
            tr.appendChild(td);
          });
          htmlTable.appendChild(tr);
        });
      } else {
        const tr = document.createElement('tr');
        const td = document.createElement('td');
        td.textContent = 'No table rows detected';
        tr.appendChild(td);
        htmlTable.appendChild(tr);
      }

      wrapper.appendChild(htmlTable);

      // add CSV download button
      const dlBtn = document.createElement('button');
      dlBtn.textContent = 'Download Table CSV';
      dlBtn.addEventListener('click', () => downloadTableCSV(t.data, `${result.filename || 'table'}_table${idx+1}.csv`));
      wrapper.appendChild(dlBtn);
    });
  } else {
    const nt = document.createElement('div');
    nt.textContent = 'No tables detected.';
    wrapper.appendChild(nt);
  }

  container.appendChild(wrapper);
}

function downloadTableCSV(data, filename) {
  if (!data || data.length === 0) {
    alert('No table data to download');
    return;
  }
  let csvLines = [];
  data.forEach(row => {
    const cells = Array.isArray(row) ? row : Object.values(row);
    const escaped = cells.map(c => {
      if (c === null || c === undefined) return '';
      const s = String(c).replace(/"/g, '""');
      return `"${s}"`;
    });
    csvLines.push(escaped.join(','));
  });
  const blob = new Blob([csvLines.join('\n')], { type: 'text/csv' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}


