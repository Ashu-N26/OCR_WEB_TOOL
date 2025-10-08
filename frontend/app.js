// frontend/app.js
document.getElementById('btnUpload').addEventListener('click', uploadFiles);

function escapeHtml(s) {
  return s.replace(/[&<>"'`]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;','`':'&#96;'}[c]));
}

async function uploadFiles() {
  const input = document.getElementById('files');
  const status = document.getElementById('status');
  const output = document.getElementById('output');
  output.innerHTML = '';
  if (!input.files || input.files.length === 0) {
    status.textContent = 'Please choose at least one file.';
    return;
  }

  status.textContent = 'Uploading...';
  try {
    const results = [];
    // send files sequentially (makes server errors easier to debug)
    for (let i=0; i < input.files.length; i++) {
      const file = input.files[i];
      status.textContent = `Uploading ${file.name} (${i+1}/${input.files.length})...`;

      const fd = new FormData();
      // IMPORTANT: backend expects form key "file" (not "files")
      fd.append('file', file);

      // Use relative path so it works on Render combined deployment
      const resp = await fetch('/extract', {
        method: 'POST',
        body: fd
      });

      // Throw helpful error if server responds non-2xx
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`Server ${resp.status}: ${text}`);
      }

      const json = await resp.json();
      results.push(json);
    }

    // Display results
    status.textContent = 'Done.';
    results.forEach(r => {
      const card = document.createElement('div');
      card.className = 'result-card';
      const h = document.createElement('h3');
      h.textContent = r.filename || 'result';
      card.appendChild(h);

      const pre = document.createElement('pre');
      pre.textContent = r.extracted_text || JSON.stringify(r, null, 2);
      card.appendChild(pre);

      output.appendChild(card);
    });

  } catch (err) {
    console.error(err);
    status.textContent = 'Error: ' + err.message;
    output.innerHTML = `<pre>${escapeHtml(err.message)}</pre>`;
  }
}

