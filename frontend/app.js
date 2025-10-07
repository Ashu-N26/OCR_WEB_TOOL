async function uploadFiles() {
    const filesInput = document.getElementById('files');
    const files = filesInput.files;
    if (files.length === 0) return alert("Select files first");

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) formData.append("files", files[i]);

    const res = await fetch("http://localhost:8000/upload", {
        method: "POST",
        body: formData
    });
    const data = await res.json();
    displayResults(data.results);
}

function displayResults(results) {
    const output = document.getElementById('output');
    output.innerHTML = "";

    results.forEach((file, index) => {
        const div = document.createElement('div');
        div.innerHTML = `<h2>${file.filename}</h2>
                         <h3>Text:</h3><pre>${file.text}</pre>
                         <h3>Tables:</h3><pre>${JSON.stringify(file.tables, null, 2)}</pre>
                         <button onclick="downloadText(${index})">Download Text</button>
                         <button onclick="downloadTables(${index})">Download Tables (CSV)</button>`;
        output.appendChild(div);
    });

    window.resultsData = results;
}

function downloadText(index) {
    const data = window.resultsData[index].text;
    const blob = new Blob([data], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = window.resultsData[index].filename + "_text.txt";
    a.click();
    URL.revokeObjectURL(url);
}

function downloadTables(index) {
    const tables = window.resultsData[index].tables;
    if (tables.length === 0) return alert("No tables to download");

    tables.forEach((table, i) => {
        let csv = [];
        const headers = Object.keys(table[0]);
        csv.push(headers.join(","));
        table.forEach(row => {
            csv.push(headers.map(h => row[h]).join(","));
        });
        const blob = new Blob([csv.join("\n")], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${window.resultsData[index].filename}_table${i+1}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    });
}
