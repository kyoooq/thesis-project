# Flask API — GAD Thesis Backend

## Install

```
pip install -r requirements.txt
pip install -r requirements-flask.txt
python -m spacy download en_core_web_lg
```

## Run

```
python app.py
```

Serves at `http://127.0.0.1:5000`.

First startup takes ~15-30 seconds while BERT + spaCy load into memory.
Subsequent requests are fast.

## Endpoints

### `GET /api/health`
Sanity check. Returns `{"status": "ok"}`.

### `POST /api/assess`
Upload and analyze a paper.

**Request:** `multipart/form-data` with a single `file` field (.pdf or .docx).

**Response (200):**
```json
{ "resultId": "7c4a8d09-..." }
```

**Errors:**
- `400` — no file, wrong extension
- `413` — file larger than 25 MB
- `500` — analysis failed

### `GET /api/results/<result_id>`
Fetch a stored assessment.

**Response (200):** the full assessment dict (overallScore, overallLabel, stats, rows, ...).

**Errors:**
- `400` — malformed id
- `404` — result not found

## Testing without the frontend

```
# upload
curl -F "file=@path/to/paper.pdf" http://127.0.0.1:5000/api/assess

# fetch by id
curl http://127.0.0.1:5000/api/results/<id-from-above>
```

## Hooking up `home.html`

Replace the `TODO: POST /api/assess` block in `home.html`'s `handleAssess()`:

```javascript
async function handleAssess() {
    setUploadState('processing');

    const formData = new FormData();
    formData.append('file', fileObj);   // fileObj comes from your upload state

    try {
        const res = await fetch('http://127.0.0.1:5000/api/assess', {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.error || 'Assessment failed.');
        }

        const data = await res.json();
        window.location.href = `results.html?id=${data.resultId}`;
    } catch (err) {
        alert(err.message);
        setUploadState('ready');
    }
}
```

## Hooking up `results.html`

Replace the `TODO: Fetch real results from API` block:

```javascript
useEffect(() => {
    const id = new URLSearchParams(window.location.search).get('id');
    if (!id) return;

    fetch(`http://127.0.0.1:5000/api/results/${id}`)
        .then(r => r.ok ? r.json() : Promise.reject(r))
        .then(data => setResults(data))
        .catch(err => console.error('Failed to load results:', err));
}, []);
```

## Storage layout

```
storage/
    uploads/    <result_id>.pdf | <result_id>.docx
    results/    <result_id>.json
```

Both directories are created on first run. Files accumulate — delete
manually if you need to reset.

## Known limitations

- No auth yet. Anyone who can hit the server can upload and retrieve any
  result by id. Fine for local development; add auth before hosting anywhere.
- The Flask dev server is single-process and single-threaded. Concurrent
  uploads will queue. Also fine for thesis demo; use gunicorn or similar
  for production.
- `MAX_CONTENT_LENGTH` is 25 MB; adjust in `app.py` if you need larger files.
