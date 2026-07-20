import { useEffect, useState } from 'react'
import { api } from '../bridge'

const RGB_CHANNELS = [{ key: 'r', label: 'R' }, { key: 'g', label: 'G' }, { key: 'b', label: 'B' }]

// A valid [R, G, B] band-index triple: default to the first three bands, and
// clamp every entry into the available band range.
function normalizeDisplayBands(bands, channels) {
  const out = Array.isArray(bands) ? bands.slice(0, 3) : []
  while (out.length < 3) out.push(Math.min(out.length, channels - 1))
  return out.map((b) => Math.min(Math.max(Number(b) || 0, 0), channels - 1))
}

// Number of tiles along one axis, mirroring data.tile_grid: fixed-size steps
// with the last one snapped back to the edge. Only an estimate for the dialog —
// the backend does the real thing.
function tileCount(total, tile, stride) {
  if (!total || !tile || stride < 1) return 0
  if (tile >= total) return 1
  const n = Math.floor((total - tile) / stride) + 1
  return (n - 1) * stride === total - tile ? n : n + 1
}

// Centered modal for a project's data profile. In 'create' mode (a folder
// without a project_config.json) the user reviews the proposed defaults before
// anything is written; in 'geotiff' mode it additionally collects the tile
// geometry used to cut one raster into a project; in 'edit' mode it edits the
// open project's settings. Band count and patch size define the model, so they
// are fixed once the project exists and become read-only in edit mode.
export default function ProjectSetup({ mode = 'create', root, source, defaults, error, busy, progress, onSubmit, onCancel }) {
  const [form, setForm] = useState(() => ({
    ...defaults,
    display_bands: normalizeDisplayBands(defaults.display_bands, defaults.input_channels),
  }))
  const editing = mode === 'edit'
  const tiling = mode === 'geotiff'
  const patchInvalid = !form.input_patch_size || form.input_patch_size % 32 !== 0

  // Tile validity: a tile smaller than the model patch yields images that
  // SentinelImage refuses to open, and the overlap must leave a forward step.
  const tileW = Number(form.tile_width) || 0
  const tileH = Number(form.tile_height) || 0
  const overlap = Number(form.overlap) || 0
  const tileTooSmall = tiling && Math.min(tileW, tileH) < Number(form.input_patch_size || 0)
  const overlapInvalid = tiling && (overlap < 0 || overlap >= Math.min(tileW, tileH))
  const outputMissing = tiling && !String(form.output_root || '').trim()
  const estimate = tiling && !tileTooSmall && !overlapInvalid
    ? tileCount(defaults.source_width, Math.min(tileW, defaults.source_width), tileW - overlap) *
      tileCount(defaults.source_height, Math.min(tileH, defaults.source_height), tileH - overlap)
    : 0

  // How many images the split would hold out, when the image count is known
  // (only when tiling — a plain folder is not counted until the project opens).
  // Floor, matching the backend's int(ratio * N): with few images this is 0,
  // which the user should see here rather than discover in the report.
  const validationCount = tiling && estimate
    ? Math.floor(estimate * (form.validation_ratio ?? 0.2))
    : null

  useEffect(() => {
    function onKey(e) {
      // Not while working: the Cancel button is disabled for the same reason,
      // and tiling keeps running on the backend regardless of the dialog.
      if (e.key === 'Escape' && !busy) onCancel()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onCancel, busy])

  function set(key, value) {
    setForm((f) => ({ ...f, [key]: value }))
  }

  // Changing the band count keeps existing names and pads with generic ones,
  // and clamps the displayed-band selection into the new range.
  function setChannels(n) {
    n = Math.max(1, Math.min(64, Math.round(n) || 1))
    setForm((f) => ({
      ...f,
      input_channels: n,
      band_names: Array.from({ length: n }, (_, i) => f.band_names[i] ?? `Band ${i + 1}`),
      display_bands: f.display_bands.map((b) => Math.min(b, n - 1)),
    }))
  }

  function setBandName(i, name) {
    setForm((f) => {
      const band_names = [...f.band_names]
      band_names[i] = name
      return { ...f, band_names }
    })
  }

  function setDisplayBand(channel, index) {
    setForm((f) => {
      const display_bands = [...f.display_bands]
      display_bands[channel] = index
      return { ...f, display_bands }
    })
  }

  // Re-probe the band count when the images folder changes (create only: in
  // edit mode the band count is fixed with the model, and when tiling the
  // folder does not exist yet — the count comes from the source raster).
  // Native folder chooser for the tiling output. The picker returns an existing
  // folder, which the backend requires to be empty — so the typed default
  // (a fresh, non-existent sibling) stays available as the other option.
  async function browseOutput() {
    const res = await api('pick_folder')
    if (res?.path) set('output_root', res.path)
  }

  async function probeFolder() {
    if (editing || tiling) return
    const res = await api('probe_bands', root, form.images_folder || '.')
    if (res.bands) setChannels(res.bands)
  }

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <h2>{editing ? 'Project Settings' : tiling ? 'New Project from GeoTIFF' : 'New Project'}</h2>
        <div className="modal-sub">
          {tiling
            ? `${source} — ${defaults.source_width}x${defaults.source_height} px, ${defaults.input_channels} bands`
            : root}
        </div>
        {tiling && !defaults.source_georeferenced && (
          <div className="modal-error">
            This GeoTIFF has no CRS; tiles will carry none either.
          </div>
        )}

        <div className="form-row">
          <label>Project name</label>
          <input
            type="text"
            value={form.project_name}
            onChange={(e) => set('project_name', e.target.value)}
          />
        </div>

        {tiling && (
          <>
            <div className="form-row">
              <label>Output folder</label>
              <div className="path-picker">
                <input
                  type="text"
                  value={form.output_root}
                  onChange={(e) => set('output_root', e.target.value)}
                  title="The tiles are written here and this becomes the project folder"
                />
                <button type="button" disabled={busy} onClick={browseOutput}>Browse…</button>
              </div>
            </div>
            <div className="form-row">
              <label>Tile size</label>
              <div className="tile-size">
                <input
                  type="number"
                  min="32"
                  step="32"
                  value={form.tile_width}
                  onChange={(e) => set('tile_width', Number(e.target.value))}
                />
                <span className="tile-x">×</span>
                <input
                  type="number"
                  min="32"
                  step="32"
                  value={form.tile_height}
                  onChange={(e) => set('tile_height', Number(e.target.value))}
                />
              </div>
              <span className={tileTooSmall ? 'modal-error' : 'hint'}>
                {tileTooSmall
                  ? `px — must be at least the patch size (${form.input_patch_size} px)`
                  : 'px, width × height'}
              </span>
            </div>
            <div className="form-row">
              <label>Overlap</label>
              <input
                type="number"
                min="0"
                step="16"
                value={form.overlap}
                onChange={(e) => set('overlap', Number(e.target.value))}
              />
              <span className={overlapInvalid ? 'modal-error' : 'hint'}>
                {overlapInvalid
                  ? 'px — must be smaller than the tile'
                  : `px between neighbouring tiles${estimate ? ` — about ${estimate} tiles` : ''}`}
              </span>
            </div>
          </>
        )}

        <div className="form-row">
          <label>Images folder</label>
          <input
            type="text"
            value={form.images_folder}
            onChange={(e) => set('images_folder', e.target.value)}
            onBlur={probeFolder}
            title="Relative to the project folder ('.' = the folder itself)"
          />
        </div>

        <div className="form-row">
          <label>Number of bands</label>
          <input
            type="number"
            min="1"
            max="64"
            value={form.input_channels}
            disabled={editing || tiling}
            onChange={(e) => setChannels(Number(e.target.value))}
          />
          <span className="hint">
            {editing
              ? 'fixed for this project'
              : tiling
                ? 'from the source GeoTIFF'
                : defaults.bands_detected ? 'detected from the images' : 'no readable image found'}
          </span>
        </div>

        <div className="form-row align-top">
          <label>Band names</label>
          <div className="band-grid">
            {form.band_names.map((name, i) => (
              <input
                key={i}
                type="text"
                value={name}
                placeholder={`Band ${i + 1}`}
                onChange={(e) => setBandName(i, e.target.value)}
              />
            ))}
          </div>
        </div>

        <div className="form-row">
          <label title="Which bands are mapped to the red, green and blue display channels">
            Displayed bands
          </label>
          <div className="rgb-bands">
            {RGB_CHANNELS.map((ch, c) => (
              <div key={ch.key} className="rgb-band">
                <span className={`rgb-dot rgb-${ch.key}`}>{ch.label}</span>
                <select
                  value={form.display_bands[c]}
                  onChange={(e) => setDisplayBand(c, Number(e.target.value))}
                >
                  {form.band_names.map((name, i) => (
                    <option key={i} value={i}>{name}</option>
                  ))}
                </select>
              </div>
            ))}
          </div>
        </div>

        <div className="form-row">
          <label>Patch size</label>
          <input
            type="number"
            min="32"
            max="512"
            step="32"
            value={form.input_patch_size}
            disabled={editing}
            onChange={(e) => set('input_patch_size', Number(e.target.value))}
          />
          <span className={patchInvalid ? 'modal-error' : 'hint'}>
            {editing ? 'fixed for this project' : 'px, multiple of 32'}
          </span>
        </div>

        <div className="form-row">
          <label title="Share of the images held out of training, used by the accuracy report">
            Validation split
          </label>
          <input
            type="number"
            min="0"
            max="90"
            step="5"
            value={Math.round((form.validation_ratio ?? 0.2) * 100)}
            disabled={editing}
            onChange={(e) => set('validation_ratio',
              Math.min(Math.max(Number(e.target.value) || 0, 0), 90) / 100)}
          />
          <span className="hint">
            {editing
              ? 'fixed for this project — right-click a thumbnail to move one image'
              : `% held out${validationCount !== null ? ` — about ${validationCount} image${validationCount === 1 ? '' : 's'}` : ''}`}
          </span>
        </div>

        <div className="form-row">
          <label title="Re-classify the most uncertain pixels with a small PointRend head for sharper boundaries">
            Boundary refinement
          </label>
          <input
            type="checkbox"
            checked={!!form.use_pointrend}
            onChange={(e) => set('use_pointrend', e.target.checked)}
          />
          <span className="hint">PointRend point head on uncertain pixels</span>
        </div>

        <div className="form-row">
          <label>User masks folder</label>
          <input
            type="text"
            value={form.masks_user}
            onChange={(e) => set('masks_user', e.target.value)}
          />
        </div>

        <div className="form-row">
          <label>AI masks folder</label>
          <input
            type="text"
            value={form.masks_ai}
            onChange={(e) => set('masks_ai', e.target.value)}
          />
        </div>

        {error && <div className="modal-error">{error}</div>}

        {progress && (
          <div className="tiling-progress">
            <div className="tiling-bar">
              <div
                className={progress.total ? 'tiling-fill' : 'tiling-fill indeterminate'}
                style={progress.total
                  ? { width: `${Math.round((progress.done / progress.total) * 100)}%` }
                  : undefined}
              />
            </div>
            <span className="hint">
              {progress.total
                ? `tiling ${progress.done} / ${progress.total}`
                : 'preparing…'}
            </span>
          </div>
        )}

        <div className="modal-actions">
          <button onClick={onCancel} disabled={busy}>Cancel</button>
          <button
            className="primary"
            disabled={busy || patchInvalid || !form.project_name.trim()
              || tileTooSmall || overlapInvalid || outputMissing}
            onClick={() => onSubmit(form)}
          >
            {editing
              ? (busy ? 'Saving…' : 'Save Settings')
              : (busy ? (tiling ? 'Tiling…' : 'Creating…') : 'Create Project')}
          </button>
        </div>
      </div>
    </div>
  )
}
