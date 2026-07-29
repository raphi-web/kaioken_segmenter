import { useMemo, useState, useEffect } from 'react'

// Model accuracy on the held-out validation images next to the training
// images. A missing metric renders as an em dash, never as 0 — the two mean
// very different things and the backend deliberately sends null.
const fmt = (v) => (v === null || v === undefined ? '—' : v.toFixed(4))

const COLUMNS = [
  { key: 'name', label: 'image', type: 'string' },
  { key: 'role', label: 'set', type: 'string' },
  { key: 'status', label: 'status', type: 'string' },
  { key: 'labeled_px', label: 'labeled px', type: 'number' },
  { key: 'tp', label: 'TP', type: 'number' },
  { key: 'fp', label: 'FP', type: 'number' },
  { key: 'fn', label: 'FN', type: 'number' },
  { key: 'iou', label: 'IoU', type: 'number' },
  { key: 'f1', label: 'F1', type: 'number' },
]

// Rows sorted by one column at a time; missing metrics (null) always sink to
// the bottom regardless of direction, rather than flip-flopping to the top.
function sortRows(rows, sort) {
  if (!sort.key) return rows
  const { key, dir } = sort
  const type = COLUMNS.find((c) => c.key === key)?.type
  return [...rows].sort((a, b) => {
    const av = a[key]
    const bv = b[key]
    if (av == null && bv == null) return 0
    if (av == null) return 1
    if (bv == null) return -1
    if (type === 'string') return dir * av.localeCompare(bv)
    return dir * (av - bv)
  })
}

function SetSummary({ title, summary }) {
  if (!summary || !summary.images) {
    return (
      <div className="report-set">
        <h3>{title}</h3>
        <div className="report-empty">no images in this set</div>
      </div>
    )
  }
  if (!summary.scoreable) {
    return (
      <div className="report-set">
        <h3>{title}</h3>
        <div className="report-empty">
          not scoreable — no target labels among these {summary.images} images
        </div>
      </div>
    )
  }
  const { micro, macro } = summary
  return (
    <div className="report-set">
      <h3>{title}</h3>
      <div className="report-headline">{fmt(micro.iou)}</div>
      <div className="report-headline-label">target IoU (pooled)</div>
      <dl className="report-metrics">
        <dt>F1</dt><dd>{fmt(micro.f1)}</dd>
        <dt>precision</dt><dd>{fmt(micro.precision)}</dd>
        <dt>recall</dt><dd>{fmt(micro.recall)}</dd>
        <dt>accuracy</dt><dd>{fmt(micro.accuracy)}</dd>
        <dt>mean IoU</dt>
        <dd>{fmt(macro.iou)} <span className="hint">over {summary.macro_images}</span></dd>
      </dl>
      <div className="hint">
        {summary.scored_images} of {summary.images} images scored
      </div>
    </div>
  )
}

export default function AccuracyReport({ report, progress, error, onSave, onCancel, onClose, onOpenImage }) {
  const running = !report && !error
  const [sort, setSort] = useState({ key: null, dir: 1 })
  useEffect(() => {
    function onKey(e) {
      if (e.key === 'Escape' && !running) onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose, running])

  function toggleSort(key) {
    setSort((s) => (s.key === key ? { key, dir: -s.dir } : { key, dir: 1 }))
  }

  const rows = useMemo(() => sortRows(report?.rows ?? [], sort), [report, sort])

  const gap = report
    && report.validation.micro.iou !== null
    && report.training.micro.iou !== null
    ? report.training.micro.iou - report.validation.micro.iou
    : null

  return (
    <div className="modal-backdrop">
      <div className="modal wide">
        <h2>Accuracy Report</h2>
        <div className="modal-sub">
          {report
            ? `${report.project_name} — ${report.generated_at}, model trained for ${report.total_epochs} epochs`
            : 'Scoring the model over every labeled image…'}
        </div>

        {running && (
          <div className="tiling-progress">
            <div className="tiling-bar">
              <div
                className={progress?.total ? 'tiling-fill' : 'tiling-fill indeterminate'}
                style={progress?.total
                  ? { width: `${Math.round((progress.done / progress.total) * 100)}%` }
                  : undefined}
              />
            </div>
            <span className="hint">
              {progress?.total ? `image ${progress.done} / ${progress.total}` : 'preparing…'}
            </span>
          </div>
        )}

        {error && <div className="modal-error">{error}</div>}

        {report && (
          <>
            <div className="report-sets">
              <SetSummary title="Validation (held out)" summary={report.validation} />
              <SetSummary title="Training" summary={report.training} />
            </div>

            {gap !== null && (
              <div className="report-gap">
                Training scores <b>{gap.toFixed(4)}</b> IoU above validation
                {gap > 0.15 ? ' — a gap that large usually means the model has memorized your labels.' : '.'}
              </div>
            )}

            {!report.validation.images && (
              <div className="modal-error">
                No validation images. Right-click a thumbnail to hold one out, or
                raise the validation split when creating the project.
              </div>
            )}

            <div className="report-table-wrap">
              <table className="report-table">
                <thead>
                  <tr>
                    {COLUMNS.map((c) => (
                      <th key={c.key} className="sortable" onClick={() => toggleSort(c.key)}>
                        {c.label}
                        {sort.key === c.key ? (sort.dir === 1 ? ' ▲' : ' ▼') : ''}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => (
                    <tr key={r.name} className={r.status === 'scored' ? '' : 'dim'}>
                      <td
                        className="mono clickable"
                        title={`Double-click to switch the view to ${r.name}`}
                        onDoubleClick={() => onOpenImage(r.name)}
                      >
                        {r.name}
                      </td>
                      <td className={r.role === 'validation' ? 'role-val' : ''}>{r.role}</td>
                      <td>{r.status}</td>
                      <td>{r.labeled_px.toLocaleString()}</td>
                      <td>{r.tp.toLocaleString()}</td>
                      <td>{r.fp.toLocaleString()}</td>
                      <td>{r.fn.toLocaleString()}</td>
                      <td>{fmt(r.iou)}</td>
                      <td>{fmt(r.f1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        <div className="modal-actions">
          {running
            ? <button onClick={onCancel}>Cancel</button>
            : <>
                <button onClick={onClose}>Close</button>
                <button className="primary" disabled={!report} onClick={onSave}>
                  Save HTML…
                </button>
              </>}
        </div>
      </div>
    </div>
  )
}
