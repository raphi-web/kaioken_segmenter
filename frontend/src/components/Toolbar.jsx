import { CLASSES } from '../constants'
import Dropdown from './Dropdown'

export default function Toolbar({
  tool, setTool, epochs, setEpochs, status, starting, labeledPixels, undoDepth,
  onTrain, onStop, onUndo, onAdoptPrediction, onClearLabels, onExportModel,
  onExportOnnx, onExportExecutable, onExportMask, onAccuracyReport, onReset,
  hasProject, onNewProject, onNewFromGeotiff, onSaveProject, onSaveUserMask, onOpenSettings,
  samAvailable, executableAvailable,
}) {
  const training = status?.state === 'training'
  // The gap between the click and the first epoch is backend work, not idleness,
  // so it locks the same controls a running epoch does.
  const busy = training || starting
  // A run is "training" from the click onward, but it spends its first stretch
  // scanning masks rather than fitting, and the two have separate readouts.
  const collecting = training && status.stage === 'collecting'
  const running = training && status.stage !== 'collecting'
  const trainPct = running && status.epochs ? Math.round((status.epoch / status.epochs) * 100) : 0
  const scanPct = collecting && status.scan_total
    ? Math.round((status.scan_done / status.scan_total) * 100) : 0

  return (
    <div className="toolbar">
      <div className="toolbar-group">
        <span className="group-label">File</span>
        <Dropdown label="Project" disabled={busy}>
          <button onClick={onNewProject}>New / Open</button>
          <button
            onClick={onNewFromGeotiff}
            title="Cut one large GeoTIFF into tiles and use them as a new project"
          >
            From GeoTIFF…
          </button>
          <button disabled={!hasProject} onClick={onSaveProject}>Save Project</button>
          <button disabled={!hasProject} onClick={onSaveUserMask}>Save Masks</button>
          <button disabled={!hasProject} onClick={onOpenSettings}>Settings…</button>
        </Dropdown>
        <Dropdown label="Export / Model" disabled={busy}>
          <button onClick={onExportModel}>Export Model</button>
          <button onClick={onExportOnnx}>Export ONNX</button>
          <button
            disabled={!executableAvailable}
            title={executableAvailable
              ? 'Export a clickable standalone predictor (executable + model.onnx)'
              : 'Predictor executable not built (standalone/predictor.spec)'}
            onClick={onExportExecutable}
          >
            Export Executable
          </button>
          <button onClick={onExportMask}>Export Mask</button>
          <button
            disabled={!hasProject}
            title={hasProject
              ? 'Score the model on the held-out validation images and on the training images'
              : 'Open a project to score its validation images'}
            onClick={onAccuracyReport}
          >
            Generate Accuracy Report
          </button>
          <button className="danger" onClick={onReset}>Reset Model</button>
        </Dropdown>
      </div>

      <div className="toolbar-group">
        <span className="group-label">Class</span>
        <div className="segmented">
          {CLASSES.map((c) => (
            <button
              key={c.id}
              className={!tool.eraser && tool.classId === c.id ? 'active' : ''}
              style={{ '--class-color': `rgb(${c.color.join(',')})` }}
              onClick={() => setTool({ ...tool, classId: c.id, eraser: false })}
            >
              <span className="swatch" />
              {c.name}
            </button>
          ))}
          <button
            className={tool.eraser ? 'active' : ''}
            onClick={() => setTool({ ...tool, eraser: !tool.eraser })}
          >
            Eraser
          </button>
        </div>
      </div>

      <div className="toolbar-group">
        <span className="group-label">Tool</span>
        <div className="segmented">
          <button
            className={tool.mode === 'brush' ? 'active' : ''}
            onClick={() => setTool({ ...tool, mode: 'brush' })}
          >
            Brush
          </button>
          <button
            className={tool.mode === 'polygon' ? 'active' : ''}
            onClick={() => setTool({ ...tool, mode: 'polygon' })}
          >
            Polygon
          </button>
          <button
            className={tool.mode === 'sam' ? 'active' : ''}
            disabled={!samAvailable}
            title={samAvailable
              ? 'Click an object to snap a polygon to its edges (Efficient-SAM2)'
              : 'SAM2 model files not found in sam2/'}
            onClick={() => setTool({ ...tool, mode: 'sam' })}
          >
            SAM2
          </button>
        </div>
        <button disabled={!undoDepth} onClick={onUndo} title="Ctrl+Z">Undo</button>
        <button
          disabled={busy}
          onClick={onAdoptPrediction}
          title="Fill all unlabeled pixels with the current model prediction (your painted pixels are kept); undoable"
        >
          Adopt Prediction
        </button>
        <button
          className="danger"
          disabled={busy || !labeledPixels}
          onClick={onClearLabels}
          title="Remove all labels from this image (undoable with Ctrl+Z)"
        >
          Clear Labels
        </button>
      </div>

      <div className="toolbar-group">
        <span className="group-label">Brush {tool.brushSize}px</span>
        <input
          type="range"
          min="1"
          max="40"
          value={tool.brushSize}
          onChange={(e) => setTool({ ...tool, brushSize: Number(e.target.value) })}
          onWheel={(e) => {
            e.preventDefault()
            const next = tool.brushSize + (e.deltaY < 0 ? 1 : -1)
            setTool({ ...tool, brushSize: Math.min(40, Math.max(1, next)) })
          }}
        />
      </div>

      <div className="toolbar-group">
        <span className="group-label">Epochs</span>
        <input
          type="number"
          min="1"
          max="200"
          title="Run as entered. A short run may stay in warm-up throughout: the semi-supervised phase only starts once the model holds a stable target IoU"
          value={epochs}
          disabled={busy}
          onChange={(e) => setEpochs(Number(e.target.value))}
        />
        {training
          ? <button className="danger" onClick={onStop}>Stop ({status.epoch}/{status.epochs})</button>
          : <button className="primary" onClick={onTrain} disabled={starting}>
              {starting ? 'Starting…' : 'Train'}
            </button>}
      </div>

      <div className="toolbar-group status">
        {status?.state === 'error' && (
          <div className="stat">
            <span className="stat-label">Error</span>
            {/* Last non-empty line: the message for a plain training failure
                (nothing labeled, all held out), the exception line for a
                traceback. Indexing from the end alone breaks on the former. */}
            <span className="stat-value error">
              {status.error?.split('\n').filter(Boolean).at(-1) ?? 'see console'}
            </span>
          </div>
        )}
        {starting && !training && (
          <>
            <span className="spinner" />
            <div className="stat">
              <span className="stat-label">Status</span>
              <span className="stat-value">Starting…</span>
            </div>
          </>
        )}
        {collecting && (
          <>
            <span className="spinner" />
            <div className="stat" title="Reading and shape-checking every mask in the project to find the labeled images">
              <span className="stat-label">Collecting</span>
              <span className="stat-value">
                {status.scan_total
                  ? `${status.scan_done}/${status.scan_total} images`
                  : 'labeled images…'}
              </span>
            </div>
            {status.scan_total > 0 && (
              <div className="train-meter">
                <span style={{ width: `${scanPct}%` }} />
              </div>
            )}
          </>
        )}
        {running && (
          <>
            <span className="spinner" />
            <div className="stat">
              <span className="stat-label">Epoch</span>
              <span className="stat-value accent">{status.epoch}/{status.epochs}</span>
            </div>
            <div className="train-meter">
              <span style={{ width: `${trainPct}%` }} />
            </div>
            {status.images > 1 && (
              <div className="stat">
                <span className="stat-label">Images</span>
                <span className="stat-value">{status.images}</span>
              </div>
            )}
            {status.loss_sup != null && (
              <>
                {/* The gate opens on measured IoU, not on a fixed epoch, so
                    without this the stretch of zeroed Pseudo/Cons below reads
                    as a fault rather than as the warm-up doing its job. */}
                <div className="stat" title="Pseudo-labeling and consistency stay off until the model holds a stable target IoU, then fade in over the rest of the run">
                  <span className="stat-label">Phase</span>
                  <span className="stat-value">
                    {status.ramp ? `Semi-sup ${Math.round(status.ramp * 100)}%` : 'Warm-up'}
                  </span>
                </div>
                <div className="stat">
                  <span className="stat-label">Sup</span>
                  <span className="stat-value">{status.loss_sup}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Pseudo</span>
                  <span className="stat-value">{status.loss_pseudo}</span>
                </div>
                <div className="stat">
                  <span className="stat-label">Cons</span>
                  <span className="stat-value">{status.loss_cons}</span>
                </div>
                {status.iou != null && (
                  <div className="stat" title="Target-class IoU vs your labels over the labeled training pixels, this epoch">
                    <span className="stat-label">IoU</span>
                    <span className="stat-value accent">{status.iou.toFixed(3)}</span>
                  </div>
                )}
              </>
            )}
          </>
        )}
      </div>
    </div>
  )
}
