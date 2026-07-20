import { CLASSES } from '../constants'
import Dropdown from './Dropdown'

export default function Toolbar({
  tool, setTool, epochs, setEpochs, status, labeledPixels, undoDepth,
  onTrain, onStop, onUndo, onAdoptPrediction, onClearLabels, onExportModel,
  onExportOnnx, onExportExecutable, onExportMask, onAccuracyReport, onReset,
  hasProject, onNewProject, onNewFromGeotiff, onSaveProject, onSaveUserMask, onOpenSettings,
  samAvailable, executableAvailable,
}) {
  const training = status?.state === 'training'
  const trainPct = training && status.epochs ? Math.round((status.epoch / status.epochs) * 100) : 0

  return (
    <div className="toolbar">
      <div className="toolbar-group">
        <span className="group-label">File</span>
        <Dropdown label="Project" disabled={training}>
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
        <Dropdown label="Export / Model" disabled={training}>
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
          disabled={training}
          onClick={onAdoptPrediction}
          title="Fill all unlabeled pixels with the current model prediction (your painted pixels are kept); undoable"
        >
          Adopt Prediction
        </button>
        <button
          className="danger"
          disabled={training || !labeledPixels}
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
          value={epochs}
          disabled={training}
          onChange={(e) => setEpochs(Number(e.target.value))}
        />
        {training
          ? <button className="danger" onClick={onStop}>Stop ({status.epoch}/{status.epochs})</button>
          : <button className="primary" onClick={onTrain}>Train</button>}
      </div>

      <div className="toolbar-group status">
        {status?.state === 'error' && (
          <div className="stat">
            <span className="stat-label">Error</span>
            <span className="stat-value error">{status.error?.split('\n').at(-2) ?? 'see console'}</span>
          </div>
        )}
        {training && (
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
