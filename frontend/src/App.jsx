import { useEffect, useRef, useState } from 'react'
import { api, fromBase64, toBase64 } from './bridge'
import { UNLABELED } from './constants'
import AccuracyReport from './components/AccuracyReport'
import ContextMenu from './components/ContextMenu'
import InferencePane from './components/InferencePane'
import LabelerPane from './components/LabelerPane'
import ProjectSetup from './components/ProjectSetup'
import ThumbnailStrip from './components/ThumbnailStrip'
import Toolbar from './components/Toolbar'
import './App.css'

const UNDO_LIMIT = 50

export default function App() {
  const [image, setImage] = useState(null)
  const [overlay, setOverlay] = useState(null)
  const [uncertainty, setUncertainty] = useState(null)
  const [showUncertainty, setShowUncertainty] = useState(false)
  const [status, setStatus] = useState(null)
  const [tool, setTool] = useState({ classId: 0, brushSize: 8, eraser: false, mode: 'brush' })
  const [epochs, setEpochs] = useState(10)
  const [labeledPixels, setLabeledPixels] = useState(0)
  const [labelsVersion, setLabelsVersion] = useState(0) // bumped on undo
  const [undoDepth, setUndoDepth] = useState(0)
  // Operation feedback (success + errors). The toolbar toast was removed; the
  // setters stay so the plumbing is here if feedback is resurfaced later.
  const [, setMessage] = useState('')
  const [view, setView] = useState(null) // shared zoom/pan: {scale, x, y}
  const [project, setProject] = useState(null) // {root, name, images, active, classes}
  const [setup, setSetup] = useState(null) // {root, defaults} while the creation dialog is open
  const [setupError, setSetupError] = useState('')
  const [setupBusy, setSetupBusy] = useState(false)
  const [tilingProgress, setTilingProgress] = useState(null) // {done, total} while tiling
  const [thumbs, setThumbs] = useState({}) // image name -> base64 PNG | null
  const [thumbProgress, setThumbProgress] = useState(null)
  const [menu, setMenu] = useState(null) // {name, x, y} while a thumbnail menu is open
  const [menuError, setMenuError] = useState('')
  const [report, setReport] = useState(null) // {result, error, progress} while the report modal is open
  const [samAvailable, setSamAvailable] = useState(false) // Efficient-SAM2 files present
  const [executableAvailable, setExecutableAvailable] = useState(false) // prebuilt predictor present
  const labelsRef = useRef(null)
  const undoStack = useRef([]) // Map<pixelIndex, previousValue> per stroke/fill
  const autosaveTimer = useRef(null)
  const showUncertaintyRef = useRef(false)
  const prevState = useRef('idle')

  // Install a backend image payload (base image + its persisted labels) and
  // reset all per-image UI state.
  function applyImagePayload(img) {
    if (!img) {
      setImage(null)
      setOverlay(null)
      setUncertainty(null)
      labelsRef.current = null
      setLabeledPixels(0)
      return
    }
    labelsRef.current = img.labels
      ? fromBase64(img.labels)
      : new Uint8Array(img.width * img.height).fill(UNLABELED)
    setImage({ png: img.png, width: img.width, height: img.height, name: img.name })
    setLabeledPixels(img.labeled_pixels ?? 0)
    undoStack.current = []
    setUndoDepth(0)
    setLabelsVersion((v) => v + 1)
    setView(null) // let the viewport re-fit to the new image size
    setOverlay(null)
    setUncertainty(null)
    api('get_overlay').then((o) => setOverlay(o.png))
    if (showUncertaintyRef.current) api('get_uncertainty').then((u) => setUncertainty(u.png))
  }

  useEffect(() => {
    api('get_project').then((p) => { if (p) setProject(p) })
    api('get_image').then(applyImagePayload)
    api('sam_available').then((r) => setSamAvailable(!!r.available))
    api('executable_available').then((r) => setExecutableAvailable(!!r.available))
  }, [])

  // Poll thumbnail generation while it runs; thumbnails appear as they finish.
  useEffect(() => {
    if (!project) return
    let cancelled = false
    async function poll() {
      for (;;) {
        const t = await api('get_thumbnails')
        if (cancelled) return
        setThumbs(t.thumbs)
        setThumbProgress({ running: t.running, done: t.done, total: t.total })
        if (!t.running) return
        await new Promise((r) => setTimeout(r, 800))
      }
    }
    poll()
    return () => { cancelled = true }
  }, [project])

  async function refreshOverlays() {
    const o = await api('get_overlay')
    setOverlay(o.png)
    if (showUncertaintyRef.current) {
      const u = await api('get_uncertainty')
      setUncertainty(u.png)
    } else {
      setUncertainty(null) // stale once the model changed; refetch on demand
    }
  }

  // Poll training status; refresh the inference overlays when a run finishes.
  useEffect(() => {
    const timer = setInterval(async () => {
      const s = await api('get_status')
      setStatus(s)
      if (prevState.current === 'training' && s.state !== 'training') {
        await refreshOverlays()
      }
      prevState.current = s.state
    }, 500)
    return () => clearInterval(timer)
  }, [])

  function toggleUncertainty(on) {
    setShowUncertainty(on)
    showUncertaintyRef.current = on
    if (on) api('get_uncertainty').then((u) => setUncertainty(u.png))
  }

  async function pushLabels() {
    const res = await api('set_labels', toBase64(labelsRef.current))
    setLabeledPixels(res.labeled_pixels)
    scheduleAutosave()
  }

  // Debounced write-to-disk of the current image's mask, so user edits persist
  // automatically without a full GeoTIFF write on every stroke. Backend also
  // autosaves on image switch, so a pending timer can never lose work.
  function scheduleAutosave() {
    if (autosaveTimer.current) clearTimeout(autosaveTimer.current)
    autosaveTimer.current = setTimeout(() => {
      autosaveTimer.current = null
      api('autosave_user_mask')
    }, 1000)
  }

  function recordUndo(diff) {
    undoStack.current.push(diff)
    if (undoStack.current.length > UNDO_LIMIT) undoStack.current.shift()
    setUndoDepth(undoStack.current.length)
  }

  async function handleUndo() {
    const diff = undoStack.current.pop()
    if (!diff) return
    setUndoDepth(undoStack.current.length)
    for (const [i, v] of diff) labelsRef.current[i] = v
    setLabelsVersion((v) => v + 1)
    await pushLabels()
  }

  useEffect(() => {
    function onKey(e) {
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault()
        handleUndo()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  // Fill unlabeled pixels with the model prediction (painted pixels stay),
  // as one undoable step.
  async function handleAdoptPrediction() {
    setMessage('')
    const res = await api('transfer_prediction')
    if (!res.ok) {
      if (res.error) setMessage(res.error)
      return
    }
    const next = fromBase64(res.labels)
    const prev = labelsRef.current
    const diff = new Map()
    for (let i = 0; i < next.length; i++) {
      if (prev[i] !== next[i]) diff.set(i, prev[i])
    }
    if (diff.size) recordUndo(diff)
    labelsRef.current = next
    setLabeledPixels(res.labeled_pixels)
    setLabelsVersion((v) => v + 1)
    if (diff.size) scheduleAutosave()
    setMessage(`Prediction adopted (${diff.size.toLocaleString()} px filled)`)
  }

  // SAM2 snap: segment from the pending include/exclude clicks and return
  // polygon vertices (image coords) to preview; null on error/no object.
  async function handleSamSnap(points) {
    setMessage('')
    const res = await api('sam_snap', points.map((p) => [p.x, p.y, p.label]))
    if (!res.ok) {
      if (res.error) setMessage(res.error)
      return null
    }
    return res.polygons
  }

  // Reset every labeled pixel of the current image to unlabeled, undoably.
  async function handleClearLabels() {
    const labels = labelsRef.current
    if (!labels) return
    const diff = new Map()
    for (let i = 0; i < labels.length; i++) {
      if (labels[i] !== UNLABELED) {
        diff.set(i, labels[i])
        labels[i] = UNLABELED
      }
    }
    if (!diff.size) return
    recordUndo(diff)
    setLabelsVersion((v) => v + 1)
    setMessage(`Labels cleared (${diff.size.toLocaleString()} px)`)
    await pushLabels()
  }

  async function handleTrain() {
    await pushLabels()
    const res = await api('start_training', epochs)
    if (!res.ok) setMessage(res.error)
    else setMessage(`Training on ${res.images} labeled image${res.images === 1 ? '' : 's'}`)
  }

  async function handleExport(method) {
    setMessage('')
    const res = await api(method)
    if (res.ok) setMessage(`Saved: ${res.path}`)
    else if (res.error) setMessage(res.error)
  }

  // Move one image between the training and validation sets. The backend
  // returns the refreshed project, so the strip's badges follow immediately.
  async function handleSetRole(name, role) {
    const res = await api('set_image_role', name, role)
    if (res.ok) setProject(res.project)
    else if (res.error) setMenuError(res.error)
  }

  // Score the model over every labeled image. Runs on a backend worker (it can
  // take minutes), so the modal opens first and polls like the tiling dialog.
  async function handleAccuracyReport() {
    await pushLabels()
    setReport({ result: null, error: null, progress: { done: 0, total: 0 } })
    const started = await api('generate_accuracy_report')
    if (!started.ok) {
      setReport({ result: null, error: started.error || 'Could not start the report' })
      return
    }
    for (;;) {
      const p = await api('accuracy_report_progress')
      if (!p.running) {
        setReport({ result: p.result, error: p.error, progress: null })
        return
      }
      setReport({ result: null, error: null, progress: { done: p.done, total: p.total } })
      await new Promise((r) => setTimeout(r, 300))
    }
  }

  async function handleReset() {
    setMessage('')
    const res = await api('reset_model')
    if (!res.ok) {
      if (res.error) setMessage(res.error)
      return
    }
    setMessage('Model reset to pretrained weights')
    await refreshOverlays()
  }

  function installProject(res) {
    setProject(res.project)
    setThumbs({})
    applyImagePayload(res.image)
    if (res.image_error) setMessage(res.image_error)
    else if (!res.project.images.length) setMessage('Project opened — no images found in folder')
    else setMessage(res.created ? `Project created: ${res.project.name}` : `Project opened: ${res.project.name}`)
  }

  async function handleNewProject() {
    setMessage('')
    const res = await api('new_project')
    if (!res.ok) {
      if (res.error) setMessage(res.error)
      return
    }
    if (res.needs_setup) {
      // No config in that folder yet: collect the data profile first.
      setSetupError('')
      setSetup({ mode: 'create', root: res.root, defaults: res.defaults })
      return
    }
    installProject(res)
  }

  // Build a project by tiling one big GeoTIFF: same setup dialog, plus the
  // tile geometry. Nothing is written until the dialog is confirmed.
  async function handleNewFromGeotiff() {
    setMessage('')
    const res = await api('new_project_from_geotiff')
    if (!res.ok) {
      if (res.error) setMessage(res.error)
      return
    }
    setSetupError('')
    setSetup({ mode: 'geotiff', source: res.source, defaults: res.defaults })
  }

  // Open the settings dialog prefilled with the current project's config.
  function handleOpenSettings() {
    if (!project) return
    const dp = project.data_profile
    setSetupError('')
    setSetup({
      mode: 'edit',
      root: project.root,
      defaults: {
        project_name: project.name,
        input_channels: dp.input_channels,
        input_patch_size: dp.input_patch_size[0],
        band_names: dp.band_names ?? [],
        display_bands: dp.display_bands,
        use_pointrend: dp.use_pointrend ?? false,
        validation_ratio: project.validation_ratio ?? 0.2,
        images_folder: project.paths.images_folder,
        masks_user: project.paths.masks_user,
        masks_ai: project.paths.masks_ai,
        bands_detected: true,
      },
    })
  }

  async function handleSubmitSetup(settings) {
    setSetupBusy(true)
    let res
    if (setup.mode === 'edit') {
      res = await api('update_settings', settings)
    } else if (setup.mode === 'geotiff') {
      // Tiling runs on a backend worker; poll it so the dialog can show a bar.
      const started = await api('create_project_from_geotiff', setup.source, settings)
      if (!started.ok) {
        setSetupBusy(false)
        setSetupError(started.error || 'Could not create the project')
        return
      }
      setTilingProgress({ done: 0, total: 0 })
      for (;;) {
        const p = await api('tiling_progress')
        setTilingProgress({ done: p.done, total: p.total })
        if (!p.running) {
          res = p.error ? { ok: false, error: p.error } : p.result
          break
        }
        await new Promise((r) => setTimeout(r, 300))
      }
      setTilingProgress(null)
    } else {
      res = await api('create_project', setup.root, settings)
    }
    setSetupBusy(false)
    if (!res.ok) {
      setSetupError(res.error || (setup.mode === 'edit' ? 'Could not save settings' : 'Could not create the project'))
      return
    }
    setSetup(null)
    if (setup.mode === 'edit') {
      if (res.reopened) {
        // A folder change reopens the project: reload images/labels/thumbnails.
        installProject(res)
      } else {
        setProject(res.project)
        // Only the display bands changed: swap the base RGB, keep labels/view.
        if (res.image) setImage((img) => (img ? { ...img, png: res.image.png } : img))
        setMessage('Settings saved')
      }
    } else {
      installProject(res)
    }
  }

  async function handleSelectImage(name) {
    if (name === image?.name) return
    setMessage('')
    const res = await api('load_image', name)
    if (!res.ok) {
      if (res.error) setMessage(res.error)
      return
    }
    applyImagePayload(res.image)
    setProject((p) => (p ? { ...p, active: name } : p))
  }

  return (
    <div className="app">
      <Toolbar
        tool={tool}
        setTool={setTool}
        epochs={epochs}
        setEpochs={setEpochs}
        status={status}
        labeledPixels={labeledPixels}
        undoDepth={undoDepth}
        onTrain={handleTrain}
        onStop={() => api('stop_training')}
        onUndo={handleUndo}
        onAdoptPrediction={handleAdoptPrediction}
        onClearLabels={handleClearLabels}
        onExportModel={() => handleExport('export_model')}
        onExportOnnx={() => handleExport('export_onnx')}
        onExportExecutable={() => handleExport('export_executable')}
        onExportMask={() => handleExport('export_mask')}
        onAccuracyReport={handleAccuracyReport}
        onReset={handleReset}
        hasProject={!!project}
        onNewProject={handleNewProject}
        onNewFromGeotiff={handleNewFromGeotiff}
        onSaveProject={() => handleExport('save_project')}
        onSaveUserMask={async () => { await pushLabels(); await handleExport('save_user_mask') }}
        onOpenSettings={handleOpenSettings}
        samAvailable={samAvailable}
        executableAvailable={executableAvailable}
      />
      <div className="panes">
        <InferencePane
          image={image}
          overlay={overlay}
          uncertainty={uncertainty}
          showUncertainty={showUncertainty}
          onToggleUncertainty={toggleUncertainty}
          training={status?.state === 'training'}
          view={view}
          setView={setView}
        />
        <LabelerPane
          image={image}
          labels={labelsRef.current}
          labelsVersion={labelsVersion}
          tool={tool}
          view={view}
          setView={setView}
          onStrokeEnd={pushLabels}
          onDiff={recordUndo}
          onSamSnap={handleSamSnap}
        />
      </div>
      <ThumbnailStrip
        images={project?.images}
        thumbs={thumbs}
        activeName={image?.name}
        validation={project?.validation}
        progress={thumbProgress}
        disabled={status?.state === 'training'}
        onSelect={handleSelectImage}
        onContextMenu={(name, x, y) => {
          setMenuError('')
          setMenu({ name, x, y })
        }}
      />
      {menu && (
        <ContextMenu x={menu.x} y={menu.y} onClose={() => setMenu(null)}>
          <span className="context-title">{menu.name}</span>
          {project?.validation?.includes(menu.name)
            ? <button onClick={() => handleSetRole(menu.name, 'training')}>
                Move to training set
              </button>
            : <button onClick={() => handleSetRole(menu.name, 'validation')}>
                Set image to validation
              </button>}
        </ContextMenu>
      )}
      {menuError && <div className="floating-error">{menuError}</div>}
      {report && (
        <AccuracyReport
          report={report.result}
          progress={report.progress}
          error={report.error}
          onSave={() => handleExport('save_accuracy_report')}
          onCancel={() => api('cancel_accuracy_report')}
          onClose={() => setReport(null)}
        />
      )}
      {setup && (
        <ProjectSetup
          mode={setup.mode}
          root={setup.root}
          source={setup.source}
          defaults={setup.defaults}
          error={setupError}
          busy={setupBusy}
          progress={tilingProgress}
          onSubmit={handleSubmitSetup}
          onCancel={() => setSetup(null)}
        />
      )}
    </div>
  )
}
