import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  LABEL_ALPHA, UNLABELED, WAND_EDGE_ALPHA, WAND_EDGE_DARK, WAND_EDGE_LIGHT,
  WAND_PREVIEW_ALPHA, WAND_SAM_TOL_MIN, WAND_SAM_TOL_SPAN, WAND_TOL_SCALE,
  WAND_TOLERANCE_DEFAULT, WAND_WARN_FRACTION, wandBudget,
} from '../constants'
import Viewport from './Viewport'

// Chromium (and Qt WebEngine) ignores custom CSS cursors larger than 128px, so
// the native brush cursor is only used up to this on-screen diameter; beyond it
// the DOM circle takes over.
const MAX_CURSOR_PX = 128
const CURSOR_PAD = 2 // room around the circle for its 1.5px stroke
const CROSS_ARM = 8 // half-length of each center crosshair arm, in px

// SAM2 click markers, matching the app's --accent / --danger (see App.css).
// Deliberately independent of CLASS_COLORS: these mark prompt points, not
// labeled pixels, and must stay readable whichever class is being painted.
const SAM_INCLUDE_COLOR = [242, 111, 181] // pink
const SAM_EXCLUDE_COLOR = [229, 72, 77]   // red

// Stroke the center crosshair ("+") marking the exact painted pixel. Uses the
// current strokeStyle/lineWidth so it matches whatever it's drawn over.
function strokeCrosshair(ctx, cx, cy) {
  ctx.beginPath()
  ctx.moveTo(cx - CROSS_ARM, cy)
  ctx.lineTo(cx + CROSS_ARM, cy)
  ctx.moveTo(cx, cy - CROSS_ARM)
  ctx.lineTo(cx, cy + CROSS_ARM)
  ctx.stroke()
}

// Turn an offscreen canvas into a CSS cursor value centered on the pointer.
function canvasCursor(canvas) {
  const hot = Math.round(canvas.width / 2)
  return `url(${canvas.toDataURL()}) ${hot} ${hot}, crosshair`
}

// Paint the brush preview into an offscreen canvas and return it as a cursor
// value: `url(<png>) <hotspot> <hotspot>, crosshair`, centered on the pointer.
// Returns null when the image would exceed the browser's cursor size cap.
function makeBrushCursor(diameter, color) {
  const size = Math.ceil(Math.max(diameter, CROSS_ARM * 2)) + CURSOR_PAD * 2
  if (size > MAX_CURSOR_PX) return null
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  const c = size / 2
  ctx.beginPath()
  ctx.arc(c, c, diameter / 2, 0, Math.PI * 2)
  ctx.fillStyle = `rgba(${color.join(',')}, 0.2)`
  ctx.fill()
  ctx.lineWidth = 1.5
  ctx.strokeStyle = `rgba(${color.join(',')}, 0.9)`
  ctx.stroke()
  strokeCrosshair(ctx, c, c)
  return canvasCursor(canvas)
}

// Crosshair-only cursor for oversized brushes, where the big circle is drawn by
// the DOM `.brush-cursor` div. Rendered with the exact same canvas code as the
// native brush cursor's crosshair, so the two are pixel-identical.
function makeCrosshairCursor(color) {
  const size = CROSS_ARM * 2 + CURSOR_PAD * 2
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const ctx = canvas.getContext('2d')
  const c = size / 2
  ctx.lineWidth = 1.5
  ctx.strokeStyle = `rgba(${color.join(',')}, 0.9)`
  strokeCrosshair(ctx, c, c)
  return canvasCursor(canvas)
}

// One labelled wand slider with its readout. All three respond to the scroll
// wheel, like the overlay-opacity control above them.
function WandSlider({ label, value, min, max, readout, onChange }) {
  const step = (delta) => onChange(Math.min(max, Math.max(min, value + delta)))
  return (
    <label className="wand-slider">
      <span className="wand-slider-name">{label}</span>
      <input
        type="range"
        min={min}
        max={max}
        step="1"
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        onWheel={(e) => {
          e.preventDefault()
          step(e.deltaY < 0 ? 1 : -1)
        }}
      />
      <span className="wand-slider-value">{readout}</span>
    </label>
  )
}

// Right pane: base RGB image with a brush-paintable label canvas on top.
// Labels live in a Uint8Array (values 0/1/UNLABELED); the canvas is a
// colorized rendering of that array, redrawn per stroke segment.
// Left button paints (brush mode) or places polygon vertices (polygon mode);
// middle-drag pans (handled by the surrounding Viewport).
// Every stroke/fill reports a Map<pixelIndex, previousValue> via onDiff so
// the app can undo it.
export default function LabelerPane({
  image, labels, labelsVersion, tool, view, setView, onStrokeEnd, onDiff, onSamSnap,
  onWandSelect, classColors, samAvailable,
}) {
  const canvasRef = useRef(null)
  const [canvasReady, setCanvasReady] = useState(false)
  const cursorRef = useRef(null)
  const drawing = useRef(false)
  const lastPos = useRef(null)
  const strokeDiff = useRef(null)
  const [opacity, setOpacity] = useState(1)
  const [samBusy, setSamBusy] = useState(false) // a SAM2 snap request is in flight
  const [verts, setVerts] = useState([]) // polygon vertices in image coords
  const vertsRef = useRef(verts)
  vertsRef.current = verts
  // Pending SAM2 selection: the clicks so far ({x, y, label}, label 1 =
  // include / 0 = exclude) and the mask they currently produce, previewed as
  // polygons. Nothing touches the labels until the user commits, which is what
  // makes an exclude click able to *shrink* the selection — the mask is
  // re-predicted from the whole click list rather than painted per click.
  const [samPoints, setSamPoints] = useState([])
  const [samPolys, setSamPolys] = useState([])
  const samPointsRef = useRef(samPoints)
  samPointsRef.current = samPoints
  const samPolysRef = useRef(samPolys)
  samPolysRef.current = samPolys
  const samSeq = useRef(0) // drops out-of-order responses from stale clicks
  const settings = useRef(tool)
  settings.current = tool
  const closeRef = useRef(() => {}) // fresh closePolygon for the window key listener
  const samCommitRef = useRef(() => {}) // ditto for the SAM2 selection
  const samUndoRef = useRef(() => {})
  const samCancelRef = useRef(() => {})

  // ---------- magic wand ----------
  // Pending selection: the clicks so far ({x, y, label}, 1 = include /
  // 0 = exclude) and the mask they currently produce, previewed on its own
  // canvas. Nothing touches the labels until Enter.
  const wandCanvasRef = useRef(null)
  const [wandSamples, setWandSamples] = useState([])
  const wandSamplesRef = useRef(wandSamples)
  wandSamplesRef.current = wandSamples
  const wandMaskRef = useRef(null)
  // Only the tolerances the user has actually moved; anything untouched falls
  // through to that class's measured seed, so a fresh class starts on evidence
  // rather than on whatever the last class happened to need.
  const [wandTolerances, setWandTolerances] = useState({})
  const [wandSam, setWandSam] = useState(false)
  const wandSamRef = useRef(wandSam)
  wandSamRef.current = wandSam
  const [wandSamTolerance, setWandSamTolerance] = useState(50)
  const [wandLevel, setWandLevel] = useState('fine')
  const [wandBudgetStep, setWandBudgetStep] = useState(100)
  const [wandSampleSize, setWandSampleSize] = useState(9)
  const [wandProtect, setWandProtect] = useState(true)
  const [wandGlobal, setWandGlobal] = useState(false)
  const [wandStats, setWandStats] = useState(null) // {count, available, ...}
  const [wandBusy, setWandBusy] = useState(false)
  const wandSeq = useRef(0)
  const wandCommitRef = useRef(() => {})
  const wandUndoRef = useRef(() => {})
  const wandCancelRef = useRef(() => {})

  // Kept per class so Target and Background can remember different settings,
  // but both start from the same measured default.
  const wandTolerance = wandTolerances[tool.classId] ?? WAND_TOLERANCE_DEFAULT
  const wandRunaway = !!wandStats
    && wandStats.count > WAND_WARN_FRACTION * Math.max(wandStats.valid, 1)

  // Callback ref: the canvas only mounts once the Viewport has a `view`
  // (see Viewport), which lands after image/labels are already set. Track the
  // node's presence so the redraw effect can paint the persisted mask the
  // moment the canvas appears, rather than waiting for the first stroke.
  const setCanvas = useCallback((node) => {
    canvasRef.current = node
    setCanvasReady(!!node)
  }, [])

  useEffect(() => {
    if (image && canvasReady) redraw()
  }, [image, labels, labelsVersion, canvasReady, opacity, classColors])

  // Leaving polygon mode discards an unfinished polygon.
  useEffect(() => {
    if (tool.mode !== 'polygon') setVerts([])
    // Leaving SAM2 mode drops an uncommitted selection rather than leaving an
    // invisible one to be committed later by a stray Enter.
    if (tool.mode !== 'sam') samCancelRef.current()
    if (tool.mode !== 'wand') wandCancelRef.current()
  }, [tool.mode])

  // Switching images drops the pending selection. The canvas resets itself when
  // its width/height change, but the mask behind it would not: a stray Enter
  // could otherwise paint the previous image's selection onto this one.
  useEffect(() => {
    wandCancelRef.current()
  }, [image?.name])

  // Any change to what the selection MEANS re-runs the pending clicks rather
  // than clearing them. For the class and the source that is a correctness
  // requirement: the tolerance changes with both, so the preview would
  // otherwise be the one the old threshold produced while the slider reads the
  // new number, and Enter would commit a mask matching neither. For the level
  // it is the point of the selector — the three are only comparable on the
  // same click.
  //
  // Runs as an effect, not from the change handlers, so wandRun always closes
  // over the state as it is AFTER the change. Dragging a slider is cheap: the
  // backend caches its distance fields per click, so a re-run at a new
  // tolerance is a threshold pass, not a re-measure.
  useEffect(() => {
    if (tool.mode !== 'wand') return
    if (wandSamplesRef.current.some((p) => p.label)) wandRun(wandSamplesRef.current)
  }, [tool.classId, tool.eraser, wandTolerance, wandSam, wandSamTolerance, wandLevel,
      wandBudgetStep, wandSampleSize, wandProtect, wandGlobal])

  // Enter commits the pending shape (polygon fill / SAM2 / wand selection),
  // Escape cancels it; Backspace drops just the last click.
  useEffect(() => {
    function onKey(e) {
      const mode = settings.current.mode
      if (mode === 'polygon') {
        if (e.key === 'Enter') closeRef.current()
        else if (e.key === 'Escape') setVerts([])
      } else if (mode === 'sam' || mode === 'wand') {
        const wand = mode === 'wand'
        if (e.key === 'Enter') (wand ? wandCommitRef : samCommitRef).current()
        else if (e.key === 'Escape') (wand ? wandCancelRef : samCancelRef).current()
        else if (e.key === 'Backspace') {
          e.preventDefault() // otherwise the webview treats it as "go back"
          ;(wand ? wandUndoRef : samUndoRef).current()
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  function redraw() {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const { width, height } = image
    const imgData = ctx.createImageData(width, height)
    for (let i = 0; i < labels.length; i++) {
      const v = labels[i]
      if (!(v in classColors)) continue
      const [r, g, b] = classColors[v]
      const o = i * 4
      imgData.data[o] = r
      imgData.data[o + 1] = g
      imgData.data[o + 2] = b
      // Bake the overlay opacity into the pixels instead of using CSS opacity
      // on the canvas element: a sub-1 CSS opacity promotes the canvas to its
      // own compositing layer, which snaps to device pixels independently of
      // the base <img> and makes the labels jitter against it while zooming.
      imgData.data[o + 3] = Math.round(LABEL_ALPHA * opacity)
    }
    ctx.clearRect(0, 0, width, height)
    ctx.putImageData(imgData, 0, 0)
  }

  function canvasPos(e) {
    const rect = canvasRef.current.getBoundingClientRect()
    return {
      x: ((e.clientX - rect.left) / rect.width) * image.width,
      y: ((e.clientY - rect.top) / rect.height) * image.height,
    }
  }

  function paintPixel(diff, idx, value) {
    if (labels[idx] === value) return
    if (!diff.has(idx)) diff.set(idx, labels[idx])
    labels[idx] = value
  }

  function stamp(x, y) {
    const { classId, brushSize, eraser } = settings.current
    const value = eraser ? UNLABELED : classId
    const r = brushSize
    const { width, height } = image
    const x0 = Math.max(0, Math.floor(x - r))
    const x1 = Math.min(width - 1, Math.ceil(x + r))
    const y0 = Math.max(0, Math.floor(y - r))
    const y1 = Math.min(height - 1, Math.ceil(y + r))
    for (let py = y0; py <= y1; py++) {
      for (let px = x0; px <= x1; px++) {
        if ((px - x) ** 2 + (py - y) ** 2 <= r * r) {
          paintPixel(strokeDiff.current, py * width + px, value)
        }
      }
    }
  }

  function strokeTo(pos) {
    const last = lastPos.current ?? pos
    const dist = Math.hypot(pos.x - last.x, pos.y - last.y)
    const steps = Math.max(1, Math.ceil(dist / (settings.current.brushSize / 2)))
    for (let i = 1; i <= steps; i++) {
      stamp(last.x + ((pos.x - last.x) * i) / steps, last.y + ((pos.y - last.y) * i) / steps)
    }
    lastPos.current = pos
    redraw()
  }

  // Even-odd scanline fill of one closed polygon (image coords) into `diff`.
  function scanFillPolygon(poly, value, diff) {
    const { width, height } = image
    const ys = poly.map((p) => p.y)
    const yStart = Math.max(0, Math.floor(Math.min(...ys)))
    const yEnd = Math.min(height - 1, Math.ceil(Math.max(...ys)))
    for (let py = yStart; py <= yEnd; py++) {
      const yc = py + 0.5
      const xs = []
      for (let i = 0; i < poly.length; i++) {
        const a = poly[i]
        const b = poly[(i + 1) % poly.length]
        if (a.y <= yc !== b.y <= yc) {
          xs.push(a.x + ((yc - a.y) / (b.y - a.y)) * (b.x - a.x))
        }
      }
      xs.sort((p, q) => p - q)
      for (let k = 0; k + 1 < xs.length; k += 2) {
        const xa = Math.max(0, Math.round(xs[k]))
        const xb = Math.min(width - 1, Math.round(xs[k + 1]) - 1)
        for (let px = xa; px <= xb; px++) {
          paintPixel(diff, py * width + px, value)
        }
      }
    }
  }

  // Fill one or more closed polygons as a single undoable edit, using the
  // current class (or the eraser). Shared by the polygon tool and SAM2 snap.
  function commitPolygons(polys) {
    const { classId, eraser } = settings.current
    const value = eraser ? UNLABELED : classId
    const diff = new Map()
    for (const poly of polys) {
      if (poly.length >= 3) scanFillPolygon(poly, value, diff)
    }
    if (diff.size) {
      onDiff(diff)
      redraw()
      onStrokeEnd()
    }
  }

  function closePolygon() {
    const poly = vertsRef.current
    setVerts([])
    commitPolygons([poly])
  }
  closeRef.current = closePolygon

  // Re-run SAM2 over the whole pending click list and show the mask it returns.
  // Every click re-predicts from scratch (the backend replays the click chain),
  // so removing or adding a point always yields the mask that click set implies.
  async function samPredict(points) {
    if (!points.length) {
      setSamPolys([])
      return
    }
    const seq = ++samSeq.current
    setSamBusy(true)
    try {
      const polygons = await onSamSnap(points) // [[x, y], ...][] image coords, or null
      if (seq !== samSeq.current) return // a newer click already superseded this
      if (polygons?.length) {
        setSamPolys(polygons.map((poly) => poly.map(([x, y]) => ({ x, y }))))
      }
      // On a failed/empty prediction keep the previous preview rather than
      // blanking it, so a stray click does not throw away a good selection.
    } finally {
      if (seq === samSeq.current) setSamBusy(false)
    }
  }

  // Add an include (label 1) or exclude (label 0) click to the pending selection.
  function samAddPoint(pos, label) {
    const next = [...samPointsRef.current, { ...pos, label }]
    samPointsRef.current = next
    setSamPoints(next)
    samPredict(next)
  }

  // Drop the most recent click (Backspace) — the mask reverts to what the
  // remaining clicks imply, so a mis-click costs one keystroke, not the selection.
  function samUndoPoint() {
    const next = samPointsRef.current.slice(0, -1)
    samPointsRef.current = next
    setSamPoints(next)
    if (!next.length) setSamPolys([])
    samPredict(next)
  }

  function samCancel() {
    samSeq.current++ // abandon any in-flight prediction
    samPointsRef.current = []
    setSamPoints([])
    setSamPolys([])
    setSamBusy(false)
  }
  samUndoRef.current = samUndoPoint
  samCancelRef.current = samCancel

  // Commit the previewed mask into the labels as a normal polygon fill, so it
  // persists to masks_user and feeds training exactly like a hand-drawn one.
  function samCommit() {
    const polys = samPolysRef.current
    if (!polys.length) return
    samCancel()
    commitPolygons(polys)
  }
  samCommitRef.current = samCommit

  // Re-run the wand over the whole pending click list. Every click recomputes
  // from scratch: the backend resolves each sample independently and unions the
  // results, so the mask a click set implies never depends on the order they
  // arrived in.
  async function wandRun(samples) {
    if (!samples.length) {
      wandMaskRef.current = null
      setWandStats(null)
      drawWandPreview(null)
      return
    }
    const seq = ++wandSeq.current
    setWandBusy(true)
    const sam = wandSamRef.current
    try {
      const res = await onWandSelect(samples.filter((p) => p.label), {
        tolerance: sam
          ? WAND_SAM_TOL_MIN + (wandSamTolerance / 100) * WAND_SAM_TOL_SPAN
          : wandTolerance / WAND_TOL_SCALE,
        source: sam ? 'sam' : 'image',
        level: wandLevel,
        max_pixels: wandBudget(wandBudgetStep),
        sample_size: wandSampleSize,
        negatives: samples.filter((p) => !p.label).map((p) => [p.x, p.y]),
        global: wandGlobal,
        // The eraser only ever acts on labeled pixels, so protecting them would
        // make it a no-op. The caller decides, not the backend.
        protect: wandProtect && !settings.current.eraser,
      })
      if (seq !== wandSeq.current) return // a newer click already superseded this
      if (res) {
        wandMaskRef.current = res.mask
        setWandStats(res)
        drawWandPreview(res.mask)
      }
      // A failed request keeps the previous preview rather than blanking it, so
      // a stray click does not throw away a good selection.
    } finally {
      if (seq === wandSeq.current) setWandBusy(false)
    }
  }

  // A positive click needs at least one include sample to grow from; a lone
  // negative has nothing to subtract from, so it just waits for one.
  function wandAddSample(pos, label) {
    const next = [...wandSamplesRef.current, { x: Math.floor(pos.x), y: Math.floor(pos.y), label }]
    wandSamplesRef.current = next
    setWandSamples(next)
    if (next.some((p) => p.label)) wandRun(next)
  }

  function wandUndoSample() {
    const next = wandSamplesRef.current.slice(0, -1)
    wandSamplesRef.current = next
    setWandSamples(next)
    wandRun(next.some((p) => p.label) ? next : [])
  }

  function wandCancel() {
    wandSeq.current++ // abandon any in-flight request
    wandSamplesRef.current = []
    setWandSamples([])
    wandMaskRef.current = null
    setWandStats(null)
    setWandBusy(false)
    drawWandPreview(null)
  }
  wandUndoRef.current = wandUndoSample
  wandCancelRef.current = wandCancel

  // Paint the selection into the label buffer as ONE undoable diff.
  function wandCommit() {
    const mask = wandMaskRef.current
    if (!mask) return
    const { classId, eraser } = settings.current
    const value = eraser ? UNLABELED : classId
    const diff = new Map()
    for (let i = 0; i < mask.length; i++) {
      if (mask[i]) paintPixel(diff, i, value)
    }
    wandCancel()
    if (diff.size) {
      onDiff(diff)
      redraw()
      onStrokeEnd()
    }
  }
  wandCommitRef.current = wandCommit

  // Fill in the class colour, then a two-tone boundary on top of it. The fill
  // alone is not enough: it is the class colour, so Snow over bright terrain or
  // Meadow over a yellow-green field is nearly invisible. One edge tone is not
  // enough either — a light outline vanishes on Snow, a dark one vanishes in
  // shadow — so selected edge pixels get the light tone and the unselected
  // pixels touching them get the dark one, and the pair reads on both.
  function drawWandPreview(mask) {
    const canvas = wandCanvasRef.current
    if (!canvas || !image) return
    const { width, height } = image
    const ctx = canvas.getContext('2d')
    ctx.clearRect(0, 0, width, height)
    if (!mask) return
    const img = ctx.createImageData(width, height)
    const data = img.data
    const { classId, eraser } = settings.current
    const [r, g, b] = eraser ? [255, 255, 255] : (classColors[classId] ?? [255, 255, 255])
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x
        const o = i * 4
        const on = mask[i]
        // Image-border pixels count as edge, so a selection running off the
        // edge still reads as bounded rather than as trailing away.
        const up = y > 0 ? mask[i - width] : !on
        const down = y < height - 1 ? mask[i + width] : !on
        const left = x > 0 ? mask[i - 1] : !on
        const right = x < width - 1 ? mask[i + 1] : !on
        if (on) {
          const edge = !up || !down || !left || !right
          if (edge) {
            data[o] = WAND_EDGE_LIGHT
            data[o + 1] = WAND_EDGE_LIGHT
            data[o + 2] = WAND_EDGE_LIGHT
            data[o + 3] = WAND_EDGE_ALPHA
          } else {
            data[o] = r
            data[o + 1] = g
            data[o + 2] = b
            data[o + 3] = WAND_PREVIEW_ALPHA
          }
        } else if (up || down || left || right) {
          data[o] = WAND_EDGE_DARK
          data[o + 1] = WAND_EDGE_DARK
          data[o + 2] = WAND_EDGE_DARK
          data[o + 3] = WAND_EDGE_ALPHA
        }
      }
    }
    ctx.putImageData(img, 0, 0)
  }

  // Oversized-brush fallback circle (see cursorCss): only mounted when the
  // brush is too big for a native CSS cursor. The stack's CSS space is the
  // image scaled by the zoom (layout scaling, see Viewport), so image
  // coordinates map to CSS pixels via view.scale. Positioned with `transform`
  // (compositor-only, no layout reflow) via direct style updates, avoiding a
  // React re-render per pointer move.
  function moveCursor(pos) {
    const el = cursorRef.current
    if (!el) return // native-cursor mode: no DOM circle mounted
    const scale = view?.scale ?? 1
    el.style.display = 'block'
    el.style.transform = `translate3d(${pos.x * scale}px, ${pos.y * scale}px, 0)`
  }

  function hideCursor() {
    if (cursorRef.current) cursorRef.current.style.display = 'none'
  }

  function onPointerDown(e) {
    if (!image) return
    if (settings.current.mode === 'wand') {
      // Same gesture language as SAM2 mode: left includes, right (or
      // Ctrl/Alt+left, for trackpads that swallow secondary click) excludes.
      if (e.button === 2) {
        e.preventDefault()
        wandAddSample(canvasPos(e), 0)
      } else if (e.button === 0) {
        wandAddSample(canvasPos(e), e.ctrlKey || e.altKey ? 0 : 1)
      }
      return
    }
    if (settings.current.mode === 'sam') {
      // Left click includes, right click (or Ctrl/Alt+left) excludes. Right
      // click is the primary gesture because it needs no second hand; the
      // modifier is there for trackpads that swallow secondary click.
      if (e.button === 2) {
        e.preventDefault()
        samAddPoint(canvasPos(e), 0)
      } else if (e.button === 0) {
        samAddPoint(canvasPos(e), e.ctrlKey || e.altKey ? 0 : 1)
      }
      return
    }
    if (e.button !== 0) return
    if (settings.current.mode === 'polygon') {
      const pos = canvasPos(e)
      setVerts((v) => {
        const next = [...v, pos]
        vertsRef.current = next // keep the ref fresh for same-tick key events
        return next
      })
      return
    }
    e.currentTarget.setPointerCapture(e.pointerId)
    drawing.current = true
    lastPos.current = null
    strokeDiff.current = new Map()
    strokeTo(canvasPos(e))
  }

  function onPointerMove(e) {
    const pos = canvasPos(e)
    moveCursor(pos)
    if (drawing.current) strokeTo(pos)
  }

  function onPointerUp() {
    if (!drawing.current) return
    drawing.current = false
    lastPos.current = null
    if (strokeDiff.current?.size) onDiff(strokeDiff.current)
    strokeDiff.current = null
    onStrokeEnd()
  }

  const brushColor = tool.eraser ? [255, 255, 255] : classColors[tool.classId]
  const polygonMode = tool.mode === 'polygon'
  const samMode = tool.mode === 'sam'
  const wandMode = tool.mode === 'wand'

  // Brush cursor: draw it into a native CSS cursor (zero pointer lag) while it
  // fits the browser's 128px cap; beyond that, fall back to the DOM circle.
  const scale = view?.scale ?? 1
  const oversized = Math.ceil(tool.brushSize * 2 * scale) + CURSOR_PAD * 2 > MAX_CURSOR_PX
  const cursorCss = useMemo(() => {
    if (wandMode) return wandBusy ? 'wait' : 'crosshair'
    if (samMode) return samBusy ? 'wait' : 'crosshair'
    if (polygonMode) return 'crosshair'
    const color = tool.eraser ? [255, 255, 255] : classColors[tool.classId]
    return makeBrushCursor(tool.brushSize * 2 * scale, color) ?? makeCrosshairCursor(color)
  }, [tool.brushSize, scale, tool.eraser, tool.classId, polygonMode, samMode, samBusy,
      wandMode, wandBusy, classColors])

  return (
    <div className="pane">
      <div className="pane-title">
        <span className="pane-name">Interactive Labeler</span>
        {/* No class/eraser chip here: the toolbar already shows what is
            selected, so repeating it beside the title is noise. */}
        {polygonMode && (
          <span className="hint">click: add point · Enter/double-click: fill · Esc: cancel</span>
        )}
        {samMode && (
          <span className="hint">
            {samBusy ? 'SAM2 working…'
              : samPoints.length
                ? 'right-click: exclude · left-click: include · Backspace: undo point · Enter: fill · Esc: cancel'
                : 'click an object · then right-click parts to exclude them'}
          </span>
        )}
        {wandMode && (
          <span className={wandRunaway ? 'hint warn' : 'hint'}>
            {wandBusy ? (wandSam ? 'SAM2 encoding…' : 'selecting…')
              : wandStats
                ? [
                    `${wandSamples.length} click${wandSamples.length === 1 ? '' : 's'}`,
                    `${wandStats.count.toLocaleString()} px`,
                    `${(100 * wandStats.count / Math.max(wandStats.valid, 1)).toFixed(1)}%`,
                    wandStats.capped
                      && `capped of ${wandStats.available.toLocaleString()}`,
                    wandStats.protected
                      && `${wandStats.protected.toLocaleString()} protected`,
                    wandRunaway && 'runaway — narrow the tolerance',
                  ].filter(Boolean).join(' · ')
                : 'click a region · right-click to exclude · Enter: fill · Esc: cancel'}
          </span>
        )}
        <span className="spacer" />
        <label className="opacity-control">
          overlay {Math.round(opacity * 100)}%
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(Number(e.target.value))}
            onWheel={(e) => {
              e.preventDefault()
              const next = opacity + (e.deltaY < 0 ? 0.05 : -0.05)
              setOpacity(Math.min(1, Math.max(0, Number(next.toFixed(2)))))
            }}
          />
        </label>
      </div>
      {/* A dedicated row rather than more chips beside the title: the title bar
          already overflowed at three sliders. Mounted only in wand mode. */}
      {wandMode && (
        <div className="wand-controls">
          <WandSlider
            label="tolerance"
            value={wandSam ? wandSamTolerance : wandTolerance}
            min={1}
            max={100}
            readout={wandSam
              // The band that works under SAM features is narrow enough to be
              // worth being able to name, so the cosine value is shown too.
              ? (WAND_SAM_TOL_MIN + (wandSamTolerance / 100) * WAND_SAM_TOL_SPAN).toFixed(2)
              : (wandTolerance / WAND_TOL_SCALE).toFixed(3)}
            onChange={(v) => {
              if (wandSam) setWandSamTolerance(v)
              else setWandTolerances((t) => ({ ...t, [tool.classId]: v }))
            }}
          />
          <WandSlider
            label="max px"
            value={wandBudgetStep}
            min={1}
            max={100}
            readout={wandBudget(wandBudgetStep)?.toLocaleString() ?? 'none'}
            onChange={setWandBudgetStep}
          />
          <WandSlider
            label="sample"
            value={wandSampleSize}
            min={1}
            max={25}
            readout={`${wandSampleSize} px`}
            onChange={setWandSampleSize}
          />
          <span className="spacer" />
          <label
            className="wand-toggle"
            title={samAvailable
              ? 'Group by SAM2 encoder features (texture and learned appearance) instead of by the image bands'
              : 'SAM2 model files not found in sam2/'}
          >
            <input
              type="checkbox"
              checked={wandSam}
              disabled={!samAvailable}
              onChange={(e) => setWandSam(e.target.checked)}
            />
            SAM features
          </label>
          {wandSam && (
            <div className="segmented compact">
              {['fine', 'mid', 'deep'].map((level) => (
                <button
                  key={level}
                  className={wandLevel === level ? 'active' : ''}
                  title={{
                    fine: '32ch at 256² — boundaries where the material changes',
                    mid: '64ch at 128²',
                    deep: '256ch at 64² — one cell covers 8×8 px',
                  }[level]}
                  onClick={() => setWandLevel(level)}
                >
                  {level}
                </button>
              ))}
            </div>
          )}
          <label className="wand-toggle" title="Never paint over pixels that already carry a label">
            <input
              type="checkbox"
              checked={wandProtect}
              onChange={(e) => setWandProtect(e.target.checked)}
            />
            protect labels
          </label>
          <label className="wand-toggle" title="Drop the connectivity bound: select matching pixels anywhere in the image">
            <input
              type="checkbox"
              checked={wandGlobal}
              onChange={(e) => setWandGlobal(e.target.checked)}
            />
            whole image
          </label>
        </div>
      )}
      <Viewport image={image} view={view} setView={setView}>
        <img src={`data:image/png;base64,${image?.png}`} alt="" draggable={false} />
        <canvas
          ref={setCanvas}
          width={image?.width}
          height={image?.height}
          style={{ cursor: cursorCss }}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onPointerLeave={hideCursor}
          onWheel={hideCursor}
          onDoubleClick={polygonMode ? closePolygon : undefined}
          // In SAM2 and wand mode right-click is the "exclude" gesture, so the
          // webview's context menu must not open on it.
          onContextMenu={samMode || wandMode ? (e) => e.preventDefault() : undefined}
        />
        {/* The wand preview lives on its own canvas above the label canvas so
            the pending selection never touches the label buffer, and so it can
            be redrawn per slider step without repainting the labels. */}
        <canvas
          ref={wandCanvasRef}
          className="wand-preview"
          width={image?.width}
          height={image?.height}
          style={{ display: wandMode ? 'block' : 'none' }}
        />
        {polygonMode && verts.length > 0 && image && (
          <svg
            className="poly-preview"
            viewBox={`0 0 ${image.width} ${image.height}`}
          >
            <polyline
              points={verts.map((p) => `${p.x},${p.y}`).join(' ')}
              fill={`rgba(${brushColor.join(',')}, 0.15)`}
              stroke={`rgba(${brushColor.join(',')}, 0.9)`}
              vectorEffect="non-scaling-stroke"
            />
            {verts.map((p, i) => (
              <circle
                key={i}
                cx={p.x}
                cy={p.y}
                r={view ? 3 / view.scale : 3}
                fill={`rgb(${brushColor.join(',')})`}
              />
            ))}
          </svg>
        )}
        {samMode && (samPolys.length > 0 || samPoints.length > 0) && image && (
          <svg className="poly-preview" viewBox={`0 0 ${image.width} ${image.height}`}>
            {samPolys.map((poly, i) => (
              <polygon
                key={i}
                points={poly.map((p) => `${p.x},${p.y}`).join(' ')}
                fill={`rgba(${brushColor.join(',')}, 0.35)`}
                stroke={`rgba(${brushColor.join(',')}, 0.9)`}
                vectorEffect="non-scaling-stroke"
              />
            ))}
            {/* Include = pink, exclude = red (the app's --accent / --danger).
                Translucent fill with the same colour at full opacity for the
                stroke, so a marker stays legible over both the image and the
                selection fill. The exclude marker keeps its bar: pink and red
                are close enough that colour alone is a weak distinction. */}
            {samPoints.map((p, i) => {
              const r = view ? 6 / view.scale : 6
              const color = p.label ? SAM_INCLUDE_COLOR : SAM_EXCLUDE_COLOR
              return (
                <g key={i}>
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={r}
                    fill={`rgba(${color.join(',')}, 0.7)`}
                    stroke={`rgb(${color.join(',')})`}
                    strokeWidth={2}
                    vectorEffect="non-scaling-stroke"
                  />
                  {!p.label && (
                    <line
                      x1={p.x - r * 0.6}
                      y1={p.y}
                      x2={p.x + r * 0.6}
                      y2={p.y}
                      stroke={`rgb(${color.join(',')})`}
                      strokeWidth={2}
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                </g>
              )
            })}
          </svg>
        )}
        {wandMode && wandSamples.length > 0 && image && (
          // The same pink/red language the SAM2 markers use in this pane, so
          // the two click-driven tools read the same way. The most recent
          // marker is brighter and thicker, so it is clear what Backspace will
          // take. The exclude marker keeps its bar: pink and red are close
          // enough that colour alone is a weak distinction.
          <svg className="poly-preview" viewBox={`0 0 ${image.width} ${image.height}`}>
            {wandSamples.map((p, i) => {
              const r = view ? 6 / view.scale : 6
              const color = p.label ? SAM_INCLUDE_COLOR : SAM_EXCLUDE_COLOR
              const latest = i === wandSamples.length - 1
              return (
                <g key={i}>
                  <circle
                    cx={p.x + 0.5}
                    cy={p.y + 0.5}
                    r={r}
                    fill={`rgba(${color.join(',')}, ${latest ? 0.85 : 0.5})`}
                    stroke={`rgb(${color.join(',')})`}
                    strokeWidth={latest ? 3 : 1.5}
                    vectorEffect="non-scaling-stroke"
                  />
                  {!p.label && (
                    <line
                      x1={p.x + 0.5 - r * 0.6}
                      y1={p.y + 0.5}
                      x2={p.x + 0.5 + r * 0.6}
                      y2={p.y + 0.5}
                      stroke={`rgb(${color.join(',')})`}
                      strokeWidth={2}
                      vectorEffect="non-scaling-stroke"
                    />
                  )}
                </g>
              )
            })}
          </svg>
        )}
        {!polygonMode && !samMode && !wandMode && oversized && (
          <div
            ref={cursorRef}
            className="brush-cursor"
            style={{
              width: tool.brushSize * 2 * scale,
              height: tool.brushSize * 2 * scale,
              marginLeft: -tool.brushSize * scale,
              marginTop: -tool.brushSize * scale,
              borderWidth: 1.5,
              borderColor: `rgba(${brushColor.join(',')}, 0.9)`,
              background: `rgba(${brushColor.join(',')}, 0.2)`,
            }}
          />
        )}
      </Viewport>
    </div>
  )
}
