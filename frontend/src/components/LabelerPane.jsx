import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CLASS_COLORS, LABEL_ALPHA, UNLABELED } from '../constants'
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

// Right pane: base RGB image with a brush-paintable label canvas on top.
// Labels live in a Uint8Array (values 0/1/UNLABELED); the canvas is a
// colorized rendering of that array, redrawn per stroke segment.
// Left button paints (brush mode) or places polygon vertices (polygon mode);
// middle-drag pans (handled by the surrounding Viewport).
// Every stroke/fill reports a Map<pixelIndex, previousValue> via onDiff so
// the app can undo it.
export default function LabelerPane({
  image, labels, labelsVersion, tool, view, setView, onStrokeEnd, onDiff, onSamSnap,
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
  }, [image, labels, labelsVersion, canvasReady, opacity])

  // Leaving polygon mode discards an unfinished polygon.
  useEffect(() => {
    if (tool.mode !== 'polygon') setVerts([])
    // Leaving SAM2 mode drops an uncommitted selection rather than leaving an
    // invisible one to be committed later by a stray Enter.
    if (tool.mode !== 'sam') samCancelRef.current()
  }, [tool.mode])

  // Enter commits the pending shape (polygon fill / SAM2 selection), Escape
  // cancels it; in SAM2 mode Backspace drops just the last click.
  useEffect(() => {
    function onKey(e) {
      const mode = settings.current.mode
      if (mode === 'polygon') {
        if (e.key === 'Enter') closeRef.current()
        else if (e.key === 'Escape') setVerts([])
      } else if (mode === 'sam') {
        if (e.key === 'Enter') samCommitRef.current()
        else if (e.key === 'Escape') samCancelRef.current()
        else if (e.key === 'Backspace') {
          e.preventDefault() // otherwise the webview treats it as "go back"
          samUndoRef.current()
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
      if (!(v in CLASS_COLORS)) continue
      const [r, g, b] = CLASS_COLORS[v]
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

  const brushColor = tool.eraser ? [255, 255, 255] : CLASS_COLORS[tool.classId]
  const polygonMode = tool.mode === 'polygon'
  const samMode = tool.mode === 'sam'

  // Brush cursor: draw it into a native CSS cursor (zero pointer lag) while it
  // fits the browser's 128px cap; beyond that, fall back to the DOM circle.
  const scale = view?.scale ?? 1
  const oversized = Math.ceil(tool.brushSize * 2 * scale) + CURSOR_PAD * 2 > MAX_CURSOR_PX
  const cursorCss = useMemo(() => {
    if (samMode) return samBusy ? 'wait' : 'crosshair'
    if (polygonMode) return 'crosshair'
    const color = tool.eraser ? [255, 255, 255] : CLASS_COLORS[tool.classId]
    return makeBrushCursor(tool.brushSize * 2 * scale, color) ?? makeCrosshairCursor(color)
  }, [tool.brushSize, scale, tool.eraser, tool.classId, polygonMode, samMode, samBusy])

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
          // In SAM2 mode right-click is the "exclude" gesture, so the webview's
          // context menu must not open on it.
          onContextMenu={samMode ? (e) => e.preventDefault() : undefined}
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
        {!polygonMode && !samMode && oversized && (
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
