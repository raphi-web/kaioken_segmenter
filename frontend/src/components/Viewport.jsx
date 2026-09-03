import { useEffect, useRef } from 'react'

const MIN_SCALE = 0.2
const MAX_SCALE = 32

// Shared zoom/pan container. Both panes render one of these with the same
// `view` state ({scale, x, y}), so zooming/panning stays in sync.
// Wheel = zoom at cursor. Middle-drag = pan (left-drag too if allowLeftPan).
export default function Viewport({ image, view, setView, allowLeftPan = false, children }) {
  const ref = useRef(null)
  const panning = useRef(null)
  const viewRef = useRef(view)
  viewRef.current = view

  function fit() {
    const rect = ref.current.getBoundingClientRect()
    const scale = Math.min(rect.width / image.width, rect.height / image.height)
    setView({
      scale,
      x: (rect.width - image.width * scale) / 2,
      y: (rect.height - image.height * scale) / 2,
    })
  }

  // First mounted viewport computes the initial fit-to-pane view.
  useEffect(() => {
    if (image && !viewRef.current) fit()
  }, [image])

  // Native listener: React's synthetic onWheel is passive and can't preventDefault.
  useEffect(() => {
    const el = ref.current
    if (!el) return
    function onWheel(e) {
      e.preventDefault()
      const v = viewRef.current
      if (!v) return
      const rect = el.getBoundingClientRect()
      const px = e.clientX - rect.left
      const py = e.clientY - rect.top
      const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, v.scale * Math.exp(-e.deltaY * 0.0015)))
      const k = scale / v.scale
      setView({ scale, x: px - (px - v.x) * k, y: py - (py - v.y) * k })
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [setView])

  function onPointerDown(e) {
    if (e.button === 1 || (allowLeftPan && e.button === 0)) {
      e.preventDefault()
      panning.current = { x: e.clientX, y: e.clientY }
      ref.current.setPointerCapture(e.pointerId)
    }
  }

  function onPointerMove(e) {
    if (!panning.current) return
    const v = viewRef.current
    setView({ ...v, x: v.x + e.clientX - panning.current.x, y: v.y + e.clientY - panning.current.y })
    panning.current = { x: e.clientX, y: e.clientY }
  }

  function onPointerUp() {
    panning.current = null
  }

  return (
    <div
      ref={ref}
      className="viewport"
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
      onDoubleClick={allowLeftPan ? fit : undefined}
      style={allowLeftPan ? { cursor: 'grab' } : undefined}
    >
      {image && view && (
        // Zoom via layout size, not transform scale(): composited transform
        // layers get bilinear-filtered by the (Qt WebEngine) compositor, which
        // ignores image-rendering and blurs the pixels. Layout scaling honors
        // it, so zoomed-in pixels stay crisp; below 1:1 smooth scaling reads
        // better. image-rendering is inherited by the img/canvas children.
        //
        // Geometry is rounded to whole pixels only here, at paint time (`view`
        // itself stays fractional so repeated zoom steps don't accumulate
        // drift). A fractional box leaves antialiased half-pixel edges, and
        // when zooming out the shrinking element vacates area that WebEngine
        // does not repaint — the leftover edges pile up as ghost lines around
        // the image. `will-change` puts the stack on its own compositor layer
        // so the vacated area is recomposited rather than partially
        // invalidated; integer translate keeps that layer resample-free.
        <div
          className="canvas-stack"
          style={{
            width: Math.round(image.width * view.scale),
            height: Math.round(image.height * view.scale),
            transform: `translate(${Math.round(view.x)}px, ${Math.round(view.y)}px)`,
            imageRendering: view.scale >= 1 ? 'pixelated' : 'auto',
            willChange: 'transform',
          }}
        >
          {children}
        </div>
      )}
    </div>
  )
}
