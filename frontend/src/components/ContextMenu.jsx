import { useEffect, useLayoutEffect, useRef, useState } from 'react'

// A menu anchored at the mouse position rather than at a trigger button.
// Dismissal matches Dropdown: outside click, Escape, or any click inside.
// `position: fixed` (not the dropdown's `absolute`) so the menu escapes the
// thumbnail strip's horizontal scroll instead of being clipped by it.
export default function ContextMenu({ x, y, onClose, children }) {
  const ref = useRef(null)
  const [pos, setPos] = useState({ left: x, top: y })

  useEffect(() => {
    function onDown(e) {
      if (!ref.current?.contains(e.target)) onClose()
    }
    function onKey(e) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('pointerdown', onDown)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('pointerdown', onDown)
      window.removeEventListener('keydown', onKey)
    }
  }, [onClose])

  // Flip the menu back inside the window when opened near an edge. Measured
  // after mount because the size depends on the item labels.
  useLayoutEffect(() => {
    const box = ref.current?.getBoundingClientRect()
    if (!box) return
    setPos({
      left: Math.max(4, Math.min(x, window.innerWidth - box.width - 4)),
      top: Math.max(4, Math.min(y, window.innerHeight - box.height - 4)),
    })
  }, [x, y])

  return (
    <div className="context-menu" ref={ref} style={pos} onClick={onClose}>
      {children}
    </div>
  )
}
