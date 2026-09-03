import { useEffect, useRef, useState } from 'react'

// A toolbar dropdown: a trigger button that reveals a vertical menu of the
// given children (buttons). Closes on outside click, Escape, or after any
// click inside the menu (so selecting an item dismisses it).
export default function Dropdown({ label, disabled, children }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)

  useEffect(() => {
    if (!open) return
    function onDown(e) {
      if (!ref.current?.contains(e.target)) setOpen(false)
    }
    function onKey(e) {
      if (e.key === 'Escape') setOpen(false)
    }
    window.addEventListener('pointerdown', onDown)
    window.addEventListener('keydown', onKey)
    return () => {
      window.removeEventListener('pointerdown', onDown)
      window.removeEventListener('keydown', onKey)
    }
  }, [open])

  // Any click that reaches here bubbled up from a menu item; dismiss the menu.
  return (
    <div className="dropdown" ref={ref}>
      <button
        className={open ? 'active' : ''}
        disabled={disabled}
        onClick={() => setOpen((o) => !o)}
      >
        {label}
        <span className="caret" />
      </button>
      {open && (
        <div className="dropdown-menu" onClick={() => setOpen(false)}>
          {children}
        </div>
      )}
    </div>
  )
}
