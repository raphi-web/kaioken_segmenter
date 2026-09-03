// Bottom strip of project image thumbnails; click one to make it the active
// image in both panes, right-click one to move it between the training and
// validation sets. Thumbnails stream in while the backend generates them.
// Two states are marked on the thumbnail itself: held-out validation images
// (green) and images carrying no user labels yet (red).

import { forwardRef, useImperativeHandle, useRef } from 'react'

function Thumbnail({
  name, thumb, active, validation, unlabeled, disabled, onSelect, onContextMenu, slotRef,
}) {
  const tags = []
  if (validation) tags.push('validation image')
  if (unlabeled) tags.push('not labeled')
  // The context handler sits on the wrapper, not the button: the button is
  // disabled while training runs, and disabled buttons fire no mouse events at
  // all — the browser's own menu would appear instead of ours.
  return (
    <div className="thumb-slot" ref={slotRef} onContextMenu={onContextMenu}>
      <button
        className={`thumb${active ? ' active' : ''}${validation ? ' validation' : ''}${unlabeled ? ' unlabeled' : ''}`}
        disabled={disabled}
        onClick={onSelect}
        title={tags.length ? `${name} — ${tags.join(', ')}` : name}
      >
        {thumb
          ? <img src={`data:image/png;base64,${thumb}`} alt={name} draggable={false} />
          : <span className="thumb-placeholder" />}
        {validation && <span className="thumb-badge">VAL</span>}
        <span className="thumb-name">{name}</span>
      </button>
    </div>
  )
}

// Exposes centerOn(name) via ref so callers outside the strip (the accuracy
// report's "jump to image") can scroll a thumbnail into view without the
// strip re-centering on every ordinary click, which would feel jumpy.
const ThumbnailStrip = forwardRef(function ThumbnailStrip({
  images, thumbs, activeName, validation, labeled, activeLabeled, progress, disabled,
  onSelect, onContextMenu,
}, ref) {
  const nodes = useRef(new Map())

  useImperativeHandle(ref, () => ({
    centerOn(name) {
      nodes.current.get(name)?.scrollIntoView({ inline: 'center', block: 'nearest', behavior: 'smooth' })
    },
  }), [])

  if (!images?.length) return null
  const held = new Set(validation ?? [])
  // `labeled` is a snapshot from the last project payload, so the active image
  // is answered from the live pixel count instead — its border has to flip on
  // the first stroke, not on the next project refresh.
  const annotated = new Set(labeled ?? [])

  function handleWheel(e) {
    if (e.deltaY === 0) return
    e.currentTarget.scrollLeft += e.deltaY
    e.preventDefault()
  }

  return (
    <div className="thumb-strip" onWheel={handleWheel}>
      {images.map((name) => (
        <Thumbnail
          key={name}
          name={name}
          thumb={thumbs?.[name]}
          active={name === activeName}
          validation={held.has(name)}
          unlabeled={name === activeName ? !activeLabeled : !annotated.has(name)}
          disabled={disabled}
          onSelect={() => onSelect(name)}
          onContextMenu={(e) => {
            e.preventDefault()
            onContextMenu(name, e.clientX, e.clientY)
          }}
          slotRef={(el) => {
            if (el) nodes.current.set(name, el)
            else nodes.current.delete(name)
          }}
        />
      ))}
      {progress?.running && (
        <span className="thumb-progress">
          thumbnails {progress.done}/{progress.total}…
        </span>
      )}
    </div>
  )
})

export default ThumbnailStrip
