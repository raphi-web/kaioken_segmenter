// Bottom strip of project image thumbnails; click one to make it the active
// image in both panes, right-click one to move it between the training and
// validation sets. Thumbnails stream in while the backend generates them.

function Thumbnail({ name, thumb, active, validation, disabled, onSelect, onContextMenu }) {
  // The context handler sits on the wrapper, not the button: the button is
  // disabled while training runs, and disabled buttons fire no mouse events at
  // all — the browser's own menu would appear instead of ours.
  return (
    <div className="thumb-slot" onContextMenu={onContextMenu}>
      <button
        className={`thumb${active ? ' active' : ''}${validation ? ' validation' : ''}`}
        disabled={disabled}
        onClick={onSelect}
        title={validation ? `${name} — validation image` : name}
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

export default function ThumbnailStrip({
  images, thumbs, activeName, validation, progress, disabled, onSelect, onContextMenu,
}) {
  if (!images?.length) return null
  const held = new Set(validation ?? [])

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
          disabled={disabled}
          onSelect={() => onSelect(name)}
          onContextMenu={(e) => {
            e.preventDefault()
            onContextMenu(name, e.clientX, e.clientY)
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
}
