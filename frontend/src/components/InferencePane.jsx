import { useState } from 'react'
import Viewport from './Viewport'

// Left pane: base RGB image with the model prediction overlay (opacity slider)
// and an optional model-uncertainty heatmap on top.
export default function InferencePane({
  image, overlay, uncertainty, showUncertainty, onToggleUncertainty, training, view, setView,
}) {
  const [opacity, setOpacity] = useState(0.7)
  // Pulsing dot = a training run is actively updating the prediction; solid =
  // a prediction is loaded but idle; muted = none yet.
  const dotClass = training ? 'on' : overlay ? 'present' : ''

  return (
    <div className="pane">
      <div className="pane-title">
        <span className={`live-dot ${dotClass}`} />
        <span className="pane-name">Live Model Inference</span>
        <span className="spacer" />
        <label className="opacity-control">
          <input
            type="checkbox"
            checked={showUncertainty}
            onChange={(e) => onToggleUncertainty(e.target.checked)}
          />
          uncertainty
        </label>
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
      <Viewport image={image} view={view} setView={setView} allowLeftPan>
        <img src={`data:image/png;base64,${image?.png}`} alt="" draggable={false} />
        {overlay && (
          <img
            src={`data:image/png;base64,${overlay}`}
            alt=""
            draggable={false}
            style={{ opacity }}
          />
        )}
        {showUncertainty && uncertainty && (
          <img
            src={`data:image/png;base64,${uncertainty}`}
            alt=""
            draggable={false}
            style={{ opacity }}
          />
        )}
      </Viewport>
    </div>
  )
}
