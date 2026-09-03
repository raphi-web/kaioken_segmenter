// Thin wrapper around window.pywebview.api that waits for the bridge to be ready.

const ready = new Promise((resolve) => {
  if (window.pywebview?.api) {
    resolve()
  } else {
    window.addEventListener('pywebviewready', () => resolve(), { once: true })
  }
})

export async function api(method, ...args) {
  await ready
  return window.pywebview.api[method](...args)
}

// Decode base64 into a Uint8Array (inverse of toBase64).
export function fromBase64(b64) {
  const binary = atob(b64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return bytes
}

// Encode a Uint8Array as base64 (chunked to stay under argument limits).
export function toBase64(bytes) {
  let binary = ''
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk))
  }
  return btoa(binary)
}
