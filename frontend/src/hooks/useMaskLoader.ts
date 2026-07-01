import { useEffect, useRef } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../store/useStore'
import { clearDisplayMissCache, loadDisplayBitmap, loadPerObjectMasks, prefetchMaskWindow } from '../utils/maskLoader'

/**
 * Central mask fetch + prefetch for frame scrubbing.
 * Per-object JSON is the reliable path; display WebP is an optional fast overlay.
 */
export function useMaskLoader(): void {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const {
    project,
    currentVideoId,
    currentFrame,
    setSavedMask,
  } = store

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''
  const lastFrameRef = useRef(currentFrame)

  useEffect(() => {
    if (pid && vid) clearDisplayMissCache(pid, vid)
  }, [pid, vid])

  useEffect(() => {
    if (!pid || !vid) return

    const propagated = video?.propagated_frames ?? []
    const propagatedSet = new Set(propagated)

    const prev = lastFrameRef.current
    const direction: -1 | 0 | 1 =
      currentFrame > prev ? 1 : currentFrame < prev ? -1 : 0
    lastFrameRef.current = currentFrame

    // Current frame first — masks must always load even without display WebP.
    void loadPerObjectMasks(pid, vid, currentFrame).then(masks => {
      if (masks) setSavedMask(currentFrame, masks)
    })

    // Display WebP is optional; only fetch for the visible frame.
    void loadDisplayBitmap(pid, vid, currentFrame)

    if (propagatedSet.size === 0) return

    // Defer adjacent prefetch so current-frame work is not starved.
    const t = window.setTimeout(() => {
      prefetchMaskWindow(
        pid,
        vid,
        currentFrame,
        direction,
        propagatedSet,
        (fidx, masks) => {
          if (!useStore.getState().savedMaskCache[fidx]) {
            setSavedMask(fidx, masks)
          }
        },
      )
    }, 32)

    return () => { window.clearTimeout(t) }
  }, [pid, vid, currentFrame, video?.propagated_frames, setSavedMask])
}
