import { useEffect, useRef } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../store/useStore'
import {
  loadDisplayBitmap,
  loadPerObjectMasks,
  prefetchDisplayWindow,
  prefetchMaskWindow,
} from '../utils/maskLoader'

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
  const loadSeqRef = useRef(0)

  useEffect(() => {
    if (!pid || !vid) return

    const propagated = video?.propagated_frames ?? []
    const propagatedSet = new Set(propagated)

    const prev = lastFrameRef.current
    const frameDelta = currentFrame - prev
    const direction: -1 | 0 | 1 =
      frameDelta > 0 ? 1 : frameDelta < 0 ? -1 : 0
    const jumpDistance = Math.abs(frameDelta)
    const isFarJump = jumpDistance > Math.max(10, store.frameJump * 2)
    lastFrameRef.current = currentFrame
    const seq = ++loadSeqRef.current
    const displayAbort = new AbortController()

    // Display WebP is the scrub fast path: one compact composite per frame.
    void loadDisplayBitmap(pid, vid, currentFrame, displayAbort.signal)

    const detailTimer = window.setTimeout(() => {
      void loadPerObjectMasks(pid, vid, currentFrame).then(masks => {
        if (masks && loadSeqRef.current === seq) setSavedMask(currentFrame, masks)
      })
    }, isFarJump ? 500 : 180)

    if (propagatedSet.size === 0) {
      return () => {
        displayAbort.abort()
        window.clearTimeout(detailTimer)
      }
    }

    const displayTimer = window.setTimeout(() => {
      if (!isFarJump && loadSeqRef.current === seq) {
        prefetchDisplayWindow(pid, vid, currentFrame, direction, propagatedSet)
      }
    }, 32)

    const objectTimer = window.setTimeout(() => {
      if (isFarJump || loadSeqRef.current !== seq) return
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
    }, 220)

    return () => {
      displayAbort.abort()
      window.clearTimeout(detailTimer)
      window.clearTimeout(displayTimer)
      window.clearTimeout(objectTimer)
    }
  }, [pid, vid, currentFrame, video?.propagated_frames, setSavedMask])
}
