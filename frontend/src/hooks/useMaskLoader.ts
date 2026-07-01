import { useEffect, useRef } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../store/useStore'
import { loadCompositeBitmap, loadPerObjectMasks, prefetchMaskWindow } from '../utils/maskLoader'

/**
 * Central mask fetch + prefetch for frame scrubbing.
 * Replaces duplicate effects in AnnotationCanvas and Timeline.
 */
export function useMaskLoader(): void {
  const store = useStore()
  const video = selectCurrentVideo(store)
  const {
    project,
    currentVideoId,
    currentFrame,
    propagationStatus,
    setSavedMask,
  } = store

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''
  const lastFrameRef = useRef(currentFrame)

  useEffect(() => {
    if (!pid || !vid) return
    if (propagationStatus !== 'done' && propagationStatus !== 'running') return

    const propagated = new Set(video?.propagated_frames ?? [])
    if (!propagated.has(currentFrame)) return

    const prev = lastFrameRef.current
    const direction: -1 | 0 | 1 =
      currentFrame > prev ? 1 : currentFrame < prev ? -1 : 0
    lastFrameRef.current = currentFrame

    void loadCompositeBitmap(pid, vid, currentFrame)

    if (!useStore.getState().savedMaskCache[currentFrame]) {
      void loadPerObjectMasks(pid, vid, currentFrame).then(masks => {
        if (masks) setSavedMask(currentFrame, masks)
      })
    }

    prefetchMaskWindow(
      pid,
      vid,
      currentFrame,
      direction,
      propagated,
      (fidx, masks) => {
        if (!useStore.getState().savedMaskCache[fidx]) {
          setSavedMask(fidx, masks)
        }
      },
    )
  }, [pid, vid, currentFrame, propagationStatus, video?.propagated_frames, setSavedMask])
}
