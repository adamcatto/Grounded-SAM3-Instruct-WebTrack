import { useEffect, useRef } from 'react'
import { useStore, currentVideo as selectCurrentVideo } from '../store/useStore'
import { loadPerObjectMasks, prefetchMaskWindow } from '../utils/maskLoader'

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
    setSavedMask,
  } = store

  const pid = project?.id ?? ''
  const vid = currentVideoId ?? ''
  const lastFrameRef = useRef(currentFrame)

  useEffect(() => {
    if (!pid || !vid) return

    const propagated = video?.propagated_frames ?? []
    const propagatedSet = new Set(propagated)

    const prev = lastFrameRef.current
    const direction: -1 | 0 | 1 =
      currentFrame > prev ? 1 : currentFrame < prev ? -1 : 0
    lastFrameRef.current = currentFrame

    // Always try to load masks for the current frame (annotation + review).
    if (!useStore.getState().savedMaskCache[currentFrame]) {
      void loadPerObjectMasks(pid, vid, currentFrame).then(masks => {
        if (masks) setSavedMask(currentFrame, masks)
      })
    }

    // Directional prefetch once propagation has written frames to disk.
    if (propagatedSet.size > 0) {
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
    }
  }, [pid, vid, currentFrame, video?.propagated_frames, setSavedMask])
}
