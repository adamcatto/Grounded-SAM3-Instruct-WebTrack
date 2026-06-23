import { useMemo } from 'react'
import { Home, Target, Terminal } from 'lucide-react'
import { setProjectsRoot, type ProjectsRootInfo } from '../api/client'
import { pathParent } from '../utils/fsPaths'
import FolderBrowserModal, { type FolderBrowserQuickJump } from './FolderBrowserModal'

type Props = {
  open: boolean
  rootInfo: ProjectsRootInfo | null
  onClose: () => void
  onApplied: () => void
  addToast: (message: string, type: 'success' | 'error' | 'info') => void
}

export default function ProjectsFolderBrowserModal({
  open,
  rootInfo,
  onClose,
  onApplied,
  addToast,
}: Props) {
  const quickJumps = useMemo((): FolderBrowserQuickJump[] => {
    if (!rootInfo) return []
    const jumps: FolderBrowserQuickJump[] = []
    if (rootInfo.home) {
      jumps.push({
        id: 'home',
        label: 'Home',
        path: rootInfo.home,
        icon: <Home size={12} />,
        title: `Home (${rootInfo.home})`,
      })
    }
    if (rootInfo.app_root) {
      jumps.push({
        id: 'app',
        label: 'App folder',
        path: rootInfo.app_root,
        icon: <Terminal size={12} />,
        title: `App folder — where start_frontend.sh lives (${rootInfo.app_root})`,
      })
    }
    if (rootInfo.active_root) {
      jumps.push({
        id: 'active',
        label: 'Active',
        path: rootInfo.active_root,
        icon: <Target size={12} />,
        title: `Current projects folder (${rootInfo.active_root})`,
      })
    }
    return jumps
  }, [rootInfo])

  const fallbackPaths = useMemo(() => {
    if (!rootInfo) return []
    const parent = pathParent(rootInfo.active_root)
    return [parent, rootInfo.active_root, rootInfo.home, rootInfo.app_root].filter(
      (v, i, arr) => v && arr.indexOf(v) === i,
    )
  }, [rootInfo])

  async function handleSelect(path: string) {
    await setProjectsRoot(path)
    addToast(`Projects folder: ${path}`, 'success')
    onApplied()
    onClose()
  }

  return (
    <FolderBrowserModal
      open={open}
      mode="folder"
      title="Projects folder"
      subtitle="Folders containing a config.json show as projects"
      confirmLabel="Use this folder"
      initialPath={rootInfo?.active_root}
      fallbackPaths={fallbackPaths}
      quickJumps={quickJumps}
      showProjectBadge
      onClose={onClose}
      onSelect={handleSelect}
    />
  )
}
