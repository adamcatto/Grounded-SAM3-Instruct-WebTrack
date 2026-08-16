import { test, expect, type Page } from '@playwright/test'
import { mkdirSync, writeFileSync, copyFileSync, existsSync, readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import {
  SANDBOX,
  resetSandboxVideo,
  ensureSandboxObjects,
  sandboxObjectIdsByName,
  listSandboxObjectIds,
  deleteSandboxObject,
  getAgentLlmStatus,
  getLastAgentDump,
  getSavedMasks,
  getPointPrompts,
  waitForBackend,
} from './helpers/maskOps'
import { parseSseChunk } from '../frontend/src/api/sseParse'

const REPO = '/opt/software/Grounded-SAM3-Instruct-WebTrack'
const REPORT = join(REPO, 'reports', 'agent-e2e-a100-shared')
const SHOTS = join(REPORT, 'screenshots')
const ARTIFACTS = '/opt/cursor/artifacts/screenshots'

const PROMPT = 'Segment the two dark mice.'

let objHead = SANDBOX.objA
let objNo = SANDBOX.objB

test.describe.configure({ mode: 'serial' })
test.setTimeout(15 * 60_000)

function shotPath(name: string) {
  mkdirSync(SHOTS, { recursive: true })
  mkdirSync(ARTIFACTS, { recursive: true })
  return { repo: join(SHOTS, `${name}.png`), artifact: join(ARTIFACTS, `${name}.png`) }
}

async function screenshot(page: Page, name: string) {
  const paths = shotPath(name)
  await page.screenshot({ path: paths.repo, fullPage: true })
  if (existsSync(paths.repo)) copyFileSync(paths.repo, paths.artifact)
  return paths.repo
}

async function openSandbox(page: Page) {
  const jump = page.getByTitle('Jump to frame')
  await page.getByTitle('Switch video / project').click()
  await page.getByRole('button', { name: SANDBOX.projectName }).click()
  // Selecting a project auto-closes the drawer and picks an arbitrary video.
  // Re-open and click the sandbox clip via DOM (the list is taller than the viewport).
  await page.waitForTimeout(400)
  const vidBtn = page.locator('button').filter({ hasText: SANDBOX.videoName }).first()
  if (!(await vidBtn.isVisible().catch(() => false))) {
    await page.getByTitle('Switch video / project').click()
  }
  await vidBtn.waitFor({ state: 'attached', timeout: 20_000 })
  await vidBtn.evaluate((el: HTMLElement) => el.click())
  await jump.waitFor({ timeout: 30_000 })
}

async function scrollAgentPane(page: Page) {
  const scroller = page.locator('div.h-full.flex.flex-col').filter({ has: page.getByText('Prompt the agent…') }).locator('.overflow-y-auto').first()
  if (await scroller.count()) {
    await scroller.evaluate((el: HTMLElement) => { el.scrollTop = el.scrollHeight })
  }
}

async function jumpToFrame(page: Page, frame: number) {
  const input = page.getByTitle('Jump to frame')
  await input.click()
  await input.fill(String(frame))
  await input.press('Enter')
  await expect(input).toHaveValue(String(frame))
  await page.waitForTimeout(800)
}

test.beforeAll(async () => {
  await waitForBackend()
  const status = await getAgentLlmStatus()
  expect(status.configured, `LLM not configured: ${JSON.stringify(status)}`).toBeTruthy()
  expect(status.provider, `expected vLLM, got ${JSON.stringify(status)}`).toBe('vllm')
  await resetSandboxVideo(true)
  await ensureSandboxObjects()
  const ids = await sandboxObjectIdsByName()
  expect(ids.headshave, 'HeadShave object missing after reset').toBeTruthy()
  expect(ids.noshave, 'NoShave object missing after reset').toBeTruthy()
  objHead = ids.headshave as string
  objNo = ids.noshave as string
  const prompts = await getPointPrompts(SANDBOX.pid)
  expect(prompts, 'point prompts should be empty after reset').toEqual({})
  const masks = await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.anchorFrame)
  expect(masks.hasMasks, 'saved masks should be empty after reset').toBeFalsy()
}, { timeout: 180_000 })

test.afterAll(async () => {
  try {
    const named = await sandboxObjectIdsByName()
    const keep = new Set([named.headshave, named.noshave].filter(Boolean) as string[])
    const after = await listSandboxObjectIds()
    for (const oid of after) {
      if (!keep.has(oid)) await deleteSandboxObject(oid)
    }
    await resetSandboxVideo(true)
    await ensureSandboxObjects()
  } catch (e) {
    console.warn('sandbox reset after E2E failed', e)
  }
}, { timeout: 180_000 })

test('parses CRLF SSE agent events', () => {
  const crlf = 'event: ui\r\ndata: {"action":"set_masks","frame_idx":160,"masks":{"1":"abc"}}\r\n'
  const parsed = parseSseChunk(crlf)
  expect(parsed?.event).toBe('ui')
  expect(JSON.parse(parsed!.data)).toEqual({ action: 'set_masks', frame_idx: 160, masks: { '1': 'abc' } })
  const lf = 'event: tool_call\ndata: {"name":"inspect_frame"}\n'
  expect(parseSseChunk(lf)?.event).toBe('tool_call')
})

async function waitForCanvasMasks(page: Page, minCount: number, timeoutMs: number) {
  const canvas = page.locator('canvas[data-mask-count]')
  await expect.poll(async () => {
    const raw = await canvas.getAttribute('data-mask-count')
    return Number(raw || '0')
  }, { timeout: timeoutMs }).toBeGreaterThanOrEqual(minCount)
}

test('agent inspect + segment two mice from the UI', async ({ page }) => {
  await waitForBackend()
  await page.setViewportSize({ width: 1600, height: 960 })
  await page.goto('/')
  await page.getByText('SAM3 Web Tracker').first().waitFor({ timeout: 180_000 })
  await screenshot(page, '01-app-loaded')

  await openSandbox(page)

  await jumpToFrame(page, SANDBOX.anchorFrame)
  await expect(page.getByText(SANDBOX.projectName).first()).toBeVisible()
  await screenshot(page, '02-project-open')

  await page.getByTitle('Open agent chat').click()
  await expect(page.getByPlaceholder('Prompt the agent…')).toBeVisible()
  await screenshot(page, '03-agent-pane')

  const box = page.getByPlaceholder('Prompt the agent…')
  await box.fill(PROMPT)
  await page.getByTitle('Send').click()

  const inspect = page.getByText(/Inspect frame \d+/i)
  const agentError = page.locator('.text-red-300')
  const deadline = Date.now() + 240_000
  while (Date.now() < deadline) {
    if (await inspect.first().isVisible().catch(() => false)) break
    if (await agentError.first().isVisible().catch(() => false)) {
      throw new Error(`agent error: ${await agentError.first().innerText()}`)
    }
    await page.waitForTimeout(1000)
  }
  await expect(inspect.first()).toBeVisible({ timeout: 5_000 })
  await scrollAgentPane(page)
  await screenshot(page, '04-after-inspect')

  const segmented = page.getByText(/Point prompt on obj|Text segment:|Evaluate masks on frame/i)
  await expect(segmented.first()).toBeVisible({ timeout: 360_000 })
  await scrollAgentPane(page)

  const maskDeadline = Date.now() + 360_000
  while (Date.now() < maskDeadline) {
    const saved = await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.anchorFrame)
    if (saved.masks[objHead] && saved.masks[objNo]) break
    if (await agentError.first().isVisible().catch(() => false)) {
      throw new Error(`agent error: ${await agentError.first().innerText()}`)
    }
    await page.waitForTimeout(1000)
  }
  const savedBeforeShot = await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.anchorFrame)
  expect(savedBeforeShot.masks[objHead], 'HeadShave mask missing before screenshot').toBeTruthy()
  expect(savedBeforeShot.masks[objNo], 'NoShave mask missing before screenshot').toBeTruthy()

  await jumpToFrame(page, SANDBOX.anchorFrame)
  await waitForCanvasMasks(page, 2, 30_000)
  await screenshot(page, '05-after-segment')

  await expect(page.getByTitle('Send')).toBeVisible({ timeout: 360_000 })
  await jumpToFrame(page, SANDBOX.anchorFrame)
  await waitForCanvasMasks(page, 2, 30_000)
  await scrollAgentPane(page)
  await screenshot(page, '06-final')

  const dump = await getLastAgentDump()
  mkdirSync(REPORT, { recursive: true })
  writeFileSync(join(REPORT, 'last_dump.json'), JSON.stringify(dump, null, 2))

  const dumpDir = typeof dump.dump_dir === 'string' ? dump.dump_dir : ''
  const contextMd = typeof dump.context_md === 'string' ? dump.context_md : join(dumpDir, 'context.md')
  const shots = [
    '01-app-loaded',
    '02-project-open',
    '03-agent-pane',
    '04-after-inspect',
    '05-after-segment',
    '06-final',
  ]
  const report = [
    '# Agent E2E (A100-80GB shared profile)',
    '',
    `Project: \`${SANDBOX.projectName}\`  video: \`${SANDBOX.videoName}\`  frame: ${SANDBOX.anchorFrame}`,
    '',
    `Prompt: ${PROMPT}`,
    '',
    `LLM: \`${JSON.stringify(dump.llm ?? {})}\``,
    '',
    `Tool calls: ${dump.tool_call_count ?? 'n/a'}`,
    '',
    'Backend saved masks on disk; the canvas overlay comes from agent SSE `set_masks` (CRLF-parsed) plus a refetch after refresh/done so screenshots are not taken on a cleared overlay.',
    '',
    '## Screenshots',
    '',
    ...shots.map(s => `### ${s}\n\n![${s}](screenshots/${s}.png)\n`),
    '## Context trace',
    '',
    dumpDir
      ? `Full dump: \`${dumpDir}\`  ([context.md](${relativeToReport(contextMd)}))`
      : '_No context dump was written. Check backend logs._',
    '',
    'Inspect JPEGs in final LLM context:',
    '',
    '```json',
    JSON.stringify(dump.inspect_jpegs_in_final_context ?? [], null, 2),
    '```',
    '',
    'Inspect JPEGs seen:',
    '',
    '```json',
    JSON.stringify(dump.inspect_jpegs_seen ?? [], null, 2),
    '```',
    '',
  ].join('\n')
  writeFileSync(join(REPORT, 'README.md'), report)

  expect(dumpDir, 'agent context dump_dir missing').toBeTruthy()
  if (dumpDir && existsSync(dumpDir)) {
    for (const f of readdirSync(dumpDir)) {
      if (f.startsWith('inspect_') && f.endsWith('.jpg')) {
        copyFileSync(join(dumpDir, f), join(ARTIFACTS, f))
      }
    }
  }
  const masks = await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.anchorFrame)
  expect(masks.masks[objHead], 'HeadShave mask missing after agent run').toBeTruthy()
  expect(masks.masks[objNo], 'NoShave mask missing after agent run').toBeTruthy()
  const canvasIds = await page.locator('canvas[data-mask-ids]').getAttribute('data-mask-ids')
  expect(canvasIds, 'canvas mask ids missing').toBeTruthy()
  expect(canvasIds?.split(','), 'HeadShave not on canvas').toContain(objHead)
  expect(canvasIds?.split(','), 'NoShave not on canvas').toContain(objNo)

  const ev = lastEvaluateResult(dumpDir, SANDBOX.anchorFrame)
  if (!ev) {
    throw new Error(`agent never called evaluate_segmentation on frame ${SANDBOX.anchorFrame}`)
  }
  expect(ev.masked_object_count ?? 0, 'both mice must have masks').toBeGreaterThanOrEqual(2)
  expect(ev.incomplete, 'evaluate_segmentation still incomplete').toBeFalsy()
  const byId: Record<string, {
    bbox_xywh_norm?: number[]
    ok?: boolean
    reason?: string
    dark_iou_left?: number
    dark_iou_right?: number
  }> = {}
  for (const row of ev.objects || []) {
    if (row.object_id) byId[String(row.object_id)] = row
  }
  const hs = byId[objHead]
  const ns = byId[objNo]
  expect(hs?.ok, `HeadShave not ok: ${hs?.reason}`).toBeTruthy()
  expect(ns?.ok, `NoShave not ok: ${ns?.reason}`).toBeTruthy()
  expect(hs?.reason || '', 'HeadShave mask split').not.toMatch(/split_components/)
  expect(ns?.reason || '', 'NoShave mask split').not.toMatch(/split_components/)
  const hscx = bboxCenterX(hs?.bbox_xywh_norm)
  const nscx = bboxCenterX(ns?.bbox_xywh_norm)
  expect(nscx, `NoShave should be the left mouse, cx=${nscx}`).toBeLessThan(0.45)
  expect(hscx, `HeadShave should be the right-of-center mouse, cx=${hscx}`).toBeGreaterThan(0.55)
  expect(hscx, `HeadShave must not be the water-bottle port, cx=${hscx}`).toBeLessThan(0.74)
  expect(Number(hs?.dark_iou_right ?? 0), `HeadShave dark-blob IoU ${hs?.dark_iou_right}`).toBeGreaterThan(0.20)
  expect(Number(ns?.dark_iou_left ?? 0), `NoShave dark-blob IoU ${ns?.dark_iou_left}`).toBeGreaterThan(0.20)
})

function lastEvaluateResult(dumpDir: string, frameIdx?: number): {
  frame_idx?: number
  masked_object_count?: number
  incomplete?: boolean
  objects?: Array<{
    object_id?: string
    bbox_xywh_norm?: number[]
    ok?: boolean
    reason?: string
    dark_iou_left?: number
    dark_iou_right?: number
  }>
} | null {
  const p = join(dumpDir, 'context.json')
  if (!existsSync(p)) return null
  const ctx = JSON.parse(readFileSync(p, 'utf8')) as {
    events?: Array<{ event?: string; data?: { name?: string; result?: Record<string, unknown> } }>
  }
  const events = ctx.events || []
  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i]
    if (ev?.event === 'tool_result' && ev?.data?.name === 'evaluate_segmentation') {
      const result = (ev.data.result || null) as ReturnType<typeof lastEvaluateResult>
      if (frameIdx == null || Number(result?.frame_idx) === frameIdx) {
        return result
      }
    }
  }
  return null
}

function bboxCenterX(bbox?: number[]) {
  if (!bbox || bbox.length < 4) return NaN
  return bbox[0] + bbox[2] / 2
}

function relativeToReport(abs: string) {
  if (!abs.startsWith(REPO)) return abs
  const rel = abs.slice(REPO.length).replace(/^\//, '')
  return join('..', '..', rel)
}
