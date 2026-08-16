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

const REPO = '/opt/software/Grounded-SAM3-Instruct-WebTrack'
const REPORT = join(REPO, 'reports', 'agent-e2e-a100-shared')
const SHOTS = join(REPORT, 'screenshots')
const ARTIFACTS = '/opt/cursor/artifacts/screenshots'

const PROMPT = [
  'This home-cage frame has exactly two mice. Each mouse is ONE object: head, body, AND tail in a single connected mask. Do not split body and tail. Do not create extra objects.',
  'Shaved HeadShave is on the FAR RIGHT (torso about 0.78, 0.50). Unshaved NoShave is on the LEFT (torso about 0.32, 0.50).',
  'Inspect first. The canvas should be empty of old points. Then add_point_prompt once per existing object:',
  '1) NoShave: one positive click on the left mouse torso. Include the tail in that same mask — never put a negative click on its tail.',
  '2) HeadShave: one positive click on the right mouse torso. Do not click the circular water-bottle port (~0.85, 0.42) or empty bedding (x=0.50–0.70).',
  'evaluate_segmentation. If split_components, add another positive point on the missing tail/body, still on the same object id. Do not finish until both mice have compact connected masks on opposite sides. Do not start propagation.',
].join(' ')

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

test('agent inspect + segment two mice from the UI', async ({ page }) => {
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
  const prompt = PROMPT.replace('NoShave:', `NoShave (object ${objNo}):`).replace('HeadShave:', `HeadShave (object ${objHead}):`)
  await box.fill(prompt)
  const dumpBefore = await getLastAgentDump().catch(() => ({}))
  const dumpDirBefore = typeof dumpBefore.dump_dir === 'string' ? dumpBefore.dump_dir : ''
  await page.getByTitle('Send').click()

  const inspect = page.getByText(/Inspect frame|inspect_frame/i)
  const agentError = page.locator('.text-red-300')
  const deadline = Date.now() + 240_000
  while (Date.now() < deadline) {
    if (await inspect.first().isVisible().catch(() => false)) break
    if (await agentError.first().isVisible().catch(() => false)) {
      throw new Error(`agent error: ${await agentError.first().innerText()}`)
    }
    const dump = await getLastAgentDump().catch(() => ({}))
    const seen = Array.isArray(dump.inspect_jpegs_seen) ? dump.inspect_jpegs_seen : []
    if (dump.dump_dir && dump.dump_dir !== dumpDirBefore && seen.length > 0) break
    await page.waitForTimeout(1000)
  }
  await scrollAgentPane(page)
  await screenshot(page, '04-after-inspect')

  const segmented = page.getByText(/Text segment|Point prompt|Create object|Evaluate masks/i)
  const segDeadline = Date.now() + 360_000
  while (Date.now() < segDeadline) {
    if (await segmented.first().isVisible().catch(() => false)) break
    const dump = await getLastAgentDump().catch(() => ({}))
    const n = typeof dump.tool_call_count === 'number' ? dump.tool_call_count : 0
    if (dump.dump_dir && dump.dump_dir !== dumpDirBefore && n >= 2) break
    await page.waitForTimeout(1000)
  }
  await scrollAgentPane(page)
  await screenshot(page, '05-after-segment')

  await expect(page.getByTitle('Send')).toBeVisible({ timeout: 360_000 })
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
    `LLM: \`${JSON.stringify(dump.llm ?? {})}\``,
    '',
    `Tool calls: ${dump.tool_call_count ?? 'n/a'}`,
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

  const ev = lastEvaluateResult(dumpDir)
  if (!ev) {
    throw new Error('agent never called evaluate_segmentation')
  }
  expect(ev.masked_object_count ?? 0, 'both mice must have masks').toBeGreaterThanOrEqual(2)
  expect(ev.incomplete, 'evaluate_segmentation still incomplete').toBeFalsy()
  const byId: Record<string, { bbox_xywh_norm?: number[]; ok?: boolean; reason?: string }> = {}
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
  expect(nscx, `NoShave should be the left mouse, cx=${nscx}`).toBeLessThan(0.50)
  expect(hscx, `HeadShave should be the far-right mouse, cx=${hscx}`).toBeGreaterThan(0.70)
})

function lastEvaluateResult(dumpDir: string): {
  masked_object_count?: number
  incomplete?: boolean
  objects?: Array<{ object_id?: string; bbox_xywh_norm?: number[]; ok?: boolean; reason?: string }>
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
      return (ev.data.result || null) as ReturnType<typeof lastEvaluateResult>
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
