import { test, expect, type Page } from '@playwright/test'
import { mkdirSync, writeFileSync, copyFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'
import {
  SANDBOX,
  snapshotFrames,
  restoreSnapshot,
  listSandboxObjectIds,
  deleteSandboxObject,
  getAgentLlmStatus,
  getLastAgentDump,
  waitForBackend,
  type ProjectSnapshot,
} from './helpers/maskOps'

const REPO = '/opt/software/Grounded-SAM3-Instruct-WebTrack'
const REPORT = join(REPO, 'reports', 'agent-e2e-a100-shared')
const SHOTS = join(REPORT, 'screenshots')
const ARTIFACTS = '/opt/cursor/artifacts/screenshots'

const PROMPT =
  'Inspect the current frame. If two mice are present, segment each of them. ' +
  'Reuse existing objects when the names already match; otherwise create objects. ' +
  'Do not start propagation.'

let baseline: ProjectSnapshot
let objectIdsBefore: string[] = []

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
  await page.goto('/')
  await page.getByText('SAM3 Web Tracker').first().waitFor({ timeout: 180_000 })
  await page.getByTitle('Switch video / project').click()
  await page.getByRole('button', { name: SANDBOX.projectName }).click()
  await page.locator('button.flex-1').filter({ hasText: SANDBOX.videoName }).first().click()
  await page.getByTitle('Switch video / project').click()
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
  baseline = await snapshotFrames([SANDBOX.anchorFrame])
  objectIdsBefore = await listSandboxObjectIds()
})

test.afterAll(async () => {
  try {
    if (baseline) await restoreSnapshot(baseline, [SANDBOX.anchorFrame])
    const after = await listSandboxObjectIds()
    for (const oid of after) {
      if (!objectIdsBefore.includes(oid)) await deleteSandboxObject(oid)
    }
  } catch (e) {
    console.warn('sandbox restore failed', e)
  }
})

test('agent inspect + segment two mice from the UI', async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 960 })
  await openSandbox(page)
  await screenshot(page, '01-app-loaded')

  await jumpToFrame(page, SANDBOX.anchorFrame)
  await expect(page.getByText(SANDBOX.projectName).first()).toBeVisible()
  await screenshot(page, '02-project-open')

  await page.getByTitle('Open agent chat').click()
  await expect(page.getByPlaceholder('Prompt the agent…')).toBeVisible()
  await screenshot(page, '03-agent-pane')

  const box = page.getByPlaceholder('Prompt the agent…')
  await box.fill(PROMPT)
  await page.getByTitle('Send').click()

  await expect(page.getByText(/Inspect frame|inspect_frame/i).first()).toBeVisible({ timeout: 240_000 })
  await screenshot(page, '04-after-inspect')

  const segmented = page.getByText(/Text segment|Point prompt|Create object|Evaluate masks/i).first()
  await expect(segmented).toBeVisible({ timeout: 360_000 })
  await screenshot(page, '05-after-segment')

  await expect(page.getByTitle('Send')).toBeVisible({ timeout: 360_000 })
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
})

function relativeToReport(abs: string) {
  if (!abs.startsWith(REPO)) return abs
  const rel = abs.slice(REPO.length).replace(/^\//, '')
  return join('..', '..', rel)
}
