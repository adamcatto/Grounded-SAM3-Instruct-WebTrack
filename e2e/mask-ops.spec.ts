import { test, expect, type Page } from '@playwright/test'
import {
  SANDBOX,
  snapshotFrames,
  restoreSnapshot,
  getSavedMasks,
  getPointPrompts,
  masksSwapped,
  promptsUnchanged,
  type ProjectSnapshot,
} from './helpers/maskOps'

let baseline: ProjectSnapshot

test.describe.configure({ mode: 'serial' })

test.beforeAll(async () => {
  const frames = [SANDBOX.anchorFrame, SANDBOX.nonAnchorFrame]
  baseline = await snapshotFrames(frames)
  for (const f of frames) {
    expect(baseline.frames[f].hasMasks, `frame ${f} needs masks`).toBeTruthy()
    expect(baseline.frames[f].masks[SANDBOX.objA], `frame ${f} obj A`).toBeTruthy()
    expect(baseline.frames[f].masks[SANDBOX.objB], `frame ${f} obj B`).toBeTruthy()
  }
})

test.afterAll(async () => {
  if (baseline) await restoreSnapshot(baseline)
})

async function openSandbox(page: Page) {
  await page.goto('/')
  await page.getByText('SAM3 Web Tracker').first().waitFor({ timeout: 120_000 })

  await page.getByTitle('Switch video / project').click()
  await page.getByRole('button', { name: SANDBOX.projectName }).click()
  await page
    .locator('button.flex-1')
    .filter({ hasText: SANDBOX.videoName })
    .first()
    .click()
  await page.getByTitle('Switch video / project').click()
}

async function jumpToFrame(page: Page, frame: number) {
  const input = page.getByTitle('Jump to frame')
  await input.click()
  await input.fill(String(frame))
  await input.press('Enter')
  await expect(input).toHaveValue(String(frame))
  // Allow mask prefetch
  await page.waitForTimeout(500)
}

async function ensureMasksLoaded(page: Page, frame: number) {
  await jumpToFrame(page, frame)
  const api = await getSavedMasks(SANDBOX.pid, SANDBOX.vid, frame)
  expect(api.hasMasks).toBeTruthy()
}

test.describe('mask swap & clear (browser + API, reversible)', () => {
  test('swap masks on anchor frame via UI', async ({ page }) => {
    await openSandbox(page)
    await ensureMasksLoaded(page, SANDBOX.anchorFrame)

    const before = baseline.frames[SANDBOX.anchorFrame].masks
    const promptsBefore = await getPointPrompts(SANDBOX.pid)

    await page.getByTitle(/Swap two objects' masks/).click()
    await expect(page.getByRole('heading', { name: 'Swap object masks' })).toBeVisible()
    await page.getByRole('button', { name: 'Swap', exact: true }).click()
    await expect(page.getByText(/Swapped masks for 1 frame/)).toBeVisible({ timeout: 180_000 })

    const after = (await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.anchorFrame)).masks
    expect(masksSwapped(before, after, SANDBOX.objA, SANDBOX.objB)).toBeTruthy()

    const promptsAfter = await getPointPrompts(SANDBOX.pid)
    expect(promptsUnchanged(promptsBefore, promptsAfter)).toBeTruthy()

    // Undo via second swap (same as app undo)
    await page.getByTitle(/Swap two objects' masks/).click()
    await page.getByRole('button', { name: 'Swap', exact: true }).click()
    await expect(page.getByText(/Swapped masks for 1 frame/)).toBeVisible({ timeout: 180_000 })
    const restored = (await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.anchorFrame)).masks
    expect(restored[SANDBOX.objA]).toBe(before[SANDBOX.objA])
    expect(restored[SANDBOX.objB]).toBe(before[SANDBOX.objB])
  })

  test('swap masks on non-anchor frame via UI', async ({ page }) => {
    await openSandbox(page)
    await ensureMasksLoaded(page, SANDBOX.nonAnchorFrame)

    const before = baseline.frames[SANDBOX.nonAnchorFrame].masks

    await page.getByTitle(/Swap two objects' masks/).click()
    await page.getByRole('button', { name: 'Swap', exact: true }).click()
    await expect(page.getByText(/Swapped masks for 1 frame/)).toBeVisible({ timeout: 180_000 })

    const after = (await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.nonAnchorFrame)).masks
    expect(masksSwapped(before, after, SANDBOX.objA, SANDBOX.objB)).toBeTruthy()

    await page.getByTitle(/Swap two objects' masks/).click()
    await page.getByRole('button', { name: 'Swap', exact: true }).click()
    const restored = (await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.nonAnchorFrame)).masks
    expect(restored[SANDBOX.objA]).toBe(before[SANDBOX.objA])
  })

  test('clear object selection on anchor frame (ObjectCard)', async ({ page }) => {
    await openSandbox(page)
    await jumpToFrame(page, SANDBOX.anchorFrame)

    const objName = 'HeadShave'
    await page.getByText(objName).first().click()
    await page.getByRole('button', { name: 'Clear selection' }).click()

    await page.waitForTimeout(800)
    const masks = await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.anchorFrame)
    expect(masks.masks[SANDBOX.objA]).toBeUndefined()
    expect(masks.masks[SANDBOX.objB]).toBeTruthy()

    const prompts = await getPointPrompts(SANDBOX.pid)
    expect(prompts[SANDBOX.objA]?.[String(SANDBOX.anchorFrame)]).toBeUndefined()

    await restoreSnapshot(baseline)
    const afterRestore = await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.anchorFrame)
    expect(afterRestore.masks[SANDBOX.objA]).toBe(baseline.frames[SANDBOX.anchorFrame].masks[SANDBOX.objA])
  })

  test('clear object selection on non-anchor frame', async ({ page }) => {
    await openSandbox(page)
    await jumpToFrame(page, SANDBOX.nonAnchorFrame)

    await page.getByText('HeadShave').first().click()
    await page.getByRole('button', { name: 'Clear selection' }).click()
    await page.waitForTimeout(800)

    const masks = await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.nonAnchorFrame)
    expect(masks.masks[SANDBOX.objA]).toBeUndefined()
    expect(masks.masks[SANDBOX.objB]).toBeTruthy()

    await restoreSnapshot(baseline)
  })

  test('clear all masks for frame (LeftPanel) on non-anchor', async ({ page }) => {
    await openSandbox(page)
    await ensureMasksLoaded(page, SANDBOX.nonAnchorFrame)

    await page.getByTitle(`Clear saved masks for frame ${SANDBOX.nonAnchorFrame}`).click()
    await page.getByRole('button', { name: /This frame only/ }).click()
    await expect(page.getByText('Masks cleared')).toBeVisible({ timeout: 10_000 })

    const masks = await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.nonAnchorFrame)
    expect(masks.hasMasks).toBeFalsy()

    // Point prompts on anchor should be untouched
    const prompts = await getPointPrompts(SANDBOX.pid)
    expect(prompts[SANDBOX.objA]?.[String(SANDBOX.anchorFrame)]).toBeTruthy()

    await restoreSnapshot(baseline)
  })
})
