import { test, expect, type Page } from '@playwright/test'
import {
  SANDBOX,
  snapshotFrames,
  restoreSnapshot,
  getSavedMasks,
  type ProjectSnapshot,
} from './helpers/maskOps'

let baseline: ProjectSnapshot

test.describe.configure({ mode: 'serial' })

test.beforeAll(async () => {
  const frames = [SANDBOX.anchorFrame, SANDBOX.nonAnchorFrame]
  baseline = await snapshotFrames(frames)
  for (const f of frames) {
    expect(baseline.frames[f].hasMasks, `frame ${f} needs masks`).toBeTruthy()
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

test.describe('mask clear (browser + API, reversible)', () => {
  test('clear object on anchor frame via ObjectCard', async ({ page }) => {
    await openSandbox(page)
    await jumpToFrame(page, SANDBOX.anchorFrame)
    await page.getByText('HeadShave').first().click()
    await page.getByRole('button', { name: 'Clear selection' }).click()
    await page.waitForTimeout(1000)
    const masks = await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.anchorFrame)
    expect(masks.masks[SANDBOX.objA]).toBeUndefined()
    expect(masks.masks[SANDBOX.objB]).toBeTruthy()
    await restoreSnapshot(baseline, [SANDBOX.anchorFrame])
  })

  test('clear object on non-anchor frame', async ({ page }) => {
    await openSandbox(page)
    await jumpToFrame(page, SANDBOX.nonAnchorFrame)
    await page.getByText('HeadShave').first().click()
    await page.getByRole('button', { name: 'Clear selection' }).click()
    await page.waitForTimeout(1000)
    const masks = await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.nonAnchorFrame)
    expect(masks.masks[SANDBOX.objA]).toBeUndefined()
    expect(masks.masks[SANDBOX.objB]).toBeTruthy()
    await restoreSnapshot(baseline, [SANDBOX.nonAnchorFrame])
  })

  test('clear all masks for frame via LeftPanel', async ({ page }) => {
    await openSandbox(page)
    await jumpToFrame(page, SANDBOX.nonAnchorFrame)
    await page.getByTitle(`Clear saved masks for frame ${SANDBOX.nonAnchorFrame}`).click()
    await page.getByRole('button', { name: /This frame only/ }).click()
    await expect(page.getByText('Masks cleared')).toBeVisible({ timeout: 15_000 })
    const masks = await getSavedMasks(SANDBOX.pid, SANDBOX.vid, SANDBOX.nonAnchorFrame)
    expect(masks.hasMasks).toBeFalsy()
    await restoreSnapshot(baseline, [SANDBOX.nonAnchorFrame])
  })
})
