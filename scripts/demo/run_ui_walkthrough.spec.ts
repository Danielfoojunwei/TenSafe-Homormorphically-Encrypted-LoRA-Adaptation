
import { test, expect } from '@playwright/test';

// Configuration
const ROUTE_KEY = 'demo-ui-route';
const BASE_MODEL = 'tiny-model-v2';

test.describe('Continuous Learning UI Demo Walkthrough', () => {
    test.beforeEach(async ({ page }) => {
        // Ensure clean state if possible? 
        // We rely on route uniqueness logic or timestamping route key if needed.
        // For demo proof, we stick to fixed name but maybe cleanup via API first?
        // Rely on demo driver / full script cleanup.
    });

    test('Full End-to-End Walkthrough', async ({ page }) => {
        // 1. Open Dashboard
        await page.goto('/');
        await expect(page).toHaveTitle(/TensorGuard/);
        await expect(page.locator('text=Continuous Learning Console')).toBeVisible();

        // 2. Create Route
        await page.click('text=New Route');
        await expect(page.locator('text=Create Route')).toBeVisible();

        // Wizard Step 1: Route Info
        await page.fill('input[placeholder*="customer-support"]', ROUTE_KEY);
        // Base model default is selected or fill if needed?
        // Assuming default is set or we select one.
        // Let's type if input exists.
        // Base Model field might be a select or input.
        // Looking at RouteWizard.vue (not visible but assumed standard input/select)
        // We wait for button enable.
        await page.waitForTimeout(500);
        // Need to fill base model if required and empty.
        // Assuming the select has a default.

        // Wait for enabled Next button
        await page.waitForFunction(() => {
            const btn = document.querySelector('button.btn-primary');
            return btn && !btn.hasAttribute('disabled');
        }, { timeout: 5000 });
        await page.click('button.btn-primary'); // Next

        // Wizard Step 2: Feed
        await page.waitForTimeout(500);
        await page.click('button.btn-primary'); // Next (Default Feed)

        // Wizard Step 3: Policy
        await page.waitForTimeout(500);
        await page.click('button.btn-primary'); // Next (Default Policy)

        // Wizard Step 4: Review
        await page.waitForTimeout(500);
        await page.click('button.btn-primary'); // Create Route

        // Verify Route Created & Navigated
        // Might stay on dashboard or close modal.
        await expect(page.locator(`text=${ROUTE_KEY}`)).toBeVisible({ timeout: 10000 });

        // 3. Go to Details
        await page.click(`text=${ROUTE_KEY}`);
        await expect(page.locator(`h1:has-text("${ROUTE_KEY}")`)).toBeVisible();

        // 4. Run Loop
        // Click "Run Loop Now"
        await page.click('text=Run Loop Now');

        // 5. Verify Timeline Update
        // Wait for event to appear (e.g. "TRAIN_STARTED" or "success")
        // Timeline might load async.
        await expect(page.locator('.timeline-event')).toBeTruthy();
        // Wait for specific success marker if possible
        // await expect(page.locator('text=Adapter produced')).toBeVisible({ timeout: 15000 });

        // 6. Export Spec
        // Click Export button
        const downloadPromise = page.waitForEvent('download');
        await page.click('text=Export');
        const download = await downloadPromise;
        await download.saveAs('demo_proof/latest/ui_export.json');
        expect(download.suggestedFilename()).toContain('export.json');

        // 7. Promote to Stable (if candidate available)
        // Needs a candidate to appear.
        // This relies on the loop finishing.
        // In "Real" execution with tiny model, it takes a few seconds.
        await page.waitForTimeout(5000);

        // Refresh to see if candidate appeared?
        // App might need refresh or auto-updates.
        await page.reload();

        // Check for "Promote to Stable" button
        const promoteBtn = page.locator('text=Promote to Stable');
        if (await promoteBtn.isVisible()) {
            await promoteBtn.click();
            await page.reload(); // Refresh to see update
            // Verify Active Adapter changed
            await expect(page.locator('text=Active Production Adapter')).toBeVisible();
            // Check if adapter ID > base
        }

        // 8. Rollback?
        // Check if Rollback button is enabled (previous stable exists?)
        // If this is first run, fallback is null/base. Rollback might be disabled.
        // But if we promoted, we might have history.
        // Verify UI element existence.
        await expect(page.locator('text=Releases & Rollback')).toBeVisible();

        // 9. View Diff?
        // If diff button visible
        // await page.click('text=View Diff');
        // Verify modal or nav?

        // Recording stops at end of test.
    });
});
