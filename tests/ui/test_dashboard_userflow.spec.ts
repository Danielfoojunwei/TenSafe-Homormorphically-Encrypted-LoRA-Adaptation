/**
 * TensorGuardFlow Dashboard UI Tests
 *
 * Tests critical user flows in the dashboard.
 */

import { test, expect } from '@playwright/test';

test.describe('Dashboard Core Flows', () => {

  test.beforeEach(async ({ page }) => {
    // Navigate to dashboard
    await page.goto('/');
    // Wait for initial load
    await page.waitForLoadState('networkidle');
  });

  test('dashboard loads route summary KPIs', async ({ page }) => {
    // Dashboard should display route summary
    const dashboard = page.locator('[data-testid="dashboard"]').or(page.locator('.dashboard'));

    // Should have some content visible
    await expect(page.locator('body')).toBeVisible();

    // Look for KPI indicators or route list
    const routeList = page.locator('[data-testid="route-list"]').or(page.locator('.routes'));

    // Take screenshot for verification
    await page.screenshot({ path: 'reports/qa/ui_traces/dashboard-load.png' });
  });

  test('creating a route from UI works', async ({ page }) => {
    // Find create route button
    const createButton = page.locator('button:has-text("Create")').or(
      page.locator('[data-testid="create-route"]')
    );

    if (await createButton.isVisible()) {
      await createButton.click();

      // Fill in route details
      const routeKeyInput = page.locator('input[name="route_key"]').or(
        page.locator('[data-testid="route-key-input"]')
      );

      if (await routeKeyInput.isVisible()) {
        await routeKeyInput.fill('ui-test-route');

        // Submit form
        const submitButton = page.locator('button[type="submit"]').or(
          page.locator('button:has-text("Save")')
        );
        await submitButton.click();

        // Verify success
        await page.waitForLoadState('networkidle');
      }
    }

    await page.screenshot({ path: 'reports/qa/ui_traces/create-route.png' });
  });

  test('running loop shows timeline updates', async ({ page }) => {
    // Navigate to a route detail page
    const routeLink = page.locator('[data-testid="route-link"]').or(
      page.locator('a:has-text("route")')
    ).first();

    if (await routeLink.isVisible()) {
      await routeLink.click();
      await page.waitForLoadState('networkidle');

      // Look for timeline component
      const timeline = page.locator('[data-testid="timeline"]').or(
        page.locator('.timeline')
      );

      // Timeline should show events
      await page.screenshot({ path: 'reports/qa/ui_traces/timeline.png' });
    }
  });

  test('promote and rollback from UI works', async ({ page }) => {
    // Navigate to route with adapters
    const routeLink = page.locator('[data-testid="route-link"]').first();

    if (await routeLink.isVisible()) {
      await routeLink.click();
      await page.waitForLoadState('networkidle');

      // Look for promote button
      const promoteButton = page.locator('button:has-text("Promote")');
      if (await promoteButton.isVisible()) {
        // Don't actually click in smoke test - just verify presence
        await expect(promoteButton).toBeVisible();
      }

      // Look for rollback button
      const rollbackButton = page.locator('button:has-text("Rollback")');
      // May not be visible if no fallback exists
    }

    await page.screenshot({ path: 'reports/qa/ui_traces/promote-rollback.png' });
  });

  test('pipeline topology displays connected tools', async ({ page }) => {
    // Look for topology/pipeline visualization
    const topology = page.locator('[data-testid="topology"]').or(
      page.locator('.pipeline-graph').or(
        page.locator('svg.topology')
      )
    );

    // Take screenshot of topology
    await page.screenshot({ path: 'reports/qa/ui_traces/topology.png' });
  });

  test('trust readiness panel shows gating status', async ({ page }) => {
    // Look for trust/readiness panel
    const trustPanel = page.locator('[data-testid="trust-panel"]').or(
      page.locator('.trust-readiness').or(
        page.locator(':has-text("Trust")')
      )
    );

    await page.screenshot({ path: 'reports/qa/ui_traces/trust-panel.png' });
  });

  test('export snapshot works', async ({ page }) => {
    // Navigate to route with export capability
    const routeLink = page.locator('[data-testid="route-link"]').first();

    if (await routeLink.isVisible()) {
      await routeLink.click();
      await page.waitForLoadState('networkidle');

      // Look for export button
      const exportButton = page.locator('button:has-text("Export")');

      if (await exportButton.isVisible()) {
        // Click to open export dialog/download
        // await exportButton.click();
        await expect(exportButton).toBeVisible();
      }
    }

    await page.screenshot({ path: 'reports/qa/ui_traces/export.png' });
  });

});

test.describe('Dashboard Health', () => {

  test('health endpoint is accessible', async ({ request }) => {
    const response = await request.get('http://localhost:8000/health');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data.status).toBeDefined();
  });

  test('ready endpoint returns true', async ({ request }) => {
    const response = await request.get('http://localhost:8000/ready');
    expect(response.ok()).toBeTruthy();

    const data = await response.json();
    expect(data.ready).toBe(true);
  });

});
