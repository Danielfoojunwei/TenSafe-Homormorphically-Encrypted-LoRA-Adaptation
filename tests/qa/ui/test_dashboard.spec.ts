
import { test, expect } from '@playwright/test';

test('Dashboard loads and shows routes', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/TensorGuard/);
    await expect(page.locator('text=Continuous Learning Console')).toBeVisible();
});

test('Create Route Wizard opens', async ({ page }) => {
    await page.goto('/');
    await page.click('text=New Route');

    // Verify modal appears with correct header
    await expect(page.locator('text=Create Route').first()).toBeVisible({ timeout: 5000 });
    // Verify step indicator is visible
    await expect(page.locator('text=1. Route')).toBeVisible();
    // Verify input field is present
    await expect(page.locator('input[placeholder*="customer-support"]')).toBeVisible();
});
