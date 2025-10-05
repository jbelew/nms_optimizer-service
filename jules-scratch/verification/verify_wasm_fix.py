from playwright.sync_api import sync_playwright, expect
import os
import sys

def run_verification(playwright):
    browser = playwright.chromium.launch(headless=True)
    context = browser.new_context()
    page = context.new_page()

    try:
        # Load the local HTML file directly
        page.goto("http://127.0.0.1:8000/index.html")

        # Wait for the page to load and the ship selector to be available
        ship_selector = page.locator("#platform-selector")

        try:
            expect(ship_selector).to_be_visible(timeout=10000) # Increased timeout
        except Exception as e:
            print("Error: Failed to find selector '#platform-selector'.", file=sys.stderr)
            print("Page Title:", page.title(), file=sys.stderr)
            print("Page URL:", page.url, file=sys.stderr)
            print("\n--- Page Content ---", file=sys.stderr)
            print(page.content(), file=sys.stderr)
            print("--- End Page Content ---\n", file=sys.stderr)
            raise e

        ship_selector.select_option("sentinel")

        # Wait for the tech selector to be available and select the tech
        tech_selector = page.locator("#tech-selector")
        expect(tech_selector).to_be_visible()
        tech_selector.select_option("infra")

        # Wait for the grid to be populated
        expect(page.locator(".grid-cell.active")).to_have_count(60)

        # Find and click the optimize button
        optimize_button = page.get_by_role("button", name="Optimize")
        expect(optimize_button).to_be_enabled()
        optimize_button.click()

        # Wait for the optimization to complete and the score to be displayed.
        # The score should be close to the one we saw in the tests.
        score_element = page.locator("#total-bonus-value")
        expect(score_element).to_contain_text("2.5", timeout=30000)

        # Take a screenshot of the final state
        page.screenshot(path="jules-scratch/verification/verification.png")

    finally:
        browser.close()

with sync_playwright() as playwright:
    run_verification(playwright)