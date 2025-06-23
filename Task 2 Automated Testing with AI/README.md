# Practice Test Automation: Login Test Suite

## Overview

This project demonstrates a basic automated test suite for the login functionality on the [Practice Test Automation](https://practicetestautomation.com/practice-test-login/) website. It uses Selenium IDE to create and run tests that cover both successful and unsuccessful login attempts, serving as a baseline for understanding test automation.

## How to Run the Tests

To execute the tests, you will need the Selenium IDE browser extension.

**Prerequisites:**
*   A web browser like Google Chrome or Mozilla Firefox.
*   The [Selenium IDE extension](https://www.selenium.dev/selenium-ide/docs/en/introduction/getting-started) installed in your browser.

**Execution Steps:**
1.  Open the Selenium IDE extension in your browser.
2.  Click on **Open an existing project**.
3.  Navigate to this project's directory and select the `PracticeTestLogin.side` file.
4.  With the project loaded, click the **Run all tests** button in the toolbar.
5.  Selenium IDE will then execute the tests in your browser, and you can watch the automation in real-time.

## Test Scenarios

The `PracticeTestLogin.side` file contains a test suite with two primary scenarios:

*   **Valid Login Test:** Navigates to the login page, enters correct credentials (`username: student`, `password: Password123`), submits the form, and verifies that the user is logged in successfully by checking for a "Log out" button and a success message.
*   **Invalid Login Test:** Navigates to the login page, enters incorrect credentials, submits the form, and asserts that an appropriate error message is displayed to the user.

### Test Outcomes

Below are screenshots of the outcomes for each test case.

**Valid Login Test Outcome:**

![Valid Test](./Valid%20Test.png)

**Invalid Login Test Outcome:**

![Invalid Test](./Invalid%20Test.png)

## AI-Enhanced Test Coverage vs. Manual Testing

The provided Selenium test file, `PracticeTestLogin.side`, demonstrates basic manual test automation for a login page. It includes two static scenarios: one for a valid login and one for an invalid login. While this is a useful start, this approach offers limited test coverage on its own.

AI can significantly improve this. Instead of just testing with "student" and "wronguser," an AI-powered tool could generate hundreds of varied username and password combinations. This includes edge cases like special characters, long strings, or even common SQL injection attempts, which are often missed. Furthermore, AI-driven exploratory testing can discover and test user flows that a manual tester might not consider, such as unexpected navigation paths or unusual interactions on the page. By autonomously generating a diverse range of test data and scenarios, AI uncovers a much wider array of potential bugs, providing far more comprehensive coverage than what is demonstrated in the simple, two-path manual test script.

## File Descriptions

*   **`PracticeTestLogin.side`**: The Selenium IDE project file containing the automated test suite.
*   **`Automation Test.mp4`**: A video recording of the automated test execution.
*   **`Valid Test.png`**: A screenshot showing the successful login page.
*   **`Invalid Test.png`**: A screenshot showing the error message for an invalid login attempt.
*   **`README.md`**: This file, providing an overview and instructions for the project. 