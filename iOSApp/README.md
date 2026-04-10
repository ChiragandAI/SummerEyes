# SummerEyes iPhone App (SwiftUI Wrapper)

This folder contains a simple iOS app you can run on your iPhone using Xcode.

## What this app does
- Launches a native SwiftUI app shell.
- Loads `https://summereyes.streamlit.app` inside a `WKWebView`.
- Gives you a phone-friendly, app-like experience.

## Quick setup (about 5 minutes)
1. Open **Xcode** on your Mac.
2. Create a new iOS App project named `SummerEyesiOS`.
3. Replace the generated `ContentView.swift` and `SummerEyesiOSApp.swift` files with the files in this folder.
4. Connect your iPhone with a cable (or wireless debugging).
5. In Xcode, choose your iPhone as the run target.
6. In **Signing & Capabilities**, choose your Apple Team.
7. Press **Run**.

## Recommended iOS permissions
If you want voice upload/record features in web content to work reliably, add the following keys to your app's `Info.plist`:

- `NSMicrophoneUsageDescription` → "SummerEyes needs microphone access for voice summarization."
- `NSCameraUsageDescription` → "SummerEyes needs camera access for scanning and uploads." (optional)

## Optional enhancements
- Add pull-to-refresh support.
- Add offline state handling.
- Add a loading indicator.
- Route specific links to Safari.
