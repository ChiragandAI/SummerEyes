# Remind Me (Voice-First Scheduler for iPhone)

## What this is
`Remind Me` is a new iPhone app concept focused on one job: **schedule reminders by voice** and notify the user at the right time.

This is intentionally different from SummerEyes summarization.

## Core problem statement
People should be able to quickly schedule their day without typing. Voice should be the primary input.

## Main flow
1. Tap **Start Voice**.
2. Say a command like:
   - "Set a reminder for waking up at 5 am and practicing vocals from 6 am to 8 am"
3. Tap **Schedule from Voice**.
4. App parses tasks and schedules local notifications.
5. User can fine-tune/add reminders manually with on-screen controls.

## Current implementation
- SwiftUI app shell + reminders list.
- Speech-to-text capture via `Speech` + `AVAudioEngine`.
- Voice command parsing for:
  - "{task} at {time}"
  - "{task} from {start} to {end}"
- Local notification scheduling (`UserNotifications`).
- Manual fallback editor for quick corrections.
- Spoken confirmation feedback via `AVSpeechSynthesizer`.

## Required iOS permissions (`Info.plist`)
Add:
- `NSSpeechRecognitionUsageDescription`
- `NSMicrophoneUsageDescription`

## File map
- `RemindMeApp.swift` — app entry point
- `ContentView.swift` — voice-first UI + manual controls
- `Models.swift` — reminder model
- `VoiceSchedulerService.swift` — speech capture + parser
- `ReminderStore.swift` — reminder state + notification scheduling

## About the repository name
I can scaffold all code locally here as `Remind Me`, but creating an actual new GitHub repository named **Remind Me** must be done from your GitHub account (or with your token/CLI auth).
