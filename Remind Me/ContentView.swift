import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var store: ReminderStore
    @StateObject private var voiceService = VoiceSchedulerService()

    @State private var manualTitle = ""
    @State private var manualDate = Date()
    @State private var statusText = "Tap Start Voice and say your schedule."

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                GroupBox("Voice-first scheduling") {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Example: Set a reminder for waking up at 5 am and practicing vocals from 6 am to 8 am.")
                            .font(.footnote)
                            .foregroundStyle(.secondary)

                        Text(voiceService.transcript.isEmpty ? "Transcript appears here..." : voiceService.transcript)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(8)
                            .background(.thinMaterial)
                            .clipShape(RoundedRectangle(cornerRadius: 8))

                        HStack {
                            Button(voiceService.isRecording ? "Stop Voice" : "Start Voice") {
                                toggleVoiceCapture()
                            }
                            .buttonStyle(.borderedProminent)

                            Button("Schedule from Voice") {
                                store.addVoiceCommand(voiceService.transcript)
                                statusText = "Voice reminders scheduled."
                            }
                            .buttonStyle(.bordered)
                            .disabled(voiceService.transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                        }

                        Text(statusText)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                GroupBox("Manual edit / add") {
                    VStack(spacing: 10) {
                        TextField("Task title", text: $manualTitle)
                            .textFieldStyle(.roundedBorder)

                        DatePicker("Remind me at", selection: $manualDate)

                        Button("Add Manual Reminder") {
                            let title = manualTitle.trimmingCharacters(in: .whitespacesAndNewlines)
                            guard !title.isEmpty else { return }
                            store.addManualReminder(title: title, date: manualDate)
                            manualTitle = ""
                            statusText = "Manual reminder added."
                        }
                        .buttonStyle(.bordered)
                    }
                }

                List {
                    Section("Upcoming reminders") {
                        if store.reminders.isEmpty {
                            Text("No reminders yet.")
                                .foregroundStyle(.secondary)
                        } else {
                            ForEach(store.reminders) { reminder in
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(reminder.title)
                                        .fontWeight(.semibold)
                                    Text(reminderTimeText(reminder))
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    Text(reminder.source == .voice ? "Voice scheduled" : "Manual")
                                        .font(.caption2)
                                }
                            }
                            .onDelete { indexSet in
                                for index in indexSet {
                                    store.removeReminder(store.reminders[index])
                                }
                            }
                        }
                    }
                }
                .listStyle(.insetGrouped)
            }
            .padding()
            .navigationTitle("Remind Me")
        }
    }

    private func reminderTimeText(_ reminder: TaskReminder) -> String {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short

        if let end = reminder.endDate {
            return "\(formatter.string(from: reminder.triggerDate)) - \(formatter.string(from: end))"
        }

        return formatter.string(from: reminder.triggerDate)
    }

    private func toggleVoiceCapture() {
        Task {
            if voiceService.isRecording {
                voiceService.stopRecording()
                statusText = "Voice recording stopped."
                return
            }

            let allowed = await voiceService.requestPermissions()
            guard allowed else {
                statusText = "Please enable Speech + Microphone permissions in Settings."
                return
            }

            do {
                try voiceService.startRecording()
                statusText = "Listening..."
            } catch {
                statusText = "Couldn't start recording: \(error.localizedDescription)"
            }
        }
    }
}

#Preview {
    ContentView()
        .environmentObject(ReminderStore())
}
