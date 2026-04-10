import AVFoundation
import Foundation
import UserNotifications

@MainActor
final class ReminderStore: ObservableObject {
    @Published var reminders: [TaskReminder] = []

    private let synthesizer = AVSpeechSynthesizer()

    init() {
        Task {
            await requestNotificationPermission()
        }
    }

    func addManualReminder(title: String, date: Date) {
        let reminder = TaskReminder(title: title, triggerDate: date, source: .manual)
        add(reminder)
    }

    func addVoiceCommand(_ transcript: String) {
        let parsed = VoiceCommandParser.parse(transcript)
        guard !parsed.isEmpty else {
            speak("I couldn't understand that schedule. Try saying: set reminder for gym at 6 am")
            return
        }

        parsed.forEach(add)

        let lines = parsed.map { formatConfirmation($0) }.joined(separator: ". ")
        speak("Scheduled. \(lines)")
    }

    func removeReminder(_ reminder: TaskReminder) {
        reminders.removeAll { $0.id == reminder.id }
        UNUserNotificationCenter.current().removePendingNotificationRequests(withIdentifiers: [reminder.id.uuidString])
    }

    private func add(_ reminder: TaskReminder) {
        reminders.append(reminder)
        reminders.sort { $0.triggerDate < $1.triggerDate }
        scheduleNotification(for: reminder)
    }

    private func requestNotificationPermission() async {
        _ = try? await UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge])
    }

    private func scheduleNotification(for reminder: TaskReminder) {
        let content = UNMutableNotificationContent()
        content.title = "Reminder"
        content.body = reminder.title
        content.sound = .default

        let components = Calendar.current.dateComponents([.year, .month, .day, .hour, .minute], from: reminder.triggerDate)
        let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: false)
        let request = UNNotificationRequest(identifier: reminder.id.uuidString, content: content, trigger: trigger)

        UNUserNotificationCenter.current().add(request)
    }

    private func formatConfirmation(_ reminder: TaskReminder) -> String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short

        if let endDate = reminder.endDate {
            return "\(reminder.title) from \(formatter.string(from: reminder.triggerDate)) to \(formatter.string(from: endDate))"
        }

        return "\(reminder.title) at \(formatter.string(from: reminder.triggerDate))"
    }

    private func speak(_ message: String) {
        let utterance = AVSpeechUtterance(string: message)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-US")
        synthesizer.speak(utterance)
    }
}
