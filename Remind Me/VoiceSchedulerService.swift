import AVFoundation
import Foundation
import Speech

final class VoiceSchedulerService: NSObject, ObservableObject {
    @Published var transcript: String = ""
    @Published var isRecording = false

    private let audioEngine = AVAudioEngine()
    private let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US"))
    private var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    private var recognitionTask: SFSpeechRecognitionTask?

    func requestPermissions() async -> Bool {
        let speechAuthorized = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status == .authorized)
            }
        }

        let micAuthorized = await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { allowed in
                continuation.resume(returning: allowed)
            }
        }

        return speechAuthorized && micAuthorized
    }

    func startRecording() throws {
        stopRecording()

        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: .duckOthers)
        try session.setActive(true, options: .notifyOthersOnDeactivation)

        let request = SFSpeechAudioBufferRecognitionRequest()
        request.shouldReportPartialResults = true
        self.recognitionRequest = request

        let inputNode = audioEngine.inputNode
        let format = inputNode.outputFormat(forBus: 0)
        inputNode.removeTap(onBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            request.append(buffer)
        }

        audioEngine.prepare()
        try audioEngine.start()
        isRecording = true

        recognitionTask = recognizer?.recognitionTask(with: request) { [weak self] result, error in
            guard let self else { return }

            if let result {
                DispatchQueue.main.async {
                    self.transcript = result.bestTranscription.formattedString
                }
            }

            if error != nil || result?.isFinal == true {
                self.stopRecording()
            }
        }
    }

    func stopRecording() {
        guard isRecording else { return }
        audioEngine.stop()
        audioEngine.inputNode.removeTap(onBus: 0)
        recognitionRequest?.endAudio()
        recognitionTask?.cancel()

        recognitionRequest = nil
        recognitionTask = nil
        isRecording = false
    }
}

struct VoiceCommandParser {
    static func parse(_ command: String, now: Date = Date()) -> [TaskReminder] {
        let normalized = command.lowercased().replacingOccurrences(of: "a.m.", with: "am").replacingOccurrences(of: "p.m.", with: "pm")
        let chunks = normalized
            .replacingOccurrences(of: "set a reminder for", with: "")
            .replacingOccurrences(of: "set reminder for", with: "")
            .split(separator: " and ")
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }

        var reminders: [TaskReminder] = []

        for chunk in chunks {
            if let reminder = parseSingleTime(chunk, now: now) {
                reminders.append(reminder)
                continue
            }

            if let ranged = parseTimeRange(chunk, now: now) {
                reminders.append(ranged)
            }
        }

        return reminders
    }

    private static func parseSingleTime(_ input: String, now: Date) -> TaskReminder? {
        let pattern = #"(.+?) at (\d{1,2}(?::\d{2})?\s?(?:am|pm))"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let nsrange = NSRange(input.startIndex..<input.endIndex, in: input)
        guard let match = regex.firstMatch(in: input, range: nsrange),
              let titleRange = Range(match.range(at: 1), in: input),
              let timeRange = Range(match.range(at: 2), in: input) else { return nil }

        let title = String(input[titleRange]).trimmingCharacters(in: .whitespacesAndNewlines)
        let timeText = String(input[timeRange])

        guard let date = nextDate(from: timeText, now: now) else { return nil }
        return TaskReminder(title: title, triggerDate: date, source: .voice)
    }

    private static func parseTimeRange(_ input: String, now: Date) -> TaskReminder? {
        let pattern = #"(.+?) from (\d{1,2}(?::\d{2})?\s?(?:am|pm)) to (\d{1,2}(?::\d{2})?\s?(?:am|pm))"#
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return nil }
        let nsrange = NSRange(input.startIndex..<input.endIndex, in: input)
        guard let match = regex.firstMatch(in: input, range: nsrange),
              let titleRange = Range(match.range(at: 1), in: input),
              let startRange = Range(match.range(at: 2), in: input),
              let endRange = Range(match.range(at: 3), in: input) else { return nil }

        let title = String(input[titleRange]).trimmingCharacters(in: .whitespacesAndNewlines)
        let startText = String(input[startRange])
        let endText = String(input[endRange])

        guard let start = nextDate(from: startText, now: now),
              let end = nextDate(from: endText, now: start) else { return nil }

        return TaskReminder(title: title, triggerDate: start, endDate: end, source: .voice)
    }

    private static func nextDate(from timeText: String, now: Date) -> Date? {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "h:mm a"

        var candidate = timeText.uppercased().replacingOccurrences(of: "AM", with: " AM").replacingOccurrences(of: "PM", with: " PM")
        candidate = candidate.replacingOccurrences(of: "  ", with: " ").trimmingCharacters(in: .whitespaces)

        if !candidate.contains(":") {
            candidate = candidate.replacingOccurrences(of: " AM", with: ":00 AM").replacingOccurrences(of: " PM", with: ":00 PM")
        }

        guard let parsedTime = formatter.date(from: candidate) else { return nil }

        var calendar = Calendar.current
        calendar.timeZone = .current

        let components = calendar.dateComponents([.hour, .minute], from: parsedTime)
        var day = calendar.date(bySettingHour: components.hour ?? 0, minute: components.minute ?? 0, second: 0, of: now)

        if let day, day < now {
            return calendar.date(byAdding: .day, value: 1, to: day)
        }

        return day
    }
}
