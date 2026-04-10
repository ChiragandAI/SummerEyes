import Foundation

struct TaskReminder: Identifiable, Codable, Hashable {
    let id: UUID
    var title: String
    var triggerDate: Date
    var endDate: Date?
    var source: ReminderSource

    init(id: UUID = UUID(), title: String, triggerDate: Date, endDate: Date? = nil, source: ReminderSource) {
        self.id = id
        self.title = title
        self.triggerDate = triggerDate
        self.endDate = endDate
        self.source = source
    }
}

enum ReminderSource: String, Codable {
    case voice
    case manual
}
