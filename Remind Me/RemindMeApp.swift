import SwiftUI

@main
struct RemindMeApp: App {
    @StateObject private var store = ReminderStore()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(store)
        }
    }
}
