import SwiftUI
import WebKit

struct ContentView: View {
    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                Text("SummerEyes")
                    .font(.largeTitle)
                    .fontWeight(.bold)

                Text("Use the Streamlit app as a native iPhone experience.")
                    .font(.subheadline)
                    .multilineTextAlignment(.center)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)

                WebContainerView(urlString: "https://summereyes.streamlit.app")
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .overlay(
                        RoundedRectangle(cornerRadius: 12)
                            .stroke(Color.gray.opacity(0.2), lineWidth: 1)
                    )
            }
            .padding()
            .navigationTitle("SummerEyes")
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

struct WebContainerView: UIViewRepresentable {
    let urlString: String

    func makeUIView(context: Context) -> WKWebView {
        let config = WKWebViewConfiguration()
        config.allowsInlineMediaPlayback = true

        let webView = WKWebView(frame: .zero, configuration: config)
        webView.scrollView.keyboardDismissMode = .onDrag

        if let url = URL(string: urlString) {
            webView.load(URLRequest(url: url))
        }

        return webView
    }

    func updateUIView(_ webView: WKWebView, context: Context) {
        // No-op for static URL.
    }
}

#Preview {
    ContentView()
}
