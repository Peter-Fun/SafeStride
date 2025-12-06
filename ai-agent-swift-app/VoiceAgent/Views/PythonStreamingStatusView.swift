import SwiftUI

/// A small status indicator showing Python streaming connection status
struct PythonStreamingStatusView: View {
    @Environment(AppViewModel.self) private var viewModel
    
    var body: some View {
        if viewModel.isPythonStreamingEnabled {
            HStack(spacing: 8) {
                Circle()
                    .fill(connectionColor)
                    .frame(width: 8, height: 8)
                    .animation(.easeInOut(duration: 0.5), value: connectionColor)
                
                Text("Python: \(viewModel.pythonConnectionStatus)")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(Color.black.opacity(0.6))
            .cornerRadius(12)
            .transition(.opacity)
        }
    }
    
    private var connectionColor: Color {
        switch viewModel.pythonConnectionStatus {
        case "Connected":
            return .green
        case "Disconnected":
            return .gray
        case let status where status.contains("Failed") || status.contains("Error"):
            return .red
        default:
            return .yellow
        }
    }
}

#Preview {
    PythonStreamingStatusView()
        .environment(AppViewModel())
}
