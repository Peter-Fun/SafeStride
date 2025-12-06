import SwiftUI

/// A small status indicator showing location streaming connection status
struct LocationStreamingStatusView: View {
    @Environment(AppViewModel.self) private var viewModel
    
    var body: some View {
        if viewModel.isLocationStreamingEnabled {
            HStack(spacing: 8) {
                Circle()
                    .fill(connectionColor)
                    .frame(width: 8, height: 8)
                    .animation(.easeInOut(duration: 0.5), value: connectionColor)
                
                Text("Location: \(viewModel.locationConnectionStatus)")
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
        switch viewModel.locationConnectionStatus {
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
    LocationStreamingStatusView()
        .environment(AppViewModel())
}
