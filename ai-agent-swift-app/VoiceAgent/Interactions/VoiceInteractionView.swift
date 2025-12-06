import SwiftUI

/// A multiplatform view that shows voice-specific interaction controls.
///
/// Depending on the track availability, the view will show:
/// - agent participant view
/// - local participant camera preview
/// - local participant screen share preview
struct VoiceInteractionView: View {
    @Environment(AppViewModel.self) private var viewModel
    @Environment(\.horizontalSizeClass) private var horizontalSizeClass

    var body: some View {
        if horizontalSizeClass == .regular {
            regular()
        } else {
            compact()
        }
    }

    @ViewBuilder
    private func regular() -> some View {
        HStack {
            Spacer()
                .frame(width: 50 * .grid)
            AgentParticipantView()
            VStack {
                Spacer()
                LocalParticipantView()
            }
            .frame(width: 50 * .grid)
        }
        .safeAreaPadding()
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
    
    @ViewBuilder
    private func compact() -> some View {
        ZStack {
            VStack(spacing: 0) {
                AgentParticipantView()
                    .frame(
                        width: viewModel.isCameraEnabled ? 60 : nil,
                        height: viewModel.isCameraEnabled ? 60 : nil
                    )
                    .frame(maxWidth: .infinity, maxHeight: viewModel.isCameraEnabled ? nil : .infinity)
                    .padding(.top, viewModel.isCameraEnabled ? 16 : 0)
                    .clipShape(RoundedRectangle(cornerRadius: viewModel.isCameraEnabled ? 30 : 0))
                    .shadow(radius: 10)
                    .ignoresSafeArea(edges: viewModel.isCameraEnabled ? [] : .all)
                    .animation(.snappy, value: viewModel.isCameraEnabled)

                if viewModel.isCameraEnabled {
                    Spacer()

                    LocalParticipantView()
                        .aspectRatio(9/16, contentMode: .fit)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                        .shadow(radius: 10)
                        .padding(.horizontal)
                        .padding(.bottom, 40)
                        .animation(.snappy, value: viewModel.isCameraEnabled)
                }
            }
        }

    }
}


