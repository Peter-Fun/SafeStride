import LiveKit

/// A minimalistic dependency injection container.
/// It allows sharing common dependencies e.g. `Room` between view models and services.
@MainActor
final class Dependencies {
    static let shared = Dependencies()

    private init() {}

    lazy var room = Room(roomOptions: RoomOptions(defaultScreenShareCaptureOptions: ScreenShareCaptureOptions(useBroadcastExtension: true)))

    lazy var tokenService = TokenService()

    private lazy var localMessageSender = LocalMessageSender(room: room)
    lazy var messageSenders: [any MessageSender] = [
        localMessageSender,
    ]
    lazy var messageReceivers: [any MessageReceiver] = [
        TranscriptionStreamReceiver(room: room),
        localMessageSender,
    ]

    lazy var errorHandler: (Error?) -> Void = { _ in }
}

@MainActor
@propertyWrapper
struct Dependency<T> {
    let keyPath: KeyPath<Dependencies, T>

    init(_ keyPath: KeyPath<Dependencies, T>) {
        self.keyPath = keyPath
    }

    var wrappedValue: T {
        Dependencies.shared[keyPath: keyPath]
    }
}
