@preconcurrency import AVFoundation
import Combine
import LiveKit
import Observation
import Network
import CoreImage
import CoreLocation
#if os(iOS)
import UIKit
#endif

@MainActor
@Observable
final class AppViewModel: NSObject, CLLocationManagerDelegate {
    enum Error: LocalizedError {
        case agentNotConnected

        var errorDescription: String? {
            switch self {
            case .agentNotConnected:
                "Agent did not connect to the Room"
            }
        }
    }

    enum InteractionMode {
        case voice
        case text
    }

    let agentFeatures: AgentFeatures
    let agentConnectionTimeout: TimeInterval = 10

    private(set) var connectionState: ConnectionState = .disconnected
    private(set) var isListening = false
    var isInteractive: Bool {
        switch connectionState {
        case .disconnected where isListening,
             .connecting where isListening,
             .connected,
             .reconnecting:
            true
        default:
            false
        }
    }

    private(set) var agent: Participant?
    

    private(set) var interactionMode: InteractionMode = .voice

    private(set) var isMicrophoneEnabled = false
    private(set) var audioTrack: (any AudioTrack)?

    private(set) var isCameraEnabled = false
    private(set) var cameraTrack: (any VideoTrack)?

    private(set) var isScreenShareEnabled = false
    private(set) var screenShareTrack: (any VideoTrack)?

    private(set) var agentAudioTrack: (any AudioTrack)?
    private(set) var avatarCameraTrack: (any VideoTrack)?

    private(set) var audioDevices: [AudioDevice] = AudioManager.shared.inputDevices
    private(set) var selectedAudioDeviceID: String = AudioManager.shared.inputDevice.deviceId

    private(set) var videoDevices: [AVCaptureDevice] = []
    private(set) var selectedVideoDeviceID: String?

    private(set) var canSwitchCamera = false

    @ObservationIgnored
    private var videoConnection: NWConnection?
    @ObservationIgnored
    private var networkMonitor: NWPathMonitor?
    
    private var videoConnectionRetryCount = 0
    private let maxRetryCount = 5
    
    private let macIPAddress = "192.168.1.20" 
    private let videoPort = NWEndpoint.Port(integerLiteral: 6001)
    
    private var lastFrameSentTime = Date().addingTimeInterval(-1)
    private let frameRateLimit: TimeInterval = 1.0/15.0
    
    private var framesSentCount: UInt64 = 0
    private var lastFrameCountResetTime = Date()
    private var bytesTotalSent: UInt64 = 0
    
    private(set) var isPythonStreamingEnabled = false
    private(set) var pythonConnectionStatus = "Disconnected"
    
    private(set) var framesSentPerSecond: Double = 0.0
    private(set) var totalFramesSent: UInt64 = 0
    private(set) var totalBytesSent: UInt64 = 0

    private var pythonRenderer: PythonStreamingRenderer?

    private let locationManager = CLLocationManager()
    private(set) var currentLocation: CLLocation?
    private(set) var currentHeading: CLHeading?
    private var lastLocationSentTime = Date().addingTimeInterval(-10)

    @ObservationIgnored
    private var locationConnection: NWConnection?
    private let locationPort = NWEndpoint.Port(integerLiteral: 6000)
    
    private var locationConnectionRetryCount = 0
    private var isLocationConnecting = false
    private var lastLocationConnectionAttemptTime = Date()
    
    private var pendingLocationUpdates = [(latitude: Double, longitude: Double, heading: Double)]()
    private let maxPendingUpdates = 5
    @ObservationIgnored
    private var queueProcessTimer: Timer?
    
    private(set) var isLocationStreamingEnabled = false
    private(set) var locationConnectionStatus = "Disconnected"
    private(set) var locationPermissionStatus = "Unknown"

    private(set) var totalLocationsSent: UInt64 = 0
    private(set) var currentLatitude: Double = 0.0
    private(set) var currentLongitude: Double = 0.0
    private(set) var currentHeadingValue: Double = 0.0

    @ObservationIgnored
    @Dependency(\.room) private var room
    @ObservationIgnored
    @Dependency(\.tokenService) private var tokenService
    @ObservationIgnored
    @Dependency(\.errorHandler) private var errorHandler

    init(agentFeatures: AgentFeatures = .current) {
        self.agentFeatures = agentFeatures
        super.init()

        observeRoom()
        observeDevices()
        setupLocationServices()
        
        print("Target server: \(macIPAddress):\(videoPort)")
        print("Location server: \(macIPAddress):\(locationPort)")
        print("Agent features: \(agentFeatures)")
    }

    private func observeRoom() {
        Task { [weak self] in
            guard let changes = self?.room.changes else { return }
            for await _ in changes {
                guard let self else { return }

                connectionState = room.connectionState
                agent = room.agentParticipant

                isMicrophoneEnabled = room.localParticipant.isMicrophoneEnabled()
                audioTrack = room.localParticipant.firstAudioTrack
                isCameraEnabled = room.localParticipant.isCameraEnabled()
                cameraTrack = room.localParticipant.firstCameraVideoTrack
                isScreenShareEnabled = room.localParticipant.isScreenShareEnabled()
                screenShareTrack = room.localParticipant.firstScreenShareVideoTrack

                agentAudioTrack = room.agentParticipant?.audioTracks
                    .first(where: { $0.source == .microphone })?.track as? AudioTrack
                avatarCameraTrack = room.agentParticipant?.avatarWorker?.firstCameraVideoTrack
            }
        }
    }

    private func observeDevices() {
        do {
            try AudioManager.shared.set(microphoneMuteMode: .inputMixer) /// don't play mute sound effect
            try AudioManager.shared.setRecordingAlwaysPreparedMode(true)
        } catch {
            errorHandler(error)
        }

        AudioManager.shared.onDeviceUpdate = { [weak self] _ in
            Task { @MainActor in
                self?.audioDevices = AudioManager.shared.inputDevices
                self?.selectedAudioDeviceID = AudioManager.shared.defaultInputDevice.deviceId
            }
        }

        Task {
            do {
                canSwitchCamera = try await CameraCapturer.canSwitchPosition()
                videoDevices = try await CameraCapturer.captureDevices()
                if let backCamera = videoDevices.first(where: { $0.position == .back }) {
                    selectedVideoDeviceID = backCamera.uniqueID
                } else {
                    selectedVideoDeviceID = videoDevices.first?.uniqueID
                }
                
            } catch {
                errorHandler(error)
            }
        }
    }

    deinit {
        AudioManager.shared.onDeviceUpdate = nil
        locationConnection?.cancel()
        videoConnection?.cancel()
        networkMonitor?.cancel()
    }

    private func resetState() {
        isListening = false
        interactionMode = .voice
    }

    func connect() async {
        errorHandler(nil)
        resetState()
        do {
            if agentFeatures.contains(.voice) {
                try await connectWithVoice()
            } else {
                try await connectWithoutVoice()
            }

            try await checkAgentConnected()
        } catch {
            errorHandler(error)
            resetState()
        }
    }

    private func connectWithVoice() async throws {
        try await room.withPreConnectAudio {
            await MainActor.run { self.isListening = true }

            let connectionDetails = try await self.getConnection()

            try await self.room.connect(
                url: connectionDetails.serverUrl,
                token: connectionDetails.participantToken,
                connectOptions: .init(enableMicrophone: true)
            )
        }
    }

    private func connectWithoutVoice() async throws {
        let connectionDetails = try await getConnection()

        try await room.connect(
            url: connectionDetails.serverUrl,
            token: connectionDetails.participantToken,
            connectOptions: .init(enableMicrophone: false)
        )
    }

    private func getConnection() async throws -> TokenService.ConnectionDetails {
        let roomName = "room-\(Int.random(in: 1000 ... 9999))"
        let participantName = "user-\(Int.random(in: 1000 ... 9999))"

        return try await tokenService.fetchConnectionDetails(
            roomName: roomName,
            participantName: participantName
        )!
    }

    func disconnect() async {
        await room.disconnect()
        resetState()
    }

    private func checkAgentConnected() async throws {
        try await Task.sleep(for: .seconds(agentConnectionTimeout))
        if connectionState == .connected, agent == nil {
            await disconnect()
            throw Error.agentNotConnected
        }
    }

    func toggleTextInput() {
        switch interactionMode {
        case .voice:
            interactionMode = .text
        case .text:
            interactionMode = .voice
        }
    }

    func toggleMicrophone() async {
        do {
            try await room.localParticipant.setMicrophone(enabled: !isMicrophoneEnabled)
        } catch {
            errorHandler(error)
        }
    }

    func toggleCamera() async {
        let enable = !isCameraEnabled
        do {
            if enable, isScreenShareEnabled {
                try await room.localParticipant.setScreenShare(enabled: false)
            }

            let device = try await CameraCapturer.captureDevices().first(where: { $0.uniqueID == selectedVideoDeviceID })
            try await room.localParticipant.setCamera(enabled: enable, captureOptions: CameraCaptureOptions(device: device))
            
            if enable && isPythonStreamingEnabled {
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                    self?.addPythonRenderer()
                }
            } else if !enable {
                removePythonRenderer()
            }
        } catch {
            errorHandler(error)
        }
    }

    func toggleScreenShare() async {
        let enable = !isScreenShareEnabled
        do {
            if enable, isCameraEnabled {
                try await room.localParticipant.setCamera(enabled: false)
            }
            try await room.localParticipant.setScreenShare(enabled: enable)
        } catch {
            errorHandler(error)
        }
    }

    #if os(macOS)
    func select(audioDevice: AudioDevice) {
        selectedAudioDeviceID = audioDevice.deviceId

        let device = AudioManager.shared.inputDevices.first(where: { $0.deviceId == selectedAudioDeviceID }) ?? AudioManager.shared.defaultInputDevice
        AudioManager.shared.inputDevice = device
    }

    func select(videoDevice: AVCaptureDevice) async {
        selectedVideoDeviceID = videoDevice.uniqueID

        guard let cameraCapturer = getCameraCapturer() else { return }
        do {
            let captureOptions = CameraCaptureOptions(device: videoDevice)
            try await cameraCapturer.set(options: captureOptions)
        } catch {
            errorHandler(error)
        }
    }
    #endif

    func switchCamera() async {
        guard let cameraCapturer = getCameraCapturer() else { return }
        do {
            try await cameraCapturer.switchCameraPosition()
        } catch {
            errorHandler(error)
        }
    }

    private func getCameraCapturer() -> CameraCapturer? {
        guard let cameraTrack = cameraTrack as? LocalVideoTrack else { return nil }
        return cameraTrack.capturer as? CameraCapturer
    }
    
    func togglePythonStreaming() {
        if isPythonStreamingEnabled {
            stopPythonStreaming()
        } else {
            startPythonStreaming()
        }
    }
    
    private func startPythonStreaming() {
        print("Camera enabled: \(isCameraEnabled)")
        print("Target server: \(macIPAddress):\(videoPort)")

        setupPythonVideoConnection()
        isPythonStreamingEnabled = true
        setupNetworkMonitoring()
        
        if isCameraEnabled {
            addPythonRenderer()
        }
        
        print("Python streaming started")
    }
    
    private func stopPythonStreaming() {
        isPythonStreamingEnabled = false
        
        removePythonRenderer()
        
        videoConnection?.cancel()
        videoConnection = nil
        pythonConnectionStatus = "Disconnected"
        
        resetDebugCounters()
        
        print("Python streaming stopped")
    }
    
    private func resetDebugCounters() {
        framesSentCount = 0
        bytesTotalSent = 0
        framesSentPerSecond = 0.0
        totalFramesSent = 0
        totalBytesSent = 0
        lastFrameCountResetTime = Date()
    }
    
    private func trackFrameSendingMetrics(imageData: Data) {
        framesSentCount += 1
        totalFramesSent += 1
        bytesTotalSent += UInt64(imageData.count)
        totalBytesSent += UInt64(imageData.count)
        
        let now = Date()
        let timeSinceReset = now.timeIntervalSince(lastFrameCountResetTime)
        
        if timeSinceReset >= 1.0 {
            framesSentPerSecond = Double(framesSentCount) / timeSinceReset
            framesSentCount = 0
            lastFrameCountResetTime = now
        }
    }
    
    private func updateDebugMetrics(frameSize: Int) {
        framesSentCount += 1
        totalFramesSent += 1
        bytesTotalSent += UInt64(frameSize)
        totalBytesSent += UInt64(frameSize)
        
        let now = Date()
        let timeSinceReset = now.timeIntervalSince(lastFrameCountResetTime)
        
        if timeSinceReset >= 1.0 {
            framesSentPerSecond = Double(framesSentCount) / timeSinceReset
            framesSentCount = 0
            lastFrameCountResetTime = now
        }
    }
    
    private func addPythonRenderer() {
        guard let localVideoTrack = cameraTrack as? LocalVideoTrack else { 
            print("No renderer track available")
            return 
        }

        removePythonRenderer()
        
        pythonRenderer = PythonStreamingRenderer { [weak self] pixelBuffer in
            Task { @MainActor in
                guard let self else { return }
                if self.isPythonStreamingEnabled {
                    self.processCameraFrame(pixelBuffer)
                }
            }
        }
        
        if let renderer = pythonRenderer {
            localVideoTrack.add(videoRenderer: renderer)
            print("Renderer added")
        } else {
            print("Renderer failed to add")
        }
    }
    
    private func removePythonRenderer() {
        if let renderer = pythonRenderer,
           let localVideoTrack = cameraTrack as? LocalVideoTrack {
            localVideoTrack.remove(videoRenderer: renderer)
            print("Renderer removed")
        }
        pythonRenderer = nil
    }
    
    private func setupPythonVideoConnection() {
        videoConnection?.cancel()
        videoConnection = nil
        
        let host = NWEndpoint.Host(macIPAddress)
        let parameters = NWParameters.tcp
        
        parameters.allowLocalEndpointReuse = true
        
        let tcpOptions = NWProtocolTCP.Options()
        tcpOptions.enableKeepalive = true
        tcpOptions.keepaliveIdle = 30
        tcpOptions.keepaliveInterval = 10
        tcpOptions.keepaliveCount = 3
        tcpOptions.connectionTimeout = 10
        tcpOptions.noDelay = true
        parameters.defaultProtocolStack.transportProtocol = tcpOptions
        
        videoConnection = NWConnection(host: host, port: videoPort, using: parameters)

        videoConnection?.stateUpdateHandler = { [weak self] newState in
            DispatchQueue.main.async {
                guard let self else { return }
                switch newState {
                case .ready:
                    print("TCP connected")
                    self.videoConnectionRetryCount = 0
                    self.pythonConnectionStatus = "Connected"
                case .failed(let error):
                    print("TCP connection failed: \(error)")
                    self.videoConnection?.cancel()
                    self.videoConnection = nil
                    self.pythonConnectionStatus = "Failed: \(error.localizedDescription)"
                    
                    if self.videoConnectionRetryCount < self.maxRetryCount {
                        self.scheduleVideoReconnect()
                    }
                case .cancelled:
                    print("Video connection cancelled")
                    self.pythonConnectionStatus = "Disconnected"
                default:
                    print("Video connection status: \(newState)")
                    self.pythonConnectionStatus = "Connecting..."
                }
            }
        }
        videoConnection?.start(queue: .global())
    }
    
    private func scheduleVideoReconnect() {
        videoConnectionRetryCount += 1
        let delay = min(Double(videoConnectionRetryCount * 2), 30.0)
        
        DispatchQueue.global().asyncAfter(deadline: .now() + delay) { [weak self] in
            Task { @MainActor in
                guard let self, self.isPythonStreamingEnabled else { return }
                self.setupPythonVideoConnection()
            }
        }
    }
    
    private func setupNetworkMonitoring() {
        guard networkMonitor == nil else { return } /// Don't create multiple monitors
        
        networkMonitor = NWPathMonitor()
        networkMonitor?.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async {
                if path.status == .satisfied {
                    print("Network available")
                    self?.reconnectIfNeeded()
                } else {
                    print("Network not available")
                }
            }
        }
        networkMonitor?.start(queue: .global())
    }
    
    private func reconnectIfNeeded() {
        guard isPythonStreamingEnabled || isLocationStreamingEnabled else { return }
        
        if isPythonStreamingEnabled && videoConnection?.state != .ready {
            print("Reconnecting video")
            setupPythonVideoConnection()
        }
        
        if isLocationStreamingEnabled && locationConnection?.state != .ready {
            print("Reconnecting location")
            setupLocationConnection()
        }
    }
    
    func processCameraFrame(_ pixelBuffer: CVPixelBuffer) {
        guard isPythonStreamingEnabled else { 
            print("Frame received without enabled video")
            return 
        }
        
        guard let connection = videoConnection else {
            print("Frame received without connected video")
            return
        }
        
        guard connection.state == .ready else { 
            print("Frame received without ready connection")
            return 
        }

        let now = Date()
        guard now.timeIntervalSince(lastFrameSentTime) >= frameRateLimit else { return }
        lastFrameSentTime = now
        
        /// Convert to UIImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext()
        
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else { 
            return 
        }
        
        #if os(iOS)
        /// Orient 90 degrees
        let uiImage = UIImage(cgImage: cgImage, scale: 1.0, orientation: .right)
        
        /// Resize image
        let targetSize = CGSize(width: 720, height: 1280)
        let resizedImage = resizeImageExact(image: uiImage, targetSize: targetSize)
        
        /// Convert to JPEG with compression
        guard let imageData = resizedImage.jpegData(compressionQuality: 0.85) else { return }
        
        trackFrameSendingMetrics(imageData: imageData)

        /// Send to server
        sendVideoFrame(data: imageData)
        #endif
    }
    
    #if os(iOS)
    private func resizeImageExact(image: UIImage, targetSize: CGSize) -> UIImage {
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1.0
        
        let renderer = UIGraphicsImageRenderer(size: targetSize, format: format)
        let resizedImage = renderer.image { _ in
            image.draw(in: CGRect(origin: .zero, size: targetSize))
        }
        
        return resizedImage
    }
    #endif
    
    private func sendVideoFrame(data: Data) {
        guard let connection = videoConnection, connection.state == .ready else {
            return
        }
        
        let frameLength = UInt32(data.count)
        var lengthBytes = frameLength.bigEndian
        let lengthData = Data(bytes: &lengthBytes, count: 4)
        let completeData = lengthData + data
        
        connection.send(content: completeData, completion: .contentProcessed({ [weak self] sendError in
            if let error = sendError {
                print("Frame failed to send: \(error)")
                DispatchQueue.main.async {
                    self?.pythonConnectionStatus = "Send Error"
                }
            }
        }))
    }
    
    private func setupLocationServices() {
        locationManager.delegate = self
        locationManager.requestWhenInUseAuthorization()
        
        locationManager.headingOrientation = .portrait
        locationManager.startUpdatingLocation()
        
        if CLLocationManager.headingAvailable() {
            locationManager.startUpdatingHeading()
        } else {
            print("Heading not available")
        }
        
        setupQueueProcessTimer()
        
        updateLocationPermissionStatus()
    }
    
    private func updateLocationPermissionStatus() {
        switch locationManager.authorizationStatus {
        case .notDetermined:
            locationPermissionStatus = "Not Determined"
        case .restricted:
            locationPermissionStatus = "Restricted"
        case .denied:
            locationPermissionStatus = "Denied"
        case .authorizedAlways:
            locationPermissionStatus = "Always Authorized"
        case .authorizedWhenInUse:
            locationPermissionStatus = "When In Use"
        @unknown default:
            locationPermissionStatus = "Unknown"
        }
    }
    
    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) {
        Task { @MainActor in
            if let loc = locations.last {
                let locationCopy = CLLocation(coordinate: loc.coordinate, 
                                            altitude: loc.altitude, 
                                            horizontalAccuracy: loc.horizontalAccuracy, 
                                            verticalAccuracy: loc.verticalAccuracy, 
                                            timestamp: loc.timestamp)
                currentLocation = locationCopy
                currentLatitude = loc.coordinate.latitude
                currentLongitude = loc.coordinate.longitude
                print("Location: \(loc.coordinate.latitude), \(loc.coordinate.longitude)")
            }
            sendLocationDataIfNeeded()
        }
    }
    
    nonisolated func locationManager(_ manager: CLLocationManager, didUpdateHeading newHeading: CLHeading) {
        let headingValue = newHeading.trueHeading != -1 ? newHeading.trueHeading : newHeading.magneticHeading
        let headingCopy = newHeading.copy() as! CLHeading
        
        Task { @MainActor in
            currentHeading = headingCopy
            currentHeadingValue = headingValue
            print("Heading: \(headingValue)")
            
            sendLocationDataIfNeeded()
        }
    }
    
    nonisolated func locationManager(_ manager: CLLocationManager, didChangeAuthorization status: CLAuthorizationStatus) {
        Task { @MainActor in
            updateLocationPermissionStatus()
        }
    }
    
    func toggleLocationStreaming() {
        if isLocationStreamingEnabled {
            stopLocationStreaming()
        } else {
            startLocationStreaming()
        }
    }
    
    private func startLocationStreaming() {
        print("Target server: \(macIPAddress):\(locationPort)")
        print("Permission status: \(locationPermissionStatus)")
        
        setupLocationConnection()
        isLocationStreamingEnabled = true
        
        print("Location streaming started")
    }
    
    private func stopLocationStreaming() {
        isLocationStreamingEnabled = false
        
        locationConnection?.cancel()
        locationConnection = nil
        locationConnectionStatus = "Disconnected"

        pendingLocationUpdates.removeAll()

        queueProcessTimer?.invalidate()
        queueProcessTimer = nil
        
        print("Location streaming stopped")
    }
    
    private func setupLocationConnection() {
        locationConnection?.cancel()
        locationConnection = nil
        
        let host = NWEndpoint.Host(macIPAddress)
        let parameters = NWParameters.tcp
        
        parameters.allowLocalEndpointReuse = true
        
        let tcpOptions = NWProtocolTCP.Options()
        tcpOptions.enableKeepalive = true
        tcpOptions.keepaliveIdle = 30
        tcpOptions.keepaliveInterval = 10
        tcpOptions.keepaliveCount = 3
        tcpOptions.connectionTimeout = 10
        parameters.defaultProtocolStack.transportProtocol = tcpOptions
        
        isLocationConnecting = true
        lastLocationConnectionAttemptTime = Date()
        locationConnection = NWConnection(host: host, port: locationPort, using: parameters)

        locationConnection?.stateUpdateHandler = { [weak self] newState in
            Task { @MainActor in
                guard let self else { return }
                
                switch newState {
                case .ready:
                    print("Location connected")
                    self.locationConnectionRetryCount = 0
                    self.isLocationConnecting = false
                    self.locationConnectionStatus = "Connected"
                    self.processPendingLocationUpdates()
                    
                case .preparing:
                    print("Preparing location connection")
                    self.isLocationConnecting = true
                    self.locationConnectionStatus = "Preparing"
                    
                case .setup:
                    print("Setting up location connection")
                    self.isLocationConnecting = true
                    self.locationConnectionStatus = "Setting up"
                    
                case .failed(let error):
                    print("Location connection failed: \(error)")
                    self.locationConnection?.cancel()
                    self.locationConnection = nil
                    self.isLocationConnecting = false
                    self.locationConnectionStatus = "Failed: \(error.localizedDescription)"
                    
                    if self.locationConnectionRetryCount < self.maxRetryCount {
                        self.scheduleLocationReconnect()
                    } else {
                        print("Retry limit reached")
                    }
                case .cancelled:
                    print("Location connection cancelled")
                    self.isLocationConnecting = false
                    self.locationConnectionStatus = "Cancelled"
                default:
                    print("Location connection status: \(newState)")
                    self.locationConnectionStatus = "Connecting..."
                }
            }
        }
        locationConnection?.start(queue: .global())
    }
    
    private func scheduleLocationReconnect() {
        locationConnectionRetryCount += 1
        let delay = min(Double(locationConnectionRetryCount * 2), 30.0)
        
        DispatchQueue.global().asyncAfter(deadline: .now() + delay) { [weak self] in
            Task { @MainActor in
                guard let self, self.isLocationStreamingEnabled else { return }
                self.setupLocationConnection()
            }
        }
    }
    
    private func sendLocationDataIfNeeded() {
        let now = Date()
        guard now.timeIntervalSince(lastLocationSentTime) >= 0.1 else { return } /// Throttle to 10Hz
        lastLocationSentTime = now
        
        guard let loc = currentLocation, let heading = currentHeading else {
            return
        }
        
        let headingValue = heading.trueHeading != -1 ? heading.trueHeading : heading.magneticHeading
        sendLocationToServer(latitude: loc.coordinate.latitude,
                           longitude: loc.coordinate.longitude,
                           heading: headingValue)
    }
    
    private func sendLocationToServer(latitude: Double, longitude: Double, heading: Double) {
        guard isLocationStreamingEnabled else { return }
        
        guard let connection = locationConnection, connection.state == .ready else {
            print("Location connection not ready, queuing update")
            queueLocationUpdate(latitude: latitude, longitude: longitude, heading: heading)
            return
        }
        
        let jsonDict: [String: Any] = [
            "latitude": latitude,
            "longitude": longitude,
            "heading": heading,
            "timestamp": Date().timeIntervalSince1970
        ]
        
        guard let jsonData = try? JSONSerialization.data(withJSONObject: jsonDict) else {
            print("Failed to serialize location data")
            return
        }
        
        connection.send(content: jsonData, completion: .contentProcessed({ [weak self] sendError in
            Task { @MainActor in
                if let error = sendError {
                    print("Failed to send location data: \(error)")
                    self?.handleLocationConnectionError()
                } else {
                    self?.totalLocationsSent += 1
                    print("Location data sent")
                }
            }
        }))
    }
    
    private func queueLocationUpdate(latitude: Double, longitude: Double, heading: Double) {
        if pendingLocationUpdates.count >= maxPendingUpdates {
            pendingLocationUpdates.removeFirst()
        }
        
        pendingLocationUpdates.append((latitude, longitude, heading))
        
        print("Queued location update")
    }
    
    private func setupQueueProcessTimer() {
        queueProcessTimer?.invalidate()
        queueProcessTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            Task { @MainActor in
                if let connection = self.locationConnection, connection.state == .ready {
                    self.processPendingLocationUpdates()
                } else if !self.pendingLocationUpdates.isEmpty {
                    let now = Date()
                    if now.timeIntervalSince(self.lastLocationConnectionAttemptTime) > 10.0 {
                        self.lastLocationConnectionAttemptTime = now
                        print("Attempting location reconnect")
                        self.setupLocationConnection()
                    }
                }
            }
        }
    }
    
    private func processPendingLocationUpdates() {
        guard !pendingLocationUpdates.isEmpty else { return }
        
        print("Processing location updates")
        
        if let lastUpdate = pendingLocationUpdates.last,
           let connection = locationConnection, 
           connection.state == .ready {
            let jsonDict: [String: Any] = [
                "latitude": lastUpdate.latitude,
                "longitude": lastUpdate.longitude,
                "heading": lastUpdate.heading,
                "timestamp": Date().timeIntervalSince1970
            ]
            
            guard let jsonData = try? JSONSerialization.data(withJSONObject: jsonDict) else {
                print("Failed to serialize queued location data")
                return
            }
            
            print("Sending queued location: \(lastUpdate.latitude), \(lastUpdate.longitude)")
            connection.send(content: jsonData, completion: .contentProcessed({ [weak self] sendError in
                Task { @MainActor in
                    guard let self else { return }
                    
                    if let error = sendError {
                        print("Failed to send queued location data: \(error)")
                        self.handleLocationConnectionError()
                    } else {
                        print("Queued location sent")
                        self.pendingLocationUpdates.removeAll()
                        self.totalLocationsSent += 1
                    }
                }
            }))
        } else {
            print("Connection not ready")
        }
    }
    
    private func handleLocationConnectionError() {
        locationConnection?.cancel()
        locationConnection = nil
        scheduleLocationReconnect()
    }
}
