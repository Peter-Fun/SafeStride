import LiveKit
import AVFoundation
import LiveKitWebRTC
import ObjectiveC

/// A custom video renderer that captures frames for Python streaming
final class PythonStreamingRenderer: NSObject, VideoRenderer {
    var adaptiveStreamSize: CGSize
    
    private let frameHandler: @Sendable (CVPixelBuffer) -> Void
    
    init(frameHandler: @escaping @Sendable (CVPixelBuffer) -> Void) {
        self.frameHandler = frameHandler
        self.adaptiveStreamSize = CGSize.zero
        super.init()
    }
    
    func render(frame: VideoFrame) {
        print("Frame Received")
        
        if let cvPixelVideoBuffer = frame.buffer as? CVPixelVideoBuffer {
            let cvBufferMirror = Mirror(reflecting: cvPixelVideoBuffer)
            if let rtcTypeChild = cvBufferMirror.children.first(where: { $0.label == "_rtcType" }),
                let rtcType = rtcTypeChild.value as? NSObject {
                    let rtcTypeMirror = Mirror(reflecting: rtcType)
                    for child in rtcTypeMirror.children {
                        if CFGetTypeID(child.value as CFTypeRef) == CVPixelBufferGetTypeID() {
                            let childValue = child.value as! CVPixelBuffer
                            frameHandler(childValue)
                            return
                        }
                    }
                let methodNames = ["pixelBuffer", "buffer", "cvPixelBuffer", "nativePixelBuffer"]
                for methodName in methodNames {
                    let selector = NSSelectorFromString(methodName)
                    if rtcType.responds(to: selector) {
                        if let result = rtcType.perform(selector)?.takeUnretainedValue() {
                            if CFGetTypeID(result as CFTypeRef) == CVPixelBufferGetTypeID() {
                                let pixelBuffer = result as! CVPixelBuffer
                                frameHandler(pixelBuffer)
                                return
                            }
                        }
                    }
                }
                var methodCount: UInt32 = 0
                let methods = class_copyMethodList(object_getClass(rtcType), &methodCount)
                defer { free(methods) }
            
                for i in 0..<Int(methodCount) {
                    if let method = methods?[i] {
                        let selector = method_getName(method)
                        let methodName = NSStringFromSelector(selector)
                        if methodName.lowercased().contains("pixel") || methodName.lowercased().contains("buffer") {
                        }
                    }
                }
            }
        }

        let frameMirror = Mirror(reflecting: frame)
        let bufferMirror = Mirror(reflecting: frame.buffer)

        if let rtcFrameChild = frameMirror.children.first(where: { $0.label == "_rtcFrame" }) {
            let rtcFrame = rtcFrameChild.value
            if let rtcFrameObj = rtcFrame as? NSObject {
                if rtcFrameObj.responds(to: NSSelectorFromString("buffer")) {
                    do {
                        let rtcBuffer = rtcFrameObj.value(forKey: "buffer")
                        if let bufferObj = rtcBuffer as? NSObject,
                           bufferObj.responds(to: NSSelectorFromString("pixelBuffer")) {
                            let pixelBuffer = bufferObj.value(forKey: "pixelBuffer")
                            if CFGetTypeID(pixelBuffer as CFTypeRef) == CVPixelBufferGetTypeID() {
                                let cvPixelBuffer = pixelBuffer as! CVPixelBuffer
                                frameHandler(cvPixelBuffer)
                                return
                            }
                        }
                        if let bufferObj = rtcBuffer {
                            let rtcBufferMirror = Mirror(reflecting: bufferObj)
                        }
                    } catch {
                        print("Error when accessing buffer: \(error)")
                    }
                }
            }
        }
        if let rtcTypeChild = bufferMirror.children.first(where: { $0.label == "_rtcType" }) {
            let rtcType = rtcTypeChild.value
        }

        func findCVPixelBufferInMirror(_ mirror: Mirror, path: String = "") -> CVPixelBuffer? {
            for child in mirror.children {
                let childPath = path.isEmpty ? (child.label ?? "unlabeled") : "\(path).\(child.label ?? "unlabeled")"
                if CFGetTypeID(child.value as CFTypeRef) == CVPixelBufferGetTypeID() {
                    let pixelBuffer = child.value as! CVPixelBuffer
                    return pixelBuffer
                }
                if path.components(separatedBy: ".").count < 3 {
                    let childMirror = Mirror(reflecting: child.value)
                    if let found = findCVPixelBufferInMirror(childMirror, path: childPath) {
                        return found
                    }
                }
            }
            return nil
        }
        
        if let pixelBuffer = findCVPixelBufferInMirror(frameMirror) {
            frameHandler(pixelBuffer)
            return
        }
        
        if let pixelBuffer = findCVPixelBufferInMirror(bufferMirror) {
            frameHandler(pixelBuffer)
            return
        }
    }
    func renderFrame(_ frame: VideoFrame?) {
        guard let frame = frame else { return }
        render(frame: frame)
    }

    var canReceiveFrames: Bool {
        return true
    }
    
    var isAdaptiveStreamEnabled: Bool {
        return false
    }

    var adaptOutputFormat: Bool {
        return false
    }
    
    var renderDimensions: CGSize {
        return CGSize.zero
    }
}
