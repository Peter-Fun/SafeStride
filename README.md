# I Cant See
An AI-Powered Navigation system designed for the visually impaired.

This is the repository containing the collective source code of the project. The code is separated into three sections:

## Gemini-Live Agent

Inside "ai-agent-gemini-live" is the source code for the voice agent present in our project. We used [LiveKit](https://livekit.io) to host Gemini Live. 

## Swift iOS App

Inside "ai-agent-swift-app" is the source code for the iOS app that calls the voice agent. The app has various buttons for the user to interact with the voice agent and connect with the Python navigation backend.

These buttons include a mute button, a camera button, a server button, a GPS arrow button, a chat button, and an end call button.

The mute and camera buttons allow for the user to speak with the agent to gain an understanding of the environment in front of them.

The server and GPS arrow button allow for connection to the Python navigation backend for path navigation and obstacle detection to a given location.

The chat button allows for the chat history with the voice agent to be opened and viewed by other people who would like to assist the user and wants to get context from their conversation with the voice agent.



## Python GPS Navigation

File "roboflowpathgps.py" is the source code for the Python backend.

Functionalities include obstacle detection using a custom [Roboflow](https://roboflow.com) model, ground segmentation using the Segformer-B0 model, a Tesla-style path generation system, spatial audio following the direction of the path, navigation directions following the Google Maps API, and a remote server to connect with the app to collect location and video data.

