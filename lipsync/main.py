import subprocess

# Command to execute
command = [
    "python", "inference.py",
    "--checkpoint_path", "checkpoints/wav2lip_gan.pth",
    "--face", r"C:\Users\sharo\Desktop\Ai-character-generation\Intelligence model\lipsync\sample_data\test.mp4.mp4",
    "--audio", r"C:\Users\sharo\Desktop\Ai-character-generation\Intelligence model\lipsync\sample_data\test.wav.wav"
]

# Change directory to Wav2Lip and run the command
subprocess.run(["cd", "Wav2Lip", "&&"] + command, shell=True)
