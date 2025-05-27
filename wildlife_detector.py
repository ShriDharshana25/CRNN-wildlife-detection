import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import sounddevice as sd
import librosa
import numpy as np
import scipy.io.wavfile as wav
import cv2
import RPi.GPIO as GPIO
from time import sleep, time

# ========== GPIO SETUP ==========
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# Buzzer
GPIO.setup(23, GPIO.OUT, initial=GPIO.LOW)

# Ultrasonic Sensor
TRIG = 24
ECHO = 25
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)

# ========== DISTANCE MEASUREMENT ==========
def measure_distance():
    GPIO.output(TRIG, False)
    sleep(0.1)

    GPIO.output(TRIG, True)
    sleep(0.00001)
    GPIO.output(TRIG, False)

    timeout = time() + 5
    while GPIO.input(ECHO) == 0:
        pulse_start = time()
        if pulse_start > timeout:
            print("Timeout waiting for ECHO to go HIGH")
            return None

    timeout = time() + 5
    while GPIO.input(ECHO) == 1:
        pulse_end = time()
        if pulse_end > timeout:
            print("Timeout waiting for ECHO to go LOW")
            return None

    pulse_duration = pulse_end - pulse_start
    distance = (pulse_duration * 34300) / 2
    return round(distance, 2)

# ========== AUDIO + MODEL SETUP ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['bear', 'elephant', 'fox', 'hyena', 'leopard', 'lion', 'pig', 'tiger']
num_classes = len(class_names)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
        )
        self.rnn = nn.LSTM(input_size=64 * 16, hidden_size=128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)  # (B, C, H, W) -> (B, 64, 16, 16)
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.reshape(x.size(0), x.size(1), -1)  # (B, W, C*H)
        _, (hn, _) = self.rnn(x)  # (1, B, H)
        out = self.fc(hn[-1])
        return out

model = CRNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("/home/pi/animalscrnn_model.pth", map_location=device))
model.eval()

# Audio config
DURATION = 5
SAMPLERATE = 22050
sd.default.device = 3  # Adjust as needed

def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, sr=SAMPLERATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)

reference_audio = {
    "bear": extract_audio_features("bear.wav"),
    "elephant": extract_audio_features("elephant.wav"),
    "fox": extract_audio_features("fox.wav"),
    "hyena": extract_audio_features("hyena.wav"),
    "leopard": extract_audio_features("leopard.wav"),
    "lion": extract_audio_features("lion.wav"),
    "pig": extract_audio_features("pig.wav"),
    "tiger": extract_audio_features("tiger.wav"),
}

def match_audio(reference_features):
    print("Recording for 5 seconds...")
    audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1, dtype='float32')
    sd.wait()
    wav.write("live_audio.wav", SAMPLERATE, audio)
    live_features = extract_audio_features("live_audio.wav")

    similarity = np.dot(reference_features, live_features) / (
        np.linalg.norm(reference_features) * np.linalg.norm(live_features)
    )
    similarity *= -1  # Optional inversion
    print(f"Audio Similarity Score: {similarity:.2f}")
    return similarity >= 0.45

# ========== MAIN DETECTION LOOP ==========
def detect_animals():
    cap = cv2.VideoCapture(0)
    confidence_threshold = 0.7

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        input_img = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_img)
            probs = torch.softmax(output, dim=1)
            conf, predicted = torch.max(probs, 1)
            predicted_class = class_names[predicted.item()]
            confidence = conf.item()

        if confidence >= confidence_threshold:
            label = f"Detected: {predicted_class} ({confidence:.2f})"
            cv2.putText(frame, label, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            print(label)
            if predicted_class in reference_audio:
                print(f"Verifying audio for {predicted_class}...")
                if match_audio(reference_audio[predicted_class]):
                    dist = measure_distance()
                    if dist is not None:
                        print(f"Measured Distance: {dist} cm")
                        if dist < 10:
                            print("ALERTTT! Wild animal close.")
                            for _ in range(10):
                                GPIO.output(23, GPIO.HIGH)
                                sleep(0.3)
                                GPIO.output(23, GPIO.LOW)
                                sleep(0.3)
                        else:
                            print(f"{predicted_class.capitalize()} confirmed, but not close enough.")
                    else:
                        print("Distance measurement failed.")
                else:
                    print("Audio mismatch.")
        else:
            print("No confident prediction.")

        cv2.imshow("Wild Animal Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    GPIO.cleanup()

if __name__ == "__main__":
    detect_animals()
