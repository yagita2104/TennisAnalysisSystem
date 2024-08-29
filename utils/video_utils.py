import cv2
import os
import webbrowser
import pytesseract
import re  # Thư viện hỗ trợ biểu thức chính quy

# Cài đặt đường dẫn tới tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path):
    fourcc = cv2.VideoWriter.fourcc(*'H264')
    out = cv2.VideoWriter(output_video_path, fourcc, 24,
                          (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()


def play_video(video_path):
    file_url = 'file://' + os.path.abspath(video_path)
    webbrowser.open(file_url)


def extract_player_names(video_path):
    video = cv2.VideoCapture(video_path)
    ret, frame = video.read()
    if not ret:
        video.release()
        return None, None
    height, width, _ = frame.shape
    left_region = frame[int(height * 0.75):height, 0:int(width * 0.25)]
    gray_left_region = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_left_region)
    names = text.split('\n')
    names = [re.sub(r'\d+', '', name).strip() for name in names if name.strip()]
    names = [name for name in names if name]
    player_names = names[:2] if len(names) >= 2 else (None, None)
    video.release()
    return player_names
