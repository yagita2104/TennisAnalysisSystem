# from ultralytics import YOLO
#
# model = YOLO('yolov8x')
#
# result = model.track('input_videos/input_video.mp4', conf=0.2, save=True)
# print(result)
# print("Boxes: ")
# for box in result[0].boxes:
# print(box)
# import webbrowser
# import os
#
# # Đường dẫn tương đối đến video trong thư mục con
# relative_path = 'output_videos/output_video.avi'
#
# # Chuyển đổi đường dẫn tương đối thành URL file
# file_url = 'file://' + os.path.abspath(relative_path)
#
# # Mở video bằng trình duyệt hệ thống
# webbrowser.open(file_url)
import cv2
# import pytesseract
# import re  # Thư viện hỗ trợ biểu thức chính quy
#
# # Cài đặt đường dẫn tới tesseract
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Điều chỉnh theo đường dẫn cài đặt của bạn
#

# def extract_player_names(video_path):
#     # Mở video
#     video = cv2.VideoCapture(video_path)
#
#     # Đọc khung hình đầu tiên
#     ret, frame = video.read()
#
#     if not ret:
#         video.release()
#         return None, None
#
#     # Xác định kích thước của khung hình
#     height, width, _ = frame.shape
#
#     # Cắt phần góc dưới bên trái theo tỷ lệ
#     left_region = frame[int(height * 0.75):height, 0:int(width * 0.25)]
#
#     # Chuyển ảnh sang màu xám để cải thiện kết quả OCR
#     gray_left_region = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY)
#
#     # Thực hiện OCR
#     text = pytesseract.image_to_string(gray_left_region)
#
#     # Tách các tên từ văn bản OCR và lọc bỏ các số
#     names = text.split('\n')
#     names = [re.sub(r'\d+', '', name).strip() for name in names if name.strip()]
#
#     # Lọc và chỉ lấy các chuỗi không rỗng và không chứa số
#     names = [name for name in names if name]
#
#     # Lấy 2 tên đầu tiên nếu có ít nhất 2 tên
#     player_names = names[:2] if len(names) >= 2 else (None, None)
#
#     # Giải phóng tài nguyên
#     video.release()
#
#     return player_names
#
#
# # Sử dụng hàm để trích xuất tên cầu thủ
# video_path = 'input_videos/input_video.mp4'
# name1, name2 = extract_player_names(video_path)
# print("Tên cầu thủ 1:", name1)
# print("Tên cầu thủ 2:", name2)







