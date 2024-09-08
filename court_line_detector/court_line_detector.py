import torch
import torchvision.transforms as transforms
import cv2
from torchvision import models

class CourtLineDetector:
    def __init__(self, model_path):
        # Khởi tạo mô hình ResNet-101 và tùy chỉnh lớp fully connected để phù hợp với số lượng điểm chính
        self.model = models.resnet101(pretrained=True)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 14 * 2)  # 14 điểm chính, mỗi điểm có 2 giá trị (x, y)
        self.model.load_state_dict(torch.load(model_path, map_location='cpu'))  # Tải trọng số của mô hình đã huấn luyện
        self.transform = transforms.Compose([
            transforms.ToPILImage(),  # Chuyển đổi từ ảnh OpenCV BGR sang PIL RGB
            transforms.Resize((224, 224)),  # Thay đổi kích thước ảnh về 224x224 (kích thước đầu vào của mô hình)
            transforms.ToTensor(),  # Chuyển đổi ảnh PIL thành tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa tensor
        ])

    def predict(self, image):
        # Chuyển đổi ảnh từ định dạng BGR sang RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Áp dụng các phép biến đổi và thêm một chiều batch
        image_tensor = self.transform(image_rgb).unsqueeze(0)
        with torch.no_grad():
            # Dự đoán điểm chính từ ảnh
            outputs = self.model(image_tensor)
        keypoints = outputs.squeeze().cpu().numpy()  # Chuyển đổi đầu ra thành mảng numpy

        # Chuyển đổi tọa độ điểm chính từ kích thước 224x224 về kích thước gốc của ảnh
        original_h, original_w = image.shape[:2]
        keypoints[::2] *= original_w / 224.0
        keypoints[1::2] *= original_h / 224.0

        return keypoints

    def draw_keypoints(self, image, keypoints):
        # Vẽ các điểm chính lên ảnh
        for i in range(0, len(keypoints), 2):
            x = int(keypoints[i])
            y = int(keypoints[i + 1])
            # Vẽ số thứ tự của điểm chính
            cv2.putText(image, str(i // 2), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            # Vẽ điểm chính
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        return image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        # Vẽ các điểm chính trên từng khung hình video
        output_video_frames = []
        for frame in video_frames:
            frame = self.draw_keypoints(frame, keypoints)
            output_video_frames.append(frame)
        return output_video_frames
