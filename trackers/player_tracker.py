from ultralytics import YOLO
import cv2
import pickle
import sys

sys.path.append("../")
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    def __init__(self, model_path):
        # Khởi tạo mô hình YOLO từ đường dẫn mô hình
        self.model = YOLO(model_path)

    def choose_and_filter_players(self, court_keypoints, player_detections):
        # Lấy các phát hiện của người chơi từ khung hình đầu tiên
        player_detections_first_frame = player_detections[0]
        # Chọn các người chơi gần nhất với các điểm chính của sân
        chosen_player = self.choose_players(court_keypoints, player_detections_first_frame)
        # Lọc các phát hiện người chơi chỉ giữ lại các người chơi đã chọn
        filtered_player_detections = []
        for player_dict in player_detections:
            filtered_player_dict = {track_id: bbox for track_id, bbox in player_dict.items() if
                                    track_id in chosen_player}
            filtered_player_detections.append(filtered_player_dict)
        return filtered_player_detections

    def choose_players(self, court_keypoints, player_dict):
        distances = []
        # Tính toán khoảng cách giữa trung tâm bounding box của người chơi và các điểm chính của sân
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance

            distances.append((track_id, min_distance))
        # Sắp xếp khoảng cách theo thứ tự tăng dần
        distances.sort(key=lambda x: x[1])
        # Chọn hai người chơi gần nhất với điểm chính của sân
        chosen_players = [distances[0][0], distances[1][0]]
        return chosen_players

    def detect_frames(self, frames, read_from_stubs=False, stub_path=None):
        player_detections = []
        # Nếu đọc từ tập tin đã lưu, nạp các phát hiện người chơi từ tập tin
        if read_from_stubs and stub_path is not None:
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        # Phát hiện người chơi trong từng khung hình
        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        # Lưu các phát hiện người chơi vào tập tin nếu có đường dẫn
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        # Dự đoán các bounding boxes của người chơi trong khung hình
        results = self.model.track(frame, persist=True)[0]
        id_name_dict = results.names

        player_dict = {}
        for box in results.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_cls_id = box.cls.tolist()[0]
            object_cls_name = id_name_dict[object_cls_id]
            if object_cls_name == "person":
                player_dict[track_id] = result
        return player_dict

    def draw_bboxes(self, video_frames, player_detections):
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            # Vẽ bounding box trên khung hình
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Player Id: {track_id}", (int(bbox[0]), int(bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames
