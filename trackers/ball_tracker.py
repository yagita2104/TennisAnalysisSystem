from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self, model_path):
        # Khởi tạo mô hình YOLO từ đường dẫn mô hình
        self.model = YOLO(model_path)

    def interpolate_ball_detection(self, ball_positions):
        # Lấy các bounding boxes của quả bóng từ các khung hình
        ball_positions = [x.get(1, []) for x in ball_positions]
        # Chuyển danh sách thành pandas DataFrame
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Nội suy các dữ liệu bị thiếu
        df_ball_positions = df_ball_positions.interpolate()
        # Điền giá trị thiếu ở đầu hoặc cuối
        df_ball_positions = df_ball_positions.bfill()

        # Chuyển đổi DataFrame trở lại danh sách các từ điển
        ball_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def get_ball_shot_frames(self, ball_positions):
        # Lấy các bounding boxes của quả bóng từ các khung hình
        ball_positions = [x.get(1, []) for x in ball_positions]
        # Chuyển danh sách thành pandas DataFrame
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Thêm cột để xác định các khung hình có quả bóng bị đánh
        df_ball_positions['ball_hit'] = 0
        # Tính tọa độ y trung bình của bounding box
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        # Tính trung bình trượt của tọa độ y trung bình
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(window=5, min_periods=1, center=False).mean()
        # Tính sự thay đổi trong tọa độ y trung bình
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        minimum_change_frames_for_hit = 25
        # Phân tích sự thay đổi để phát hiện các khung hình có quả bóng bị đánh
        for i in range(1, len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 > df_ball_positions['delta_y'].iloc[i + 1]
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 < df_ball_positions['delta_y'].iloc[i + 1]
            if negative_position_change or positive_position_change:
                change_count = 0
                # Đếm số khung hình có sự thay đổi liên tiếp
                for change_frame in range(i + 1, i + int(minimum_change_frames_for_hit * 1.2) + 1):
                    negative_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] > 0 > df_ball_positions['delta_y'].iloc[change_frame]
                    positive_position_change_following_frame = df_ball_positions['delta_y'].iloc[i] < 0 < df_ball_positions['delta_y'].iloc[change_frame]
                    if negative_position_change and negative_position_change_following_frame:
                        change_count += 1
                    elif positive_position_change and positive_position_change_following_frame:
                        change_count += 1
                # Nếu có sự thay đổi đủ lớn, đánh dấu khung hình là có quả bóng bị đánh
                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1
        # Lấy danh sách các khung hình có quả bóng bị đánh
        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()
        return frame_nums_with_ball_hits

    def detect_frames(self, frames, read_from_stubs=False, stub_path=None):
        ball_detections = []

        # Nếu đọc từ tập tin đã lưu, nạp các phát hiện quả bóng từ tập tin
        if read_from_stubs and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        # Phát hiện quả bóng trong từng khung hình
        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        # Lưu các phát hiện quả bóng vào tập tin nếu có đường dẫn
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)

        return ball_detections

    def detect_frame(self, frame):
        # Dự đoán các bounding boxes của quả bóng trong khung hình
        results = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
        return ball_dict

    def draw_bboxes(self, video_frames, ball_detections):
        output_video_frames = []
        for frame, ball_dict in zip(video_frames, ball_detections):
            # Vẽ bounding box trên khung hình
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball Id: {track_id}", (int(bbox[0]), int(bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
        return output_video_frames
