import gradio as gr
from utils import read_video, save_video, play_video, extract_player_names, measure_distance, \
    convert_pixel_distance_to_meters, \
    draw_player_stats
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt
import cv2
from copy import deepcopy
import pandas as pd
import constants


def main(input_video_path, status_stub):
    # Đọc video từ đường dẫn
    video_frames = read_video(input_video_path)

    # Khởi tạo các tracker cho cầu thủ và bóng
    player_tracker = PlayerTracker(model_path='yolov8x')
    ball_tracker = BallTracker(model_path='models/yolov8_best.pt')

    # Phát hiện cầu thủ và bóng trên từng khung hình
    player_detections = player_tracker.detect_frames(video_frames, read_from_stubs=status_stub,
                                                     stub_path="tracker_stubs/player_detections.pkl")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stubs=status_stub,
                                                 stub_path="tracker_stubs/ball_detections.pkl")

    # Nội suy vị trí của bóng để điền vào những điểm còn thiếu
    ball_detections = ball_tracker.interpolate_ball_detection(ball_detections)

    # Khởi tạo mô hình phát hiện đường kẻ sân và dự đoán các điểm chính của sân
    court_model_path = "models/keypoints_resnet_101.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    # Chọn cầu thủ dựa trên các điểm chính của sân
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    # Khởi tạo đối tượng MiniCourt để chuyển đổi tọa độ từ ảnh gốc sang mini court
    mini_court = MiniCourt(video_frames[0])

    # Phát hiện các khung hình có cú đánh bóng
    ball_shot_frames = ball_tracker.get_ball_shot_frames(ball_detections)

    # Chuyển đổi tọa độ của các cầu thủ và bóng sang tọa độ của mini court
    player_mini_court_detections, ball_mini_court_detections = mini_court.convert_bounding_boxes_to_mini_court_coordinates(
        player_detections, ball_detections, court_keypoints)

    # Khởi tạo dữ liệu thống kê cầu thủ
    player_stats_data = [{
        'frame_num': 0,
        'player_1_number_of_shots': 0,
        'player_1_total_shot_speed': 0,
        'player_1_last_shot_speed': 0,
        'player_1_total_player_speed': 0,
        'player_1_last_player_speed': 0,

        'player_2_number_of_shots': 0,
        'player_2_total_shot_speed': 0,
        'player_2_last_shot_speed': 0,
        'player_2_total_player_speed': 0,
        'player_2_last_player_speed': 0,

    }]

    # Tính toán các chỉ số cầu thủ từ các cú đánh bóng
    for ball_shot_ind in range(len(ball_shot_frames) - 1):
        start_frame = ball_shot_frames[ball_shot_ind]
        end_frame = ball_shot_frames[ball_shot_ind + 1]
        ball_shot_time_in_seconds = (end_frame - start_frame) / 24  # 24fps

        # Tính khoảng cách bóng di chuyển (pixel và mét)
        distance_covered_by_ball_pixels = measure_distance(ball_mini_court_detections[start_frame][1],
                                                           ball_mini_court_detections[end_frame][1])
        distance_covered_by_ball_meters = convert_pixel_distance_to_meters(distance_covered_by_ball_pixels,
                                                                           constants.DOUBLE_LINE_WIDTH,
                                                                           mini_court.get_width_of_mini_court())

        # Tính tốc độ cú đánh bóng (km/h)
        speed_of_ball_shot = distance_covered_by_ball_meters / ball_shot_time_in_seconds * 3.6

        # Xác định cầu thủ thực hiện cú đánh bóng
        player_positons = player_mini_court_detections[start_frame]
        player_shot_ball = min(player_positons.keys(),
                               key=lambda player_id: measure_distance(player_positons[player_id],
                                                                      ball_mini_court_detections[start_frame][1]))

        # Tính tốc độ của cầu thủ đối thủ
        opponent_player_id = 1 if player_shot_ball == 2 else 2
        distance_covered_by_opponent_pixels = measure_distance(
            player_mini_court_detections[start_frame][opponent_player_id],
            player_mini_court_detections[end_frame][opponent_player_id]
        )
        distance_covered_by_opponent_meters = convert_pixel_distance_to_meters(distance_covered_by_opponent_pixels,
                                                                               constants.DOUBLE_LINE_WIDTH,
                                                                               mini_court.get_width_of_mini_court()
                                                                               )

        speed_of_opponent = distance_covered_by_opponent_meters / ball_shot_time_in_seconds * 3.6

        # Cập nhật dữ liệu thống kê của cầu thủ
        current_player_stats = deepcopy(player_stats_data[-1])
        current_player_stats['frame_num'] = start_frame
        current_player_stats[f'player_{player_shot_ball}_number_of_shots'] += 1
        current_player_stats[f'player_{player_shot_ball}_total_shot_speed'] += speed_of_ball_shot
        current_player_stats[f'player_{player_shot_ball}_last_shot_speed'] = speed_of_ball_shot

        current_player_stats[f'player_{opponent_player_id}_total_player_speed'] += speed_of_opponent
        current_player_stats[f'player_{opponent_player_id}_last_player_speed'] = speed_of_opponent

        player_stats_data.append(current_player_stats)

    # Chuyển đổi dữ liệu thống kê cầu thủ thành DataFrame và nối với khung hình video
    player_stats_data_df = pd.DataFrame(player_stats_data)
    frame_df = pd.DataFrame({'frame_num': list(range(len(video_frames)))})
    player_stats_data_df = pd.merge(frame_df, player_stats_data_df, on='frame_num', how='left')
    player_stats_data_df = player_stats_data_df.ffill()

    # Tính toán tốc độ trung bình của cú đánh và tốc độ cầu thủ
    player_stats_data_df['player_1_average_shot_speed'] = player_stats_data_df['player_1_total_shot_speed'] / \
                                                          player_stats_data_df['player_1_number_of_shots']
    player_stats_data_df['player_2_average_shot_speed'] = player_stats_data_df['player_2_total_shot_speed'] / \
                                                          player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_1_average_player_speed'] = player_stats_data_df['player_1_total_player_speed'] / \
                                                            player_stats_data_df['player_2_number_of_shots']
    player_stats_data_df['player_2_average_player_speed'] = player_stats_data_df['player_2_total_player_speed'] / \
                                                            player_stats_data_df['player_1_number_of_shots']

    # Vẽ các đối tượng trên video
    ## Vẽ khung hình bao quanh cầu thủ và bóng
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)

    ## Vẽ các điểm chính của sân
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    # Vẽ mini court và các điểm trên mini court
    output_video_frames = mini_court.draw_mini_court(output_video_frames)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, player_mini_court_detections)
    output_video_frames = mini_court.draw_points_on_mini_court(output_video_frames, ball_mini_court_detections,
                                                               color=(0, 255, 255))
    # Vẽ thống kê cầu thủ
    player_name = extract_player_names(input_video_path)
    output_video_frames = draw_player_stats(output_video_frames, player_stats_data_df, player_name)

    ## Vẽ số khung hình ở góc trên bên trái của từng khung hình video
    for i, frame in enumerate(output_video_frames):
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Lưu video đầu ra và trả về đường dẫn
    output_video_path = "output_videos/output_video.mp4"
    save_video(output_video_frames, output_video_path)
    return output_video_path


# Khởi tạo giao diện Gradio để chạy ứng dụng web
demo = gr.Interface(
    fn=main,
    inputs=[
        gr.Video(),  # Nhập video từ người dùng
        gr.Checkbox(label="Read from stubs", value=True)  # Tùy chọn để đọc dữ liệu từ các stub
    ],
    outputs="playable_video",  # Đầu ra là video có thể phát được
)

if __name__ == "__main__":
    demo.launch()  # Khởi chạy giao diện Gradio
