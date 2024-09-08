import cv2
import sys
import numpy as np

sys.path.append('../') # Thêm đường dẫn tới thư mục cha vào sys.path để truy cập các module bên ngoài
import constants
from utils import (
    convert_pixel_distance_to_meters,
    convert_meter_to_pixel_distance,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)


class MiniCourt(): # Định nghĩa class MiniCourt để vẽ sân tennis thu nhỏ và các vị trí trên sân
    def __init__(self, frame):
        # Kích thước của khung hình sân thu nhỏ
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court = 20

        # Thiết lập vị trí nền và đường biên của sân
        self.set_canvas_background_box_position(frame)
        self.set_mini_court_positon()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    # Chuyển đổi khoảng cách từ mét -> pixel
    def convert_meters_to_pixels(self, meters):
        return convert_meter_to_pixel_distance(
            meters, constants.DOUBLE_LINE_WIDTH, self.court_drawing_width)
    # Thiết lập các điểm chính trên sân thu nhỏ
    def set_court_drawing_key_points(self):
        drawing_key_points = [0] * 28

        # point 0
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1]
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5]
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3]
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7]
        # #point 8
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17]
        # #point 10
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21]
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18]) / 2)
        drawing_key_points[25] = drawing_key_points[17]
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22]) / 2)
        drawing_key_points[27] = drawing_key_points[21]

        self.drawing_key_points = drawing_key_points

    # Thiết lập đường kẻ trên sân
    def set_court_lines(self):
        self.lines = [ # Định nghĩa các đường kẻ nối các điểm bằng tọa độ
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),

            (0, 1),
            (8, 9),
            (10, 11),
            (10, 11),
            (2, 3)
        ]

    # Thiết lập vị trí của sân mini dựa trên các điểm bắt đầu và kết thúc
    def set_mini_court_positon(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    # Thiết lập vị trí nền của sân
    def set_canvas_background_box_position(self, frame):
        frame = frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    # Hàm để vẽ sân tennis trên một khung hình
    def draw_court(self, frame):
        # Lặp qua các điểm chính để vẽ các đường tròn.
        for i in range(0, len(self.drawing_key_points), 2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i + 1])

            # Vẽ đường tròn màu đỏ cho từng điểm.
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        # Vẽ các đường nối giữa các điểm.
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0] * 2]), int(self.drawing_key_points[line[0] * 2 + 1]))
            end_point = (int(self.drawing_key_points[line[1] * 2]), int(self.drawing_key_points[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)
        # Vẽ lưới giữa sân.
        net_start_point = (
            self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    # Hàm để vẽ hình chữ nhật nền cho sân mini.
    def draw_background_rectangle(self, frame):
        shapes = np.zeros_like(frame, np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha = 0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    # Hàm để vẽ sân mini trên tất cả các khung hình.
    def draw_mini_court(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame) # Vẽ nền cho sân.
            frame = self.draw_court(frame)  # Vẽ sân tennis.
            output_frames.append(frame) # Lưu khung hình đã vẽ vào danh sách.
        return output_frames

    # Trả về điểm bắt đầu của sân mini.
    def get_start_point_of_mini_court(self):
        return self.court_start_x, self.court_start_y

    # Trả về chiều rộng của sân mini.
    def get_width_of_mini_court(self):
        return self.court_drawing_width

    # Trả về các điểm chính vẽ sân.
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    # Chuyển đổi tọa độ đối tượng sang tọa độ sân mini.
    def get_mini_court_coordinates(self, object_position, closest_key_point, closest_key_point_index,
                                   player_height_in_pixels, player_height_in_meters):
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position,
                                                                                               closest_key_point)
        # Chuyển đổi khoảng cách pixel sang mét.
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels)
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels)

        # Chuyển đổi sang tọa độ sân mini.
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_coourt_keypoint = (self.drawing_key_points[closest_key_point_index * 2],
                                        self.drawing_key_points[closest_key_point_index * 2 + 1]
                                        )

        mini_court_player_position = (closest_mini_coourt_keypoint[0] + mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1] + mini_court_y_distance_pixels
                                      )

        return mini_court_player_position

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_keypoints):
        # Tạo một dictionary chứa chiều cao của mỗi người chơi tính bằng mét.
        player_heights = {
            1: constants.PLAYER_1_HEIGHT_METERS,
            2: constants.PLAYER_2_HEIGHT_METERS,
        }
        output_player_boxes = []  # Khởi tạo danh sách để lưu tọa độ của người chơi trên sân mini.
        output_ball_boxes = []  # Khởi tạo danh sách để lưu tọa độ của quả bóng trên sân mini.

        # Duyệt qua từng khung hình (frame) và các bounding box của người chơi.
        for frame_num, player_bbox in enumerate(player_boxes):
            ball_box = ball_boxes[frame_num][1]  # Lấy bounding box của quả bóng trong khung hình hiện tại.
            ball_positon = get_center_of_bbox(ball_box)  # Tính tâm của bounding box quả bóng.

            # Tìm ID của người chơi gần nhất với quả bóng.
            closest_player_id_to_ball = min(player_bbox.keys(), key=lambda x: measure_distance(ball_positon,
                                                                                               get_center_of_bbox(
                                                                                                   player_bbox[x])))

            output_player_bboxes_dict = {}  # Khởi tạo dictionary để lưu tọa độ của người chơi trên sân mini.

            # Duyệt qua từng người chơi và tính toán tọa độ trên sân mini.
            for player_id, bbox in player_bbox.items():
                foot_positon = get_foot_position(bbox)  # Lấy vị trí của chân người chơi từ bounding box.

                # Lấy điểm chính trên sân gần nhất với vị trí chân người chơi.
                closest_key_point_index = get_closest_keypoint_index(foot_positon, original_court_keypoints,
                                                                     [0, 2, 12, 13])
                closest_key_point = (original_court_keypoints[closest_key_point_index * 2],
                                     original_court_keypoints[closest_key_point_index * 2 + 1])

                # Tính chiều cao của người chơi trong pixel.
                frame_index_min = max(0, frame_num - 20)  # Giới hạn chỉ số khung hình dưới.
                frame_index_max = min(len(player_boxes), frame_num + 50)  # Giới hạn chỉ số khung hình trên.
                # Lấy chiều cao của người chơi trong các khung hình xung quanh để tính toán.
                bboxes_height_in_pixels = [get_height_of_bbox(player_boxes[i][player_id]) for i in
                                           range(frame_index_min, frame_index_max)]
                max_player_height_in_pixels = max(
                    bboxes_height_in_pixels)  # Lấy chiều cao lớn nhất trong các khung hình.

                # Chuyển đổi tọa độ từ hệ tọa độ sân thực tế sang hệ tọa độ sân mini.
                mini_court_player_position = self.get_mini_court_coordinates(foot_positon,
                                                                             closest_key_point,
                                                                             closest_key_point_index,
                                                                             max_player_height_in_pixels,
                                                                             player_heights[player_id])

                output_player_bboxes_dict[
                    player_id] = mini_court_player_position  # Lưu vị trí người chơi trên sân mini.

                # Nếu người chơi gần nhất với quả bóng, chuyển đổi tọa độ quả bóng sang tọa độ sân mini.
                if closest_player_id_to_ball == player_id:
                    closest_key_point_index = get_closest_keypoint_index(ball_positon, original_court_keypoints,
                                                                         [0, 2, 12, 13])
                    closest_key_point = (original_court_keypoints[closest_key_point_index * 2],
                                         original_court_keypoints[closest_key_point_index * 2 + 1])

                    mini_court_player_position = self.get_mini_court_coordinates(ball_positon,
                                                                                 closest_key_point,
                                                                                 closest_key_point_index,
                                                                                 max_player_height_in_pixels,
                                                                                 player_heights[player_id]
                                                                                 )
                    output_ball_boxes.append({1: mini_court_player_position})  # Lưu vị trí quả bóng trên sân mini.
            output_player_boxes.append(output_player_bboxes_dict)  # Lưu vị trí người chơi cho từng khung hình.
        return output_player_boxes, output_ball_boxes  # Trả về tọa độ của người chơi và quả bóng trên sân mini.

    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0)):
        # Duyệt qua từng khung hình (frame) và các vị trí tương ứng.
        for frame_num, frame in enumerate(frames):
            for _, position in positions[frame_num].items():
                x, y = position  # Lấy tọa độ x, y của vị trí cần vẽ.
                x = int(x)  # Chuyển đổi x thành số nguyên.
                y = int(y)  # Chuyển đổi y thành số nguyên.
                cv2.circle(frame, (x, y), 5, color,
                           -1)  # Vẽ một vòng tròn trên khung hình tại vị trí (x, y) với màu sắc cho trước.
        return frames  # Trả về danh sách các khung hình đã được vẽ.

