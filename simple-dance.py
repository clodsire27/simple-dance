import cv2
import mediapipe as mp
import numpy as np
import json

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    enable_segmentation=False, 
    smooth_landmarks=True
)
mp_drawing = mp.solutions.drawing_utils


reference_video_path = r'C:\Users\hapeo\Desktop\20241121.mp4'
cap_reference = cv2.VideoCapture(reference_video_path)
if not cap_reference.isOpened():
    print(f"Error: Could not open video file {reference_video_path}")
    exit()

keypoints_list = []
while cap_reference.isOpened():
    ret, frame = cap_reference.read()
    if not ret:
        break


    frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (640, 640)) 
    result = pose.process(frame_rgb)

    if result.pose_landmarks:
        keypoints = [(lm.x, lm.y, lm.z) for lm in result.pose_landmarks.landmark]
        keypoints_list.append(keypoints)

cap_reference.release()

if not keypoints_list:
    print("Error: No keypoints extracted from the reference video.")
    exit()

# 키포인트 저장
with open(r'C:\Users\hapeo\Desktop\reference_keypoints.json', 'w') as f:
    json.dump(keypoints_list, f)

# 기준 키포인트 로드
with open(r'C:\Users\hapeo\Desktop\reference_keypoints.json', 'r') as f:
    reference_keypoints = json.load(f)

# 카메라 입력
cap_camera = cv2.VideoCapture(0)
fps = int(cap_camera.get(cv2.CAP_PROP_FPS))
if fps == 0:
    print("Warning: Could not retrieve FPS. Setting default FPS to 30.")
    fps = 30
delay = int(1000 / fps)
frame_idx = 0
total_error = 0
frame_interval = max(1, int(fps / 10))  # 초당 10프레임 처리

# 주요 설정
important_landmarks = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
important_angle_indices = [(11, 13, 15), (12, 14, 16), (23, 25, 27), (24, 26, 28)]
start_frame = 30
end_frame = 120


def calculate_angle(a, b, c):
    ab = np.array(b) - np.array(a)
    cb = np.array(b) - np.array(c)
    angle = np.arccos(
        np.clip(np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb)), -1.0, 1.0)
    )
    return np.degrees(angle)


# 두 영상을 동시에 재생
cap_reference = cv2.VideoCapture(reference_video_path)

while cap_camera.isOpened() and cap_reference.isOpened():
    ret_camera, frame_camera = cap_camera.read()
    ret_reference, frame_reference = cap_reference.read()

    if not ret_camera or not ret_reference:
        break

    frame_idx += 1
    if frame_idx % frame_interval != 0 or frame_idx < start_frame or frame_idx > end_frame:
        continue

    # 프레임 크기 고정
    frame_camera = cv2.resize(frame_camera, (FRAME_WIDTH, FRAME_HEIGHT))
    frame_reference = cv2.resize(frame_reference, (FRAME_WIDTH, FRAME_HEIGHT))

    # 기준 영상 스켈레톤 시각화
    frame_rgb_reference = cv2.cvtColor(frame_reference, cv2.COLOR_BGR2RGB)
    result_reference = pose.process(frame_rgb_reference)
    if result_reference.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame_reference,
            result_reference.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
        )

    # 실시간 카메라 스켈레톤 시각화
    frame_rgb_camera = cv2.cvtColor(frame_camera, cv2.COLOR_BGR2RGB)
    result_camera = pose.process(frame_rgb_camera)

    if result_camera.pose_landmarks and frame_idx < len(reference_keypoints):
        user_keypoints = [(lm.x, lm.y, lm.z) for lm in result_camera.pose_landmarks.landmark]
        ref_keypoints = reference_keypoints[frame_idx]

        if len(user_keypoints) != len(ref_keypoints):
            print(f"Keypoint mismatch at frame {frame_idx}. Skipping.")
            continue

        # 가중치 적용 거리 계산
        weights = [1.0 if i in important_landmarks else 0.5 for i in range(len(user_keypoints))]
        frame_error = np.mean([
            weights[i] * np.linalg.norm(np.array(u) - np.array(r))
            for i, (u, r) in enumerate(zip(user_keypoints, ref_keypoints))
        ])

        # 각도 기반 오류 계산
        angles_error = np.mean([
            abs(calculate_angle(user_keypoints[i], user_keypoints[j], user_keypoints[k]) -
                calculate_angle(ref_keypoints[i], ref_keypoints[j], ref_keypoints[k]))
            for (i, j, k) in important_angle_indices
        ])

        # 총 오류 업데이트
        total_error += frame_error + angles_error

        # 실시간 점수 계산
        current_score = max(0, 100 - (total_error / (frame_idx + 1)) * 10)

        # 스켈레톤 그리기
        mp_drawing.draw_landmarks(
            frame_camera,
            result_camera.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
        )

        # 점수 표시
        cv2.putText(frame_camera, f"Score: {current_score:.2f}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    combined_frame = cv2.hconcat([frame_reference, frame_camera])
    cv2.imshow("Reference Video vs Real-time Pose Comparison", combined_frame)

    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

cap_camera.release()
cap_reference.release()
cv2.destroyAllWindows()

if frame_idx >= len(reference_keypoints):
    final_score = max(0, 100 - (total_error / len(reference_keypoints)) * 10)
else:
    final_score = max(0, 100 - (total_error / frame_idx) * 10)

print(f"Final Score: {final_score:.2f}")
