from deepcake.crop_video_faces import crop_video_faces
from deepcake.crop_video_faces import trim_video


trim_video(source_path = "data/videos/obama.mp4", start_time = 60 + 52, end_time = 130, target_path = "data/videos/obama_trimmed.mp4")
trim_video(source_path = "data/videos/elon.mp4", start_time = 10, end_time = 30, target_path = "data/videos/elon_trimmed.mp4")

crop_video_faces("data/videos/obama_trimmed.mp4", "data/cropped_frames/obama")
crop_video_faces("data/videos/elon_trimmed.mp4", "data/cropped_frames/elon")
