import time
import cv2
from kafka import KafkaProducer

class CameraProducer:
    def __init__(
        self,
        topic: str,
        cam_id: str,
        video_path: str,
        bootstrap_servers: str = 'localhost:9092'
    ):
        
        """
        topic: Kafka topic để gửi frame
        cam_id: ID của camera (cam1 cho register, cam2 cho query)
        video_path: đường dẫn tới video
        """
        self.topic = topic
        self.cam_id = cam_id
        self.video_path = video_path
        self.bootstrap_servers = bootstrap_servers

        self.producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            key_serializer=lambda key: key.encode('utf-8'),
            value_serializer=lambda value: value
        )

    def streaming_video(self, skip_frame: int = 2, sleep_time: float = 0.7):
        video = cv2.VideoCapture(self.video_path)

        if not video.isOpened():
            raise Exception(f"Không mở được video: {self.video_path}")

        frame_id = 0
        while True:
            frame_id += 1
            if not video.grab():
                continue

            if frame_id % skip_frame != 0:
                continue

            ret, frame = video.retrieve()
            if not ret or frame is None:
                print(f"[WARNING] Failed to retrieve frame {frame_id}")
                continue

            success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not success:
                print(f"[ERROR] Failed to encode frame {frame_id}")
                continue

            self.producer.send(self.topic, key=self.cam_id, value=buffer.tobytes())
            print(f"[STREAMING][{self.cam_id}] Sent frame {frame_id}")
            time.sleep(sleep_time)
            self.producer.flush()   
        
        video.release()


    def close(self):
        """Đóng producer khi kết thúc"""
        self.producer.close()
