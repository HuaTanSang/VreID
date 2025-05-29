import cv2
import yaml
import time
import argparse
from kafka import KafkaProducer

class VideoKafkaProducer:
    def __init__(self, kafka_topic, cam_id, video_path, config_path):
        self.kafka_topic = kafka_topic
        self.cam_id = cam_id
        self.video_path = video_path

        # Load config từ YAML
        self.config = self.load_kafka_config(config_path)

        # Tạo Kafka Producer
        self.producer = KafkaProducer(
            bootstrap_servers=self.config['bootstrap_server'],
            key_serializer=lambda key: key.encode('utf-8'),
            value_serializer=lambda value: value,
            compression_type='gzip',
            linger_ms=10
        )

    def stream_video(self):
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"[ERROR] Không thể mở video {self.video_path}")
            return

        print(f"[START] Streaming từ {self.video_path} dưới ID {self.cam_id}...")

        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[END] Kết thúc video từ {self.cam_id}")
                break

            self.stream(frame)
            time.sleep(0.05)  # Delay nhỏ cho giả lập real-time

        cap.release()
        self.close()

    def stream(self, frame):
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        ret, buffer = cv2.imencode('.jpg', frame, encode_params)
        if not ret:
            print(f"[WARN] Không thể encode frame từ {self.cam_id}")
            return

        self.producer.send(
            self.kafka_topic,
            key=self.cam_id,
            value=buffer.tobytes()
        )
        self.producer.flush()

    def close(self):
        self.producer.close()

    @staticmethod
    def load_kafka_config(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)

# ---- Entry Point ----
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video Kafka Producer")
    parser.add_argument('--video_path', required=True, help='Đường dẫn video mô phỏng camera')
    parser.add_argument('--cam_id', required=True, help='Mã ID của camera')
    parser.add_argument('--kafka_topic', default='read', help='Kafka topic gửi dữ liệu đến')
    parser.add_argument('--config_path', default='kafka_config.yml', help='Đường dẫn đến file cấu hình Kafka')

    args = parser.parse_args()

    producer = VideoKafkaProducer(
        kafka_topic=args.kafka_topic,
        cam_id=args.cam_id,
        video_path=args.video_path,
        config_path=args.config_path
    )

    producer.stream_video()