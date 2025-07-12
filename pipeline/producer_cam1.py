"""
Gửi video từ camera 1 (register) lên Kafka topic.
Chạy:
  python3 producer_cam1.py --kafka_topic sending-cam1 --video_path /path/to/video.mp4
"""

from producer import CameraProducer
import argparse

def main():
    parser = argparse.ArgumentParser(description="Streaming từ camera 1 lên Kafka")
    parser.add_argument(
        "--video_path", required=True,
        help="Đường dẫn tới file video cần phát"
    )
    parser.add_argument(
        "--kafka_topic", required=True,
        help="Kafka topic để publish frame (ví dụ: sending-cam1)"
    )
    args = parser.parse_args()

    producer = None
    try:
        producer = CameraProducer(
            topic=args.kafka_topic,
            cam_id='cam1',
            video_path=args.video_path,
        )
        print(f"[INFO] Bắt đầu streaming camera 1 → topic '{args.kafka_topic}'")
        producer.streaming_video()

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        if producer:
            producer.close()
            print("[INFO] Đã đóng producer")

if __name__ == "__main__":
    main()
