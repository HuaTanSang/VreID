import cv2
import numpy as np
from kafka import KafkaConsumer
from pymongo import MongoClient
import datetime
import argparse

def main():
    parser = argparse.ArgumentParser(description="Kafka Consumer lưu frame vào MongoDB")
    parser.add_argument('--broker', default='localhost:9092', help='Kafka broker address')
    parser.add_argument('--topic', default='street-cameras', help='Kafka topic name')
    parser.add_argument('--mongo-uri', default='mongodb://localhost:27017/', help='MongoDB URI')
    parser.add_argument('--db', default='traffic_db', help='MongoDB database name')
    parser.add_argument('--collection', default='camera_frames', help='MongoDB collection name')
    args = parser.parse_args()

    # Kết nối MongoDB
    mongo_client = MongoClient(args.mongo_uri)
    db = mongo_client[args.db]
    collection = db[args.collection]

    # Tạo Kafka consumer
    consumer = KafkaConsumer(
        args.topic,
        bootstrap_servers=args.broker,
        auto_offset_reset='latest',
        enable_auto_commit=True,
        key_deserializer=lambda k: k.decode('utf-8'),
        value_deserializer=lambda v: v  # bytes JPEG
    )

    print("[START] Đang lắng nghe dữ liệu từ Kafka...")

    try:
        for msg in consumer:
            cam_id = msg.key
            jpg_bytes = msg.value

            # Decode JPEG → numpy array (OpenCV image)
            np_arr = np.frombuffer(jpg_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                print(f"[WARN] Không thể decode frame từ {cam_id}")
                continue
                    
            # Lưu thông tin và ảnh vào MongoDB
            record = {
                'camera_id': cam_id,
                'timestamp': datetime.datetime.utcnow(),
                'image': jpg_bytes,  # raw JPEG bytes
                'width': frame.shape[1],
                'height': frame.shape[0]
            }
            collection.insert_one(record)

            print(f"[OK] Lưu frame từ {cam_id} vào MongoDB")

    except KeyboardInterrupt:
        print("\n[STOP] Người dùng dừng consumer.")
    finally:
        mongo_client.close()

if __name__ == '__main__':
    main()
