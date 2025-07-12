"""
This file is to sending video from camera 2 to the kafka server. 
Acting as query video 
Topic: sending-cam2
"""

from producer import CameraProducer
import argparse

def main(): 
    parser = argparse.ArgumentParser(description="Sending frame from camera 2")  
    parser.add_argument("--video_path", required=True, help="Query video directory")
    parser.add_argument("--kafka_topic", required=True, default='sending-cam2', help="Kafka topic of the query camera")
    
    args = parser.parse_args() 

    try: 
        producer = CameraProducer(
            topic=args.kafka_topic, 
            cam_id='cam2', 
            video_path=args.video_path, 
        )

        print(f"[INFO] Starting streaming video from camera 2")
        producer.streaming_video() 

    except Exception as e: 
        print(f"[ERROR] {e}")

    finally: 
        producer.close() 
        print("[INFO] Streaming ended and producer closed")

if __name__ == "__main__": 
    main() 