import cv2 
import threading
import numpy as np 
import time 

from queue import Queue
from kafka import KafkaConsumer


processed_images = Queue()
frame_counter = 0 # for memory robustness


def consume_massages(consumer: KafkaConsumer):
    global frame_counter

    try:
        for message in consumer:
            try:
                if not message.value:
                    print(f"[ERROR] Empty message received from topic: {message.topic}")
                    continue

                frame_buffer = np.frombuffer(message.value, dtype=np.uint8)
                if frame_buffer.size == 0:
                    print(f"[ERROR] Invalid buffer size from topic: {message.topic}")
                    continue

                image = cv2.imdecode(frame_buffer, cv2.IMREAD_COLOR)
                if image is None:
                    print(f"[ERROR] Cannot decode frame from topic: {message.topic}")
                    continue

                processed_images.put((message.topic, image))
                frame_counter += 1
                print(f"[INFO] Processed frame from topic: {message.topic}, total frames: {frame_counter}")

                if frame_counter % 100 == 0:
                    print(f"[INFO] Processed {frame_counter} frames so far.")
                    import gc
                    gc.collect()

            except Exception as e:
                print(f"[ERROR] Exception while processing message from {message.topic}: {e}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("[INFO] Consumer interrupted by user.")
    finally:
        consumer.close()
        print("[INFO] Consumer closed.")


def thread_starter(consumer1: KafkaConsumer
                    ,consumer2: KafkaConsumer):
    
    """
    Start the consumer threads for two Kafka topics.
    """

    thread1 = threading.Thread(target=consume_massages, args=(consumer1,))
    thread2 = threading.Thread(target=consume_massages, args=(consumer2,))

    thread1.start()
    time.sleep(3)  # Ensure the first thread starts before the second
    thread2.start()

    return thread1, thread2

def display_images():
    """
    Continuously display frames from the processed_images queue.
    Shows frames in separate windows based on the topic name.
    Press 'q' to quit.
    """
    
    try:
        print("[INFO] Starting display loop.")
        while True:
            try:
                topic, frame = processed_images.get(timeout=1)  # wait for frame 1 second 
                if frame is None:
                    print(f"[WARNING] Received empty frame from topic {topic}")
                    continue

                # Resize nếu muốn dễ xem hơn
                resized_frame = cv2.resize(frame, (640, 480))

                # Mỗi topic hiển thị trong một cửa sổ riêng
                cv2.imshow(f"{topic}", resized_frame)

                # Thoát nếu nhấn 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Quit signal received. Exiting display loop...")
                    break

            except Exception:
                continue  # nếu queue rỗng thì tiếp tục

    except KeyboardInterrupt:
        print("[INFO] Display interrupted by user.")

    finally:
        # Đóng toàn bộ cửa sổ khi kết thúc
        cv2.destroyAllWindows()
        print("[INFO] All OpenCV windows closed.")


def main():
    from streaming_detection import start_streaming

    # Start the streaming process in a separate thread
    spark_thread = threading.Thread(target=start_streaming)
    spark_thread.start()

    # Initialize Kafka consumers
    consumer_cam1 = KafkaConsumer(
        "receive-cam1",  # Pass topic as positional argument
        bootstrap_servers="localhost:9092",
        auto_offset_reset="latest",  # Changed to 'latest' for fresh messages
        enable_auto_commit=True
    )
    consumer_cam2 = KafkaConsumer(
        "receive-cam2",
        bootstrap_servers="localhost:9092",
        auto_offset_reset="latest",
        enable_auto_commit=True
    )

    # Start the consumer threads
    thread1, thread2 = thread_starter(consumer_cam1, consumer_cam2)
    
    # Start displaying images
    display_images()

    # Wait for threads to complete
    thread1.join()
    thread2.join()
    spark_thread.join()


if __name__ == "__main__":
    main()