import rclpy, time, math
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge
from ultralytics import YOLO

rclpy.init()
node = Node('final_demo')
pub = node.create_publisher(Twist, '/fastbot_1/cmd_vel', 10)
bridge = CvBridge()
model = YOLO('yolov8n.pt')
scan_d = [None]
img_d = [None]
node.create_subscription(LaserScan, '/fastbot_1/scan',
                         lambda m: scan_d.__setitem__(0, m), 1)
node.create_subscription(Image, '/fastbot_1/camera/image_raw',
                         lambda m: img_d.__setitem__(0, m), 1)

def move(lin, ang, sec):
    msg = Twist()
    msg.linear.x = lin
    msg.angular.z = ang
    end = time.time() + sec
    while time.time() < end:
        pub.publish(msg)
        time.sleep(0.1)
    pub.publish(Twist())
    time.sleep(1.0)

def wait(store):
    store[0] = None
    t = time.time() + 5
    while store[0] is None and time.time() < t:
        rclpy.spin_once(node, timeout_sec=0.5)
    return store[0]

def find_target(m):
    best = None
    for i, r in enumerate(m.ranges):
        if math.isfinite(r) and 1.0 < r < 5.0:
            if best is None or r < best[0]:
                best = (r, m.angle_min + i * m.angle_increment)
    return best

time.sleep(1)
for rnd in range(6):
    print('--- 第', rnd + 1, '回 ---')
    m = wait(scan_d)
    b = find_target(m) if m else None
    print('目標:', b)
    if b is None:
        move(0.0, 0.5, 3.0)
        continue
    d, a = b
    move(0.0, 0.5 if a > 0 else -0.5, abs(a) / 0.5)
    move(0.2, 0.0, (d - 0.4) / 0.2)
    im = wait(img_d)
    if im is not None:
        img = bridge.imgmsg_to_cv2(im, 'bgr8')
        res = model(img, conf=0.15, verbose=False)[0]
        print('AI検出:', [res.names[int(c)] for c in res.boxes.cls])
    move(0.0, 0.5, 2.0)
print('全完了')
pub.publish(Twist())
node.destroy_node()
rclpy.shutdown()
