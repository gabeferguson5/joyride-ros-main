# V2: IGVC Adaptation of ROI Method

import torch
import requests
import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose, ObjectHypothesis, BoundingBox2D, Point2D, Pose2D
from geometry_msgs.msg import PoseWithCovariance, Point

class TrafficSignDetector(Node):
    def __init__(self):
        super().__init__('yolov5_node')

        self.declare_parameter('pub_image', True)
        self.declare_parameter('pub_json', False)
        self.declare_parameter('pub_boxes', True)
        self.declare_parameter('pub_proc_image', True) # ADDED
        # self.declare_parameter('weights_path', '/home/joyride-obc/joyride-ros-main/src/joyride_perception/config/best.pt')
        self.declare_parameter('weights_path', '/home/joyride-obc/joyride-ros-main/src/joyride_perception/config/last.pt')
        self.subscription = self.create_subscription(
            Image,
            # '/sensors/cameras/lane/image_raw',
            '/sensors/cameras/center/image',
            self.listener_callback,
            1
        ) # may want to mess with the queue size of 1
        self.declare_parameter('pub_area_in_ROI', True) # ADDED
        self.declare_parameter('pub_contour_img', True) # ADDED
        self.declare_parameter('pub_height_stop', True) # ADDED

        self.image_publisher = self.create_publisher(Image, 'yolov5/image', 10)
        self.json_publisher = self.create_publisher(String, 'yolov5/json', 10)
        self.detection_publisher = self.create_publisher(Detection2DArray, 'yolov5/detection_boxes', 10)
        self.proc_image_publisher = self.create_publisher(Image, 'yolov5/proc_image',10) # ADDED
        self.area_publisher = self.create_publisher(Float32, 'SignDetector/StopSign_area', 10 ) # ADDED
        self.contour_publisher = self.create_publisher(Image, 'SignDetector/ContourImage', 10) # ADDED
        self.height_publisher = self.height_publisher(Float32, 'SignDetector/HeightStop', 10) # ADDED

        self.counter = 0
        self.br = CvBridge()

        #response = requests.get(self.get_parameter('weights_path').value)
        #open("data.pt", "wb").write(response.content)
        #self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./data.pt')

        # self.model = torch.hub.load('/home/joyride-obc/joyride-ros-main/src/yolov5', 'custom', path='/home/joyride-obc/joyride-ros-main/src/joyride_perception/config/best.pt')
        self.path_hubconfig = self.declare_parameter('model_repo_path', '/home/joyride-obc/joyride-ros-main/src/third_party/yolov5').get_parameter_value().string_value

       # - absolute path to best.pt
        self.path_trained_model = self.declare_parameter('model_weights_path', '/home/joyride-obc/joyride-ros-main/src/joyride_perception/config/best_bestdataset_YOLOV5.pt').get_parameter_value().string_value
        self.model = torch.hub.load(self.path_hubconfig, 'custom', path=self.path_trained_model, source='local', force_reload=False)
        self.get_logger().info("YOLOv5 Initialized")

    def inference(self, image: Image):
        return self.model(image)

    def getDetectionArray(self, df):
        dda = Detection2DArray()

        detections = []
        self.counter += 1

        for row in df.itertuples():
           
            # self.get_logger().info(f"Detected {row.name}")

            detection = Detection2D()

            detection.header.stamp = self.get_clock().now().to_msg()
            detection.header.frame_id = str(self.counter)

            hypothesises = []
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = str(row[6])
            hypothesis.hypothesis.score = float(row[5])

            pwc = PoseWithCovariance()
            pwc.pose.position = Point()
            pwc.pose.position.x = (int(row.xmin) + int(row.xmax)) / 2
            pwc.pose.position.y = (int(row.ymin) + int(row.ymax)) / 2

            hypothesis.pose = pwc

            hypothesises.append(hypothesis)
            detection.results = hypothesises

            bbox = BoundingBox2D()
            bbox.size_x = (int(row.xmax) - int(row.xmin)) / 2
            bbox.size_y = (int(row.ymax) - int(row.ymin)) / 2

            point = Point2D()
            point.x = (int(row.xmin) + int(row.xmax)) / 2
            point.y = (int(row.ymin) + int(row.ymax)) / 2

            center = Pose2D()
            center.position = point
            center.theta = 0.0

            detection.bbox = bbox

            detections.append(detection)



        dda.detections = detections
        dda.header.stamp = self.get_clock().now().to_msg()
        dda.header.frame_id = str(self.counter)

        # print("row")
        # print(row)

        return dda
       
    def add_bounding_box(self, image, results):
        for result in results.xyxy[0]:
            label = int(result[5])
            confidence = result[4]
            xmin, ymin, xmax, ymax = map(int, result[:4])

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {confidence:.2f}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
       
        return image
    
    def get_bbox_vals(self, results):
        for result_tuple in results:
            result = result_tuple
            xmin, ymin, xmax, ymax, _, _, = result
            w = xmax - xmin
            h = ymax - ymin
            bbox_vals = [xmin, ymax, w, h]
        return bbox_vals

    def contour_frame(self, current_frame, threshold1, threshold2):
        imgGray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
        imgCanny = cv2.Canny(imgBlur, threshold1, threshold2)
        processed_frame = imgCanny
        return processed_frame
    
    def get_contours(self, img, roi, bbox_info):
        total_area_roi = 0
        for bbox_tuple in bbox_info:
            bbox = bbox_tuple
            xb, yb, wb, hb = bbox
            roi = img[yb:yb+hb, xb:xb+wb]
            contoursROI, hierarchyROI = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contoursROI:
                area_roi = cv2.contourArea(cnt)
                if area_roi > 1000:
                    cv2.drawContours(roi, cnt, -1, (255, 0, 255), 7)
                    total_area_roi = total_area_roi + area_roi
        return total_area_roi
    
    def get_height(self, results):
        for result in results:
        xmin, ymin,xmax, ymax = map(int, result[:4])
        height = ymax - ymin
        return height

    def listener_callback(self, data): 
        #deleted row
        #self.get_logger().info('Got Image')
        current_frame = self.br.imgmsg_to_cv2(data)
        results = self.inference(current_frame)
 
        # set thresholds for contour
        threshold1 = 185
        threshold2 = 144

        imgContour = current_frame.copy()
        bbox_vals = self.get_bbox_vals(results)
        contoured_frame = self.contour_frame(current_frame, threshold1, threshold2)
        area_in_contours = self.get_contours(contoured_frame, imgContour, bbox_vals)

        if self.get_parameter('pub_image').value:
            #processed_image = self.br.cv2_to_imgmsg(results.ims[0])
            for result in results.xyxy[0]:
                label = int(result[5])
                confidence = result[4]
                xmin, ymin, xmax, ymax = map(int, result[:4])

                cv2.rectangle(current_frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(current_frame, f"{label} {confidence:.2f}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
               
                #cv2.putText(current_frame, f"{result.name} {confidence:.2f}", (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                #self.get_logger().info(f"Detected {result.name}")
                # Font size changed to 2 above
            #printing array for troubleshooting
            cv2.imshow('Processed Image', current_frame)
            cv2.waitKey(1)
            cv2.imwrite('/home/joyride-obc/joyride-ros-main/src/joyride_perception/first_frame4meter.jpg', current_frame)

            current_frame = self.br.cv2_to_imgmsg(current_frame)
            self.image_publisher.publish(current_frame)


            #self.image_publisher.publish(processed_image)

        if self.get_parameter('pub_json').value:
            json = String()
            json.data = results.pandas().xyxy[0].to_json(orient="records")
            self.json_publisher.publish(json)

        if self.get_parameter('pub_boxes').value:
            detections = self.getDetectionArray(results.pandas().xyxy[0])
            self.detection_publisher.publish(detections)

        # Publish the area value
        if self.get_parameter('pub_area_in_ROI').value:
            self.area_publisher.publish(area_in_contours)

        # Publish the contoured image
        if self.get_parameter('pub_contour_img').value:
            contoured_image = self.br.cv2_to_imgmsg(imgContour)
            self.contour_publisher.publish(contoured_image)

        if self.get_parameter('pub_height_stop').value:
            height = self.get_height(results)
            self.height_publisher.publish(height)

        # if self.get_parameter('pub_proc_image').value:
        #     image_with_bboxes = self.add_bounding_box(current_frame, results)
        #     image_with_bboxes_msg = self.br.cv2_to_imgmsg(image_with_bboxes)
        #     self.proc_image_publisher(image_with_bboxes_msg)
            #ros2 run rqt_image_view rqt_image_view



def main(args=None):
    rclpy.init(args=args)
    TDetector = TrafficSignDetector()
    rclpy.spin(TDetector)

    TDetector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
