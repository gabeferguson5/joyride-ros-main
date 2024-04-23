
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# Tkinter import
import tkinter as tk
from tkinter import ttk

class SliderNode(Node):
    def __init__(self):
        super().__init__('slider_node')
        self.threshold_pub = self.create_publisher(Float32MultiArray, '/hsv_thresholds', 10)
        self.sliders = [] #initialize sliders list

        # Initialize Tkinter root window
        self.root = tk.Tk()
        self.root.title('HSV Threshold Sliders')

        # Create sliders for lower and upper bounds of each HSV component
        self.create_slider_pair('Hue', 0, 179, 10, 4, 6)
        self.create_slider_pair('Saturation', 0, 255, 10, 50, 255)
        self.create_slider_pair('Value', 0, 255, 10, 50, 255)

        self.root.mainloop() #start Tkinter event loop

    def create_slider_pair(self, label, min_val, max_val, resolution, default_low, default_high):
        # Create a frame for the sliders
        slider_frame = ttk.Frame(self.root)
        slider_frame.pack(fill='x', padx=10, pady=5)

        # Create labels for lower and upper bounds
        label_lower = ttk.Label(slider_frame, text=label + ' Lower:')
        label_lower.grid(row=0, column=0, padx=(0,10))
        label_upper = ttk.Label(slider_frame, text=label + ' Upper:')
        label_upper.grid(row=0, column=2, padx=(10,0))

        #create labels to display current values
        value_lower=ttk.Label(slider_frame, text=str(min_val))
        value_lower.grid(row=0, column=1, padx=(0,10))

        value_upper=ttk.Label(slider_frame, text=str(max_val))
        value_upper.grid(row=0, column=3, padx=(10,0))

        # Create sliders for lower and upper bounds
        slider_lower = ttk.Scale(slider_frame, from_=min_val, to=max_val, length=200, orient='horizontal', command=self.slider_changed)
        slider_lower.set(default_low)
        slider_lower.grid(row=1, column=0, columnspan=2)

        slider_upper = ttk.Scale(slider_frame, from_=min_val, to=max_val, length=200, orient='horizontal', command=self.slider_changed)
        slider_upper.set(default_high)
        slider_upper.grid(row=1, column=2, columnspan=2)

        self.sliders.append((slider_lower, slider_upper, value_lower, value_upper))

    def slider_changed(self, event):
        #Update the labels with the current slider values
        for slider_lower, slider_upper, value_lower, value_upper in self.sliders:
            value_lower.config(text=str(slider_lower.get()))
            value_upper.config(text=str(slider_upper.get()))
        # Publish updated threshold values when sliders are moved
        thresholds = []
        for slider_lower, slider_upper, value_lower, value_upper in self.sliders:
            thresholds.append(slider_lower.get())
            thresholds.append(slider_upper.get())
        threshold_msg = Float32MultiArray()
        threshold_msg.data = thresholds
        self.threshold_pub.publish(threshold_msg)

def main():
    rclpy.init()
    slider_node = SliderNode()
    rclpy.spin(slider_node)
    slider_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()