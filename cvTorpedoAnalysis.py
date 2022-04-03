#! /usr/bin/env python3

import cv2 as cv
import glob
from copy import deepcopy
import numpy as np
import os

from FeatureDetection import ImageFeatureProcessing

class TorpedoAnalysis:
    def __init__(self, video_fn, y_coord, x_spacing, frames_to_hop, hsv_params, morphism_params, bgr_params, viz, x_offset, param_str=""):
        self.video_fn = video_fn
        self.vidcap = cv.VideoCapture(self.video_fn)
        self.total_frames = self.vidcap.get(7)
        self.fps = round((self.vidcap.get(cv.CAP_PROP_FPS)), 1)
        self.vid_length = self.total_frames / self.fps
        print(f"NOW PROCESSING: {self.video_fn}\nTotal Frames: {self.total_frames}\nVideo FPS:{self.fps}\nVideo Length: {self.vid_length}\n")

        self.y_coord = y_coord
        self.x_spacing = x_spacing
        self.x_offset = x_offset
        self.clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8,8))

        self.frames_to_hop = frames_to_hop
        self.viz = viz
        if self.vidcap.isOpened():
            self.ret, self.frame = self.vidcap.read()
        self.feature_detection = ImageFeatureProcessing(self.frame.shape)

        self.hsv_params = hsv_params
        self.morphism_params = morphism_params
        self.bgr_params = bgr_params
        self.param_str = param_str

    def __del__(self):
        self.vidcap.release()
        cv.destroyAllWindows()

    
    def read_frame(self):
        if self.vidcap.isOpened():
            self.ret, self.frame = self.vidcap.read()

    def throw_out_frame(self):
        if self.vidcap.isOpened():
            self.vidcap.grab()

    def draw_hor_line(self, img, y_coord):
        cv.line(img,(0, img.shape[0] - y_coord),(img.shape[1], img.shape[0] - y_coord),(255,0,0), 5)

    def draw_vert_lines(self, img, spacing, set_in_middle=True):
        self.num_of_lines = img.shape[1] // spacing

        if set_in_middle:
            for i in range(self.num_of_lines // 2 + 1):
                if i < 1:
                    color = (255, 255, 255)
                else:
                    color = (0, 255, 0)

                cv.line(img,(i * spacing + img.shape[1] // 2 + self.x_offset, 0),(i * spacing + img.shape[1] // 2 + self.x_offset, img.shape[0]),color, 3)
                cv.line(img,(img.shape[1] // 2 - i * spacing + self.x_offset, 0),(img.shape[1] // 2 - i * spacing + self.x_offset, img.shape[0]),color, 3)

    def poi(self, img, det_x):
        self.num_of_lines = img.shape[1] // self.x_spacing
        self.middle_line = self.num_of_lines / 2
        self.middle_px = img.shape[1] // 2 + self.x_offset

        offset = (det_x - self.middle_px)
        self.point_offset = round((offset / self.x_spacing), 2)

    def BGR_filter(self, img, bgr_min, bgr_max):
        # preparing the mask to overlay
        mask = cv.inRange(img, bgr_min, bgr_max)
        
        result = cv.bitwise_and(img, img, mask = mask)
        return mask, result
    
    def CLAHE(self, img):
        equ = np.zeros(np.shape(img))
        
        if len(np.shape(img)) == 2: # If it is a grayscale/one channel image
            equ = self.clahe.apply(img)
            return equ
        elif len(np.shape(img)) < 2:
            raise IndexError("Image must be at least a 2D array")
        else:
            for i in range(np.shape(img)[2]):
                equ[:,:,i] = self.clahe.apply(img[:,:,i])
            return equ
    
    def run(self):
        fail_counter = 0
        frame_counter = 0
        detection = False

        while self.vidcap.isOpened():
            frame_counter += 1

            if frame_counter % self.frames_to_hop != 0:
                self.throw_out_frame()
                continue

            print(frame_counter)
            self.read_frame()
            if not self.ret:
                if fail_counter < 100:
                    fail_counter += 1
                    continue
                else:
                    print("End of stream. Exiting...")
                    break
            orig_frame = deepcopy(self.frame)
            orig_frame = (self.CLAHE(orig_frame)).round().astype(np.uint8)
            
            #================================> Processing here

            _, hsv_mask, hsv_mask_validation_img = self.feature_detection.hsv_processor(orig_frame, *self.hsv_params)

            bgr_mask, bgr_filt_img = self.BGR_filter(hsv_mask_validation_img, self.bgr_params[0], self.bgr_params[1])
            
            gray = cv.cvtColor(bgr_filt_img, cv.COLOR_BGR2GRAY)

            morph_img = self.feature_detection.noise_removal_processor(gray, *self.morphism_params)

            # DETECTION
            fixed_y_coord = orig_frame.shape[0] - self.y_coord         
            crop = morph_img[fixed_y_coord:fixed_y_coord+1, 0:0+orig_frame.shape[1]]
            if crop.sum() > 0:
                # Save image
                detection = True
                det_name = self.video_fn[:len(self.video_fn) - 4]

                det_name_png = f"{det_name}.png"
                det_frame = orig_frame
                
                torpedo_pxels = np.nonzero(crop)[1]
                torpedo_mean = int(round(np.mean(torpedo_pxels), 0))

                self.poi(orig_frame, torpedo_mean)

                cv.circle(det_frame, (torpedo_mean, fixed_y_coord), 3, (255, 255, 255), 3)
                cv.putText(det_frame, str(self.point_offset), (torpedo_mean, fixed_y_coord), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                y0, dy = 25, 30
                for i, line in enumerate(self.param_str.split('\n')):
                    y = y0 + i*dy
                    cv.putText(det_frame, line, (0, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv.imwrite(det_name_png, orig_frame)

                with open(det_name + ".txt", 'w') as file:
                    file.write(str(self.param_str) + str("\n"))
                    file.write(str(torpedo_mean) + str("\n"))
                    file.write(str(self.point_offset) + str("\n"))

                break

            # try:
            #     img_cnts, blank_image, contours = self.feature_detection.contour_processing(morph_img, 100, enable_convex_hull=False, return_image=True, image=orig_frame, show_centers=True, show_areas=True)
            # except TypeError:
            #     pass
            
            #================================> End of processing (Just vizualization) 
            if self.viz:
                self.draw_hor_line(orig_frame, self.y_coord)
                self.draw_vert_lines(orig_frame, self.x_spacing)
                cv.imshow('frame', orig_frame)            
                cv_wk = cv.waitKey(1)
                if cv_wk == ord('q'):
                    print("Exiting...")
                    break
                elif cv_wk == ord('p'):
                    cv.waitKey(-1)


        self.__del__()

if __name__ == "__main__":
    # Scale params
    y_coord = 315
    x_spacing = 84
    x_offset = 35

    # Video params
    frames_to_hop = 3
    viz = True

    # Detection params
    hsv_params = (0, 200, 200, 255, 120, 255)
    morphism_params = (9, 9, 5.0, 21, 2, 2, 4, 7)
    bgr_params = ((0,0,0), (50, 255, 255))

    # Param str
    param_str = f"Hor-line Y-coord: {y_coord}\n" \
                f"Ver-line X-spacing: {x_spacing}\n" \
                f"Frames hop: {frames_to_hop}\n" \
                f"HSV params: {hsv_params}\n" \
                f"Morphism params: {morphism_params}\n" \
                f"BGR filter params: {bgr_params}\n"

    # # example run
    ta = TorpedoAnalysis("videos/GOPR0008.MP4", y_coord, x_spacing, frames_to_hop, hsv_params, morphism_params, bgr_params, viz, x_offset, param_str=param_str)
    ta.run()

    # # auto run
    # vid_names = glob.glob(os.getcwd() + "/videos/*.MP4")

    # for fn in vid_names:
    #     ta = TorpedoAnalysis(fn, y_coord, x_spacing, frames_to_hop, hsv_params, morphism_params, bgr_params, viz, x_offset, param_str=param_str)
    #     ta.run()

