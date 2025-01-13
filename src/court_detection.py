import sys
sys.path.append('/content/TennisProject/src')
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sympy import Line
from itertools import combinations
from court_reference import CourtReference
import scipy.signal as sp
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn.functional as F


class CourtDetector:
    """
    Detecting and tracking court in frame
    """
    def __init__(self, verbose=0):
        self.verbose = verbose
        self.colour_threshold = 200
        self.dist_tau = 3
        self.intensity_threshold = 40
        self.court_reference = CourtReference()
        self.v_width = 0
        self.v_height = 0
        self.frame = None
        self.gray = None
        self.court_warp_matrix = []
        self.game_warp_matrix = []
        self.court_score = 0
        self.baseline_top = None
        self.baseline_bottom = None
        self.net = None
        self.left_court_line = None
        self.right_court_line = None
        self.left_inner_line = None
        self.right_inner_line = None
        self.middle_line = None
        self.top_inner_line = None
        self.bottom_inner_line = None
        self.success_flag = False
        self.success_accuracy = 80
        self.success_score = 1000
        self.best_conf = None
        self.frame_points = None
        self.dist = 5
        # Use GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def detect(self, frame, verbose=0):
        """
        Detecting the court in the frame
        """
        try:
            self.verbose = verbose

            # Validate input frame
            if frame is None or frame.size == 0:
                print("Invalid input frame")
                return False

            # Get original dimensions
            self.v_height, self.v_width = frame.shape[:2]

            # Ensure minimum frame size
            min_size = 100  # Minimum dimension size
            if self.v_height < min_size or self.v_width < min_size:
                print(f"Frame too small: {self.v_width}x{self.v_height}")
                return False

            # Calculate scale factor while ensuring minimum size
            target_width = 800  # Target width for processing
            scale_factor = min(1.0, target_width / self.v_width)

            # Ensure scaled dimensions are valid
            new_width = max(min_size, int(self.v_width * scale_factor))
            new_height = max(min_size, int(self.v_height * scale_factor))

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (new_width, new_height))
            self.frame = frame  # Keep original frame

            # Convert frame to GPU tensor for faster processing
            try:
                frame_tensor = torch.from_numpy(small_frame).to(self.device)
                if len(frame_tensor.shape) == 3:
                    frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0).float()
            except Exception as e:
                print(f"Error converting frame to tensor: {str(e)}")
                return False

            # Get binary image from the frame using GPU
            try:
                self.gray = self._threshold_gpu(frame_tensor)
            except Exception as e:
                print(f"Error in thresholding: {str(e)}")
                return False

            # Filter pixel using the court known structure
            try:
                filtered = self._filter_pixels_gpu(self.gray)
                filtered_cpu = filtered.cpu().numpy().astype(np.uint8)
            except Exception as e:
                print(f"Error in pixel filtering: {str(e)}")
                return False

            # Detect lines using optimized Hough transform parameters
            try:
                horizontal_lines, vertical_lines = self._detect_lines_parallel(filtered_cpu)
            except Exception as e:
                print(f"Error in line detection: {str(e)}")
                return False

            if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
                print("Not enough lines detected for court detection")
                return False

            # Find transformation from reference court to frame's court
            try:
                court_warp_matrix, game_warp_matrix, self.court_score = self._find_homography_parallel(
                    horizontal_lines,
                    vertical_lines,
                    scale_factor
                )
            except Exception as e:
                print(f"Error in homography computation: {str(e)}")
                return False

            if court_warp_matrix is None:
                print("Could not find valid homography")
                return False

            # Scale back the transformation matrix
            try:
                court_warp_matrix = court_warp_matrix * np.array([[1/scale_factor, 0, 1/scale_factor],
                                                                [0, 1/scale_factor, 1/scale_factor],
                                                                [0, 0, 1]])
                game_warp_matrix = game_warp_matrix * np.array([[scale_factor, 0, scale_factor],
                                                                  [0, scale_factor, scale_factor],
                                                                  [0, 0, 1]])
            except Exception as e:
                print(f"Error scaling transformation matrix: {str(e)}")
                return False

            self.court_warp_matrix.append(court_warp_matrix)
            self.game_warp_matrix.append(game_warp_matrix)

            try:
                court_accuracy = self._get_court_accuracy(0)
            except Exception as e:
                print(f"Error computing court accuracy: {str(e)}")
                return False

            if court_accuracy > self.success_accuracy and self.court_score > self.success_score:
                self.success_flag = True
                print(f'Court detected successfully with accuracy = {court_accuracy:.2f}%')
                try:
                    self.find_lines_location()
                except Exception as e:
                    print(f"Error finding lines location: {str(e)}")
                    return False
                return True
            else:
                print(f'Court detection failed with accuracy = {court_accuracy:.2f}%')
                return False

        except Exception as e:
            print(f"Unexpected error in court detection: {str(e)}")
            return False

    def _threshold_gpu(self, frame_tensor):
        """
        GPU-accelerated thresholding
        """
        try:
            if len(frame_tensor.shape) == 4:  # If RGB
                # Convert to grayscale using GPU
                gray_weights = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1).to(self.device)
                gray_tensor = (frame_tensor * gray_weights).sum(dim=1, keepdim=True)
            else:
                gray_tensor = frame_tensor

            # Threshold
            threshold_value = self.colour_threshold
            binary_tensor = (gray_tensor > threshold_value).float() * 255
            return binary_tensor

        except Exception as e:
            print(f"Error in GPU thresholding: {str(e)}")
            raise

    def _filter_pixels_gpu(self, gray_tensor):
        """
        GPU-accelerated pixel filtering
        """
        # Create kernels for vertical and horizontal filtering
        kernel_size = self.dist_tau * 2 + 1
        vertical_kernel = torch.zeros((1, 1, kernel_size, 1)).to(self.device)
        horizontal_kernel = torch.zeros((1, 1, 1, kernel_size)).to(self.device)
        vertical_kernel[0, 0, 0, 0] = vertical_kernel[0, 0, -1, 0] = 1
        horizontal_kernel[0, 0, 0, 0] = horizontal_kernel[0, 0, 0, -1] = 1

        # Apply convolution for edge detection
        vertical_edges = F.conv2d(gray_tensor, vertical_kernel, padding=(kernel_size//2, 0))
        horizontal_edges = F.conv2d(gray_tensor, horizontal_kernel, padding=(0, kernel_size//2))

        # Combine edges
        edges = torch.max(vertical_edges, horizontal_edges)
        filtered = (edges > self.intensity_threshold).float() * gray_tensor

        return filtered

    def _detect_lines_parallel(self, gray):
        """
        Parallel line detection using multiple threads
        """
        def process_region(region):
            if region is None or region.size == 0:
                return []

            # Ensure region has valid dimensions
            if region.shape[0] < 2 or region.shape[1] < 2:
                return []

            try:
                # Apply additional preprocessing to enhance line detection
                region = cv2.GaussianBlur(region, (3, 3), 0)
                edges = cv2.Canny(region, 50, 150, apertureSize=3)

                # Detect lines with more lenient parameters
                lines = cv2.HoughLinesP(
                    edges,
                    rho=1,
                    theta=np.pi/180,
                    threshold=30,  # Reduced threshold
                    minLineLength=int(region.shape[1] * 0.1),  # Reduced minimum line length
                    maxLineGap=int(region.shape[1] * 0.03)  # Increased max gap
                )

                if lines is None:
                    return []

                # Filter lines by angle and length
                filtered_lines = []
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)

                    # Keep lines that are roughly horizontal (0±20°) or vertical (90±20°)
                    if (length > region.shape[1] * 0.1 and  # Minimum length check
                        ((angle < 20) or (angle > 70 and angle < 110) or (angle > 160))):
                        filtered_lines.append(line[0])

                return np.array(filtered_lines) if filtered_lines else []

            except cv2.error as e:
                print(f"OpenCV error in line detection: {str(e)}")
                return []
            except Exception as e:
                print(f"Error in line detection: {str(e)}")
                return []

        # Split image into regions for parallel processing
        try:
            # Use more regions for better coverage
            num_regions = 6
            region_height = max(2, gray.shape[0] // num_regions)
            regions = []

            # Add overlap between regions
            overlap = int(region_height * 0.2)
            for i in range(num_regions):
                start = max(0, i * region_height - overlap)
                end = min(gray.shape[0], (i + 1) * region_height + overlap)
                if end > start:
                    region = gray[start:end].copy()
                    regions.append(region)

            # Process regions in parallel
            with ThreadPoolExecutor(max_workers=num_regions) as executor:
                futures = [executor.submit(process_region, region) for region in regions]
                all_lines = []
                for future in as_completed(futures):
                    result = future.result()
                    if isinstance(result, np.ndarray) and result.size > 0:
                        all_lines.extend(result)

            if not all_lines:
                print("No lines detected in any region")
                return [], []

            lines = np.array(all_lines)
            if len(lines.shape) == 1:
                lines = lines.reshape(1, -1)

            # Classify and merge lines
            horizontal, vertical = self._classify_lines(lines)

            # Print debug info
            print(f"Detected {len(horizontal)} horizontal and {len(vertical)} vertical lines")

            # Ensure minimum number of lines
            if len(horizontal) < 2 or len(vertical) < 2:
                print(f"Insufficient lines detected: {len(horizontal)} horizontal, {len(vertical)} vertical")
                return [], []

            return self._merge_lines(horizontal, vertical)

        except Exception as e:
            print(f"Error in parallel line detection: {str(e)}")
            return [], []

    def _find_homography_parallel(self, horizontal_lines, vertical_lines, scale_factor=1.0):
        """
        Parallel homography computation
        """
        def process_combination(args):
            h_pair, v_pair, config = args
            try:
                h1, h2 = h_pair
                v1, v2 = v_pair

                # Find intersections
                i1 = line_intersection((tuple(h1[:2]), tuple(h1[2:])),
                                     (tuple(v1[0:2]), tuple(v1[2:])))
                i2 = line_intersection((tuple(h1[:2]), tuple(h1[2:])),
                                     (tuple(v2[0:2]), tuple(v2[2:])))
                i3 = line_intersection((tuple(h2[:2]), tuple(h2[2:])),
                                     (tuple(v1[0:2]), tuple(v1[2:])))
                i4 = line_intersection((tuple(h2[:2]), tuple(h2[2:])),
                                     (tuple(v2[0:2]), tuple(v2[2:])))

                intersections = [i1, i2, i3, i4]
                if any(x is None for x in intersections):
                    return None

                intersections = sort_intersection_points(intersections)

                # Early stopping if points are too close
                if any(np.linalg.norm(np.array(p1) - np.array(p2)) < 10
                      for p1, p2 in combinations(intersections, 2)):
                    return None

                # Find homography
                matrix, _ = cv2.findHomography(
                    np.float32(config),
                    np.float32(intersections),
                    method=cv2.RANSAC
                )
                if matrix is None:
                    return None

                inv_matrix = cv2.invert(matrix)[1]
                score = self._get_confi_score(matrix)

                return (matrix, inv_matrix, score)
            except:
                return None

        # Prepare combinations
        horizontal_lines = sorted(horizontal_lines,
                                key=lambda line: np.linalg.norm(line[2:] - line[:2]),
                                reverse=True)[:4]
        vertical_lines = sorted(vertical_lines,
                              key=lambda line: np.linalg.norm(line[2:] - line[:2]),
                              reverse=True)[:4]

        h_pairs = list(combinations(horizontal_lines, 2))[:50]  # Limit combinations
        v_pairs = list(combinations(vertical_lines, 2))[:50]

        # Create all combinations with configurations
        combinations_list = [(h, v, conf) for h in h_pairs
                           for v in v_pairs
                           for conf in self.court_reference.court_conf.values()]

        # Process combinations in parallel
        max_score = -np.inf
        max_mat = None
        max_inv_mat = None

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_combination, combo)
                      for combo in combinations_list]

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    matrix, inv_matrix, score = result
                    if score > max_score:
                        max_score = score
                        max_mat = matrix
                        max_inv_mat = inv_matrix

        return max_mat, max_inv_mat, max_score

    def _get_confi_score(self, matrix):
        """
        Calculate transformation score
        """
        court = cv2.warpPerspective(self.court_reference.court, matrix, self.frame.shape[1::-1])
        court[court > 0] = 1
        gray = self.gray.copy()
        gray[gray > 0] = 1
        correct = court * gray
        wrong = court - correct
        c_p = np.sum(correct)
        w_p = np.sum(wrong)
        return c_p - 0.5 * w_p

    def add_court_overlay(self, frame, homography=None, overlay_color=(255, 255, 255), frame_num=-1):
        """
        Add overlay of the court to the frame
        """
        if homography is None and len(self.court_warp_matrix) > 0 and frame_num < len(self.court_warp_matrix):
            homography = self.court_warp_matrix[frame_num]
        court = cv2.warpPerspective(self.court_reference.court, homography, frame.shape[1::-1])
        frame[court > 0, :] = overlay_color
        return frame

    def find_lines_location(self):
        """
        Finds important lines location on frame
        """
        p = np.array(self.court_reference.get_important_lines(), dtype=np.float32).reshape((-1, 1, 2))
        lines = cv2.perspectiveTransform(p, self.court_warp_matrix[-1]).reshape(-1)
        self.baseline_top = lines[:4]
        self.baseline_bottom = lines[4:8]
        self.net = lines[8:12]
        self.left_court_line = lines[12:16]
        self.right_court_line = lines[16:20]
        self.left_inner_line = lines[20:24]
        self.right_inner_line = lines[24:28]
        self.middle_line = lines[28:32]
        self.top_inner_line = lines[32:36]
        self.bottom_inner_line = lines[36:40]
        if self.verbose:
            display_lines_on_frame(self.frame.copy(), [self.baseline_top, self.baseline_bottom,
                                                       self.net, self.top_inner_line, self.bottom_inner_line],
                                   [self.left_court_line, self.right_court_line,
                                    self.right_inner_line, self.left_inner_line, self.middle_line])

    def get_extra_parts_location(self, frame_num=-1):
        parts = np.array(self.court_reference.get_extra_parts(), dtype=np.float32).reshape((-1, 1, 2))
        parts = cv2.perspectiveTransform(parts, self.court_warp_matrix[frame_num]).reshape(-1)
        top_part = parts[:2]
        bottom_part = parts[2:]
        return top_part, bottom_part

    def delete_extra_parts(self, frame, frame_num=-1):
        img = frame.copy()
        top, bottom = self.get_extra_parts_location(frame_num)
        img[int(bottom[1] - 10):int(bottom[1] + 10), int(bottom[0] - 15):int(bottom[0] + 15), :] = (0, 0, 0)
        img[int(top[1] - 10):int(top[1] + 10), int(top[0] - 15):int(top[0] + 15), :] = (0, 0, 0)
        return img

    def get_warped_court(self):
        """
        Returns warped court using the reference court and the transformation of the court
        """
        court = cv2.warpPerspective(self.court_reference.court, self.court_warp_matrix[-1], self.frame.shape[1::-1])
        court[court > 0] = 1
        return court

    def _get_court_accuracy(self, verbose=0):
        """
        Calculate court accuracy after detection
        """
        frame = self.frame.copy()
        gray = self._threshold(frame)
        gray[gray > 0] = 1
        gray = cv2.dilate(gray, np.ones((9, 9), dtype=np.uint8))
        court = self.get_warped_court()
        total_white_pixels = sum(sum(court))
        sub = court.copy()
        sub[gray == 1] = 0
        accuracy = 100 - (sum(sum(sub)) / total_white_pixels) * 100
        if verbose:
            plt.figure()
            plt.subplot(1, 3, 1)
            plt.imshow(gray, cmap='gray')
            plt.title('Grayscale frame'), plt.xticks([]), plt.yticks([])
            plt.subplot(1, 3, 2)
            plt.imshow(court, cmap='gray')
            plt.title('Projected court'), plt.xticks([]), plt.yticks([])
            plt.subplot(1, 3, 3)
            plt.imshow(sub, cmap='gray')
            plt.title('Subtraction result'), plt.xticks([]), plt.yticks([])
            plt.show()
        return accuracy

    def track_court(self, frame):
        """
        Track court location after detection
        """
        copy = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.frame_points is None:
            conf_points = np.array(self.court_reference.court_conf[self.best_conf], dtype=np.float32).reshape((-1, 1, 2))
            self.frame_points = cv2.perspectiveTransform(conf_points,
                                                         self.court_warp_matrix[-1]).squeeze().round()
        # Lines of configuration on frames
        line1 = self.frame_points[:2]
        line2 = self.frame_points[2:4]
        line3 = self.frame_points[[0, 2]]
        line4 = self.frame_points[[1, 3]]
        lines = [line1, line2, line3, line4]
        new_lines = []
        for line in lines:
            # Get 100 samples of each line in the frame
            points_on_line = np.linspace(line[0], line[1], 102)[1:-1]  # 100 samples on the line
            p1 = None
            p2 = None
            if line[0][0] > self.v_width or line[0][0] < 0 or line[0][1] > self.v_height or line[0][1] < 0:
                for p in points_on_line:
                    if 0 < p[0] < self.v_width and 0 < p[1] < self.v_height:
                        p1 = p
                        break
            if line[1][0] > self.v_width or line[1][0] < 0 or line[1][1] > self.v_height or line[1][1] < 0:
                for p in reversed(points_on_line):
                    if 0 < p[0] < self.v_width and 0 < p[1] < self.v_height:
                        p2 = p
                        break
            # if one of the ends of the line is out of the frame get only the points inside the frame
            if p1 is not None or p2 is not None:
                print('points outside screen')
                points_on_line = np.linspace(p1 if p1 is not None else line[0], p2 if p2 is not None else line[1], 102)[
                                 1:-1]

            new_points = []
            # Find max intensity pixel near each point
            for p in points_on_line:
                p = (int(round(p[0])), int(round(p[1])))
                top_y, top_x = max(p[1] - self.dist, 0), max(p[0] - self.dist, 0)
                bottom_y, bottom_x = min(p[1] + self.dist, self.v_height), min(p[0] + self.dist, self.v_width)
                patch = gray[top_y: bottom_y, top_x: bottom_x]
                y, x = np.unravel_index(np.argmax(patch), patch.shape)
                if patch[y, x] > 150:
                    new_p = (x + top_x + 1, y + top_y + 1)
                    new_points.append(new_p)
                    cv2.circle(copy, p, 1, (255, 0, 0), 1)
                    cv2.circle(copy, new_p, 1, (0, 0, 255), 1)
            new_points = np.array(new_points, dtype=np.float32).reshape((-1, 1, 2))
            # find line fitting the new points
            [vx, vy, x, y] = cv2.fitLine(new_points, cv2.DIST_L2, 0, 0.01, 0.01)
            new_lines.append(((int(x - vx * self.v_width), int(y - vy * self.v_width)),
                              (int(x + vx * self.v_width), int(y + vy * self.v_width))))

            # if less than 50 points were found detect court from the start instead of tracking
            if len(new_points) < 50:
                if self.dist > 20:
                    cv2.imshow('court', copy)
                    if cv2.waitKey(0) & 0xff == 27:
                        cv2.destroyAllWindows()
                    self.detect(frame)
                    conf_points = np.array(self.court_reference.court_conf[self.best_conf], dtype=np.float32).reshape(
                        (-1, 1, 2))
                    self.frame_points = cv2.perspectiveTransform(conf_points,
                                                                 self.court_warp_matrix[-1]).squeeze().round()

                    print('Smaller than 50')
                    return
                else:
                    print('Court tracking failed, adding 5 pixels to dist')
                    self.dist += 5
                    self.track_court(frame)
                    return
        # Find transformation from new lines
        i1 = line_intersection(new_lines[0], new_lines[2])
        i2 = line_intersection(new_lines[0], new_lines[3])
        i3 = line_intersection(new_lines[1], new_lines[2])
        i4 = line_intersection(new_lines[1], new_lines[3])
        intersections = np.array([i1, i2, i3, i4], dtype=np.float32)
        matrix, _ = cv2.findHomography(np.float32(self.court_reference.court_conf[self.best_conf]),
                                       intersections, method=0)
        inv_matrix = cv2.invert(matrix)[1]
        self.court_warp_matrix.append(matrix)
        self.game_warp_matrix.append(inv_matrix)
        self.frame_points = intersections


def sort_intersection_points(intersections):
    """
    sort intersection points from top left to bottom right
    """
    y_sorted = sorted(intersections, key=lambda x: x[1])
    p12 = y_sorted[:2]
    p34 = y_sorted[2:]
    p12 = sorted(p12, key=lambda x: x[0])
    p34 = sorted(p34, key=lambda x: x[0])
    return p12 + p34


def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line(line1[0], line1[1])
    l2 = Line(line2[0], line2[1])

    intersection = l1.intersection(l2)
    return intersection[0].coordinates


def display_lines_on_frame(frame, horizontal=(), vertical=()):
    """
    Display lines on frame for horizontal and vertical lines
    """

    '''cv2.line(frame, (int(len(frame[0]) * 4 / 7), 0), (int(len(frame[0]) * 4 / 7), 719), (255, 255, 0), 2)
    cv2.line(frame, (int(len(frame[0]) * 3 / 7), 0), (int(len(frame[0]) * 3 / 7), 719), (255, 255, 0), 2)'''
    for line in horizontal:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

    for line in vertical:
        x1, y1, x2, y2 = line
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, (x1, y1), 1, (255, 0, 0), 2)
        cv2.circle(frame, (x2, y2), 1, (255, 0, 0), 2)

    cv2.imshow('court', frame)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    # cv2.imwrite('../report/t.png', frame)
    return frame


def display_lines_and_points_on_frame(frame, lines=(), points=(), line_color=(0, 0, 255), point_color=(255, 0, 0)):
    """
    Display all lines and points given on frame
    """

    for line in lines:
        x1, y1, x2, y2 = line
        frame = cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)
    for p in points:
        frame = cv2.circle(frame, p, 2, point_color, 2)

    cv2.imshow('court', frame)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    return frame


if __name__ == '__main__':
    filename = '../images/img1.jpg'
    img = cv2.imread(filename)
    import time

    s = time.time()
    court_detector = CourtDetector()
    court_detector.detect(img, 0)
    top, bottom = court_detector.get_extra_parts_location()
    cv2.circle(img, tuple(top), 3, (0,255,0), 1)
    cv2.circle(img, tuple(bottom), 3, (0,255,0), 1)
    img[int(bottom[1]-10):int(bottom[1]+10), int(bottom[0] - 10):int(bottom[0]+10), :] = (0,0,0)
    img[int(top[1]-10):int(top[1]+10), int(top[0] - 10):int(top[0]+10), :] = (0,0,0)
    cv2.imshow('df', img)
    if cv2.waitKey(0):
        cv2.destroyAllWindows()
    print(f'time = {time.time() - s}')
