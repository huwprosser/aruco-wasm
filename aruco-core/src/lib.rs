use core::cmp::Ordering;

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::{
    u16x8_add, u16x8_extract_lane, u16x8_extmul_low_u8x16, u16x8_shr, u8x16_shuffle,
    u8x16_splat, v128, v128_load,
};

#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "env")]
extern "C" {
    fn wasm_now() -> f64;
}

#[derive(Clone, Copy, Default)]
struct Point {
    x: f64,
    y: f64,
}

type Quad = [Point; 4];

#[derive(Clone, Copy)]
struct AxisProjection {
    origin: Point,
    x: Point,
    y: Point,
    z: Point,
}

#[derive(Clone, Copy)]
struct MarkerPose {
    quaternion: [f64; 4],
    translation: [f64; 3],
}

#[derive(Clone, Copy)]
struct MarkerPoseSolution {
    axis: AxisProjection,
    pose: MarkerPose,
}

#[derive(Clone)]
struct DetectedMarker {
    confidence: f64,
    id: usize,
    pose: MarkerPose,
    corners: Quad,
    axis: AxisProjection,
}

#[derive(Clone)]
struct MarkerCandidate {
    confidence: f64,
    id: usize,
    corners: Quad,
    perimeter: f64,
}

#[derive(Clone)]
struct Candidate {
    corners: Quad,
    contour: Vec<Point>,
    perimeter: f64,
}

pub struct Detector {
    binary: Vec<u8>,
    camera_focal_length_scale: f64,
    camera_focal_x: f64,
    camera_focal_y: f64,
    camera_principal_x: f64,
    camera_principal_y: f64,
    current_max_dimension: usize,
    fast_detection_streak: u32,
    gray: Vec<u8>,
    height: usize,
    integral: Vec<u32>,
    result: Vec<f64>,
    rgba: Vec<u8>,
    slow_detection_streak: u32,
    preferred_threshold_window_index: usize,
    threshold_x_left: Vec<usize>,
    threshold_x_right: Vec<usize>,
    threshold_x_width: Vec<u32>,
    threshold_y_bottom: Vec<usize>,
    threshold_y_height: Vec<u32>,
    threshold_y_top: Vec<usize>,
    visited: Vec<u8>,
    width: usize,
}

const MARKER_SIZE: usize = 4;
const MARKER_BORDER_BITS: usize = 1;
const GRID_SIZE: usize = MARKER_SIZE + MARKER_BORDER_BITS * 2;
const HOMOGRAPHY_EPSILON: f64 = 1e-6;
const ADAPTIVE_THRESH_WINDOW_SIZES: [usize; 3] = [3, 13, 23];
const CANONICAL_CELL_SIZE: usize = 4;
const CANONICAL_SIZE: usize = GRID_SIZE * CANONICAL_CELL_SIZE;
const CANONICAL_IMAGE_PIXELS: usize = CANONICAL_SIZE * CANONICAL_SIZE;
const CELL_MARGIN_PIXELS: usize = (CANONICAL_CELL_SIZE as f64 * 0.13) as usize;
const MAX_MARKER_PERIMETER_RATE: f64 = 4.0;
const MAX_ERRONEOUS_BORDER_RATE: f64 = 0.18;
const MIN_MARKER_CONFIDENCE: f64 = 0.7;
const MIN_CONTOUR_POINTS: usize = 24;
const MIN_DISTANCE_TO_BORDER: f64 = 3.0;
const MIN_MARKER_PERIMETER_RATE: f64 = 0.03;
const MIN_OTSU_STD_DEV: f64 = 5.0;
const QUICK_DECODE_HAMMING_SLACK: usize = 1;
const QUICK_MAX_BORDER_ERRORS: usize = 3;
const MIN_CORNER_DISTANCE_RATE: f64 = 0.05;
const MIN_MARKER_DISTANCE_RATE: f64 = 0.125;
const RESULT_HEADER_LEN: usize = 1;
const RESULT_MARKER_LEN: usize = 25;
const MAX_RESULT_MARKERS: usize = 12;
const VALID_BIT_ID_THRESHOLD: f64 = 0.49;
const AXIS_LENGTH: f64 = 0.42;
const MAX_DIMENSION_CAP: usize = 320;
const MAX_HAMMING_DISTANCE: usize = 1;
const MIN_CANDIDATE_AREA: f64 = 110.0;
const MIN_DIMENSION: usize = 192;
const THRESHOLD_BIAS: u32 = 7;
const DEFAULT_FOCAL_LENGTH_RATE: f64 = 0.92;
const DIRECTION_X: [i32; 8] = [-1, -1, 0, 1, 1, 1, 0, -1];
const DIRECTION_Y: [i32; 8] = [0, -1, -1, -1, 0, 1, 1, 1];

const DICT_4X4_50: [[[u8; 2]; 4]; 50] = [
    [[181, 50], [235, 72], [76, 173], [18, 215]],
    [[15, 154], [101, 71], [89, 240], [226, 166]],
    [[51, 45], [222, 17], [180, 204], [136, 123]],
    [[153, 70], [193, 60], [98, 153], [60, 131]],
    [[84, 158], [161, 211], [121, 42], [203, 133]],
    [[121, 205], [216, 183], [179, 158], [237, 27]],
    [[158, 46], [135, 93], [116, 121], [186, 225]],
    [[196, 242], [35, 234], [79, 35], [87, 196]],
    [[254, 218], [173, 239], [91, 127], [247, 181]],
    [[207, 86], [101, 252], [106, 243], [63, 166]],
    [[249, 145], [248, 142], [137, 159], [113, 31]],
    [[17, 167], [211, 18], [229, 136], [72, 203]],
    [[14, 183], [55, 86], [237, 112], [106, 236]],
    [[42, 15], [29, 21], [240, 84], [168, 184]],
    [[36, 177], [58, 66], [141, 36], [66, 92]],
    [[38, 62], [47, 81], [124, 100], [138, 244]],
    [[70, 101], [22, 240], [166, 98], [15, 104]],
    [[102, 0], [12, 192], [0, 102], [3, 48]],
    [[108, 94], [41, 245], [122, 54], [175, 148]],
    [[118, 175], [159, 211], [245, 110], [203, 249]],
    [[134, 139], [21, 75], [209, 97], [210, 168]],
    [[176, 43], [155, 9], [212, 13], [144, 217]],
    [[204, 213], [48, 254], [171, 51], [127, 12]],
    [[221, 130], [193, 206], [65, 187], [115, 131]],
    [[254, 71], [157, 252], [226, 127], [63, 185]],
    [[148, 113], [178, 104], [142, 41], [22, 77]],
    [[172, 228], [10, 126], [39, 53], [126, 80]],
    [[165, 84], [104, 120], [42, 165], [30, 22]],
    [[33, 35], [91, 0], [196, 132], [0, 218]],
    [[52, 111], [155, 113], [246, 44], [142, 217]],
    [[68, 21], [48, 208], [168, 34], [11, 12]],
    [[87, 178], [231, 194], [77, 234], [67, 231]],
    [[158, 207], [149, 127], [243, 121], [254, 169]],
    [[240, 203], [153, 171], [211, 15], [213, 153]],
    [[8, 174], [3, 23], [117, 16], [232, 192]],
    [[9, 41], [82, 5], [148, 144], [160, 74]],
    [[24, 117], [178, 52], [174, 24], [44, 77]],
    [[4, 255], [51, 115], [255, 32], [206, 204]],
    [[13, 246], [99, 118], [111, 176], [110, 198]],
    [[28, 90], [161, 101], [90, 56], [166, 133]],
    [[23, 24], [228, 65], [24, 232], [130, 39]],
    [[42, 40], [14, 5], [20, 84], [160, 112]],
    [[50, 140], [140, 19], [49, 76], [200, 49]],
    [[56, 178], [171, 6], [77, 28], [96, 213]],
    [[36, 232], [10, 99], [23, 36], [198, 80]],
    [[46, 235], [31, 103], [215, 116], [230, 248]],
    [[45, 63], [123, 85], [252, 180], [170, 222]],
    [[75, 100], [70, 180], [38, 210], [45, 98]],
    [[80, 46], [131, 145], [116, 10], [137, 193]],
    [[80, 19], [177, 128], [200, 10], [1, 141]],
];

impl Detector {
    fn new() -> Self {
        Self {
            binary: Vec::new(),
            camera_focal_length_scale: 1.0,
            camera_focal_x: 0.0,
            camera_focal_y: 0.0,
            camera_principal_x: 0.0,
            camera_principal_y: 0.0,
            current_max_dimension: MAX_DIMENSION_CAP,
            fast_detection_streak: 0,
            gray: Vec::new(),
            height: 0,
            integral: Vec::new(),
            result: Vec::with_capacity(
                RESULT_HEADER_LEN + RESULT_MARKER_LEN * MAX_RESULT_MARKERS,
            ),
            rgba: Vec::new(),
            slow_detection_streak: 0,
            preferred_threshold_window_index: ADAPTIVE_THRESH_WINDOW_SIZES.len() / 2,
            threshold_x_left: Vec::new(),
            threshold_x_right: Vec::new(),
            threshold_x_width: Vec::new(),
            threshold_y_bottom: Vec::new(),
            threshold_y_height: Vec::new(),
            threshold_y_top: Vec::new(),
            visited: Vec::new(),
            width: 0,
        }
    }

    fn set_camera_intrinsics(
        &mut self,
        focal_length_x: f64,
        focal_length_y: f64,
        principal_x: f64,
        principal_y: f64,
        focal_length_scale: f64,
    ) {
        self.camera_focal_x = focal_length_x.max(0.0);
        self.camera_focal_y = focal_length_y.max(0.0);
        self.camera_principal_x = principal_x.max(0.0);
        self.camera_principal_y = principal_y.max(0.0);
        self.camera_focal_length_scale = focal_length_scale.max(HOMOGRAPHY_EPSILON);
    }

    fn configure_frame(&mut self, source_width: usize, source_height: usize) -> u32 {
        self.ensure_capacity(source_width, source_height);
        pack_dimensions(self.width, self.height)
    }

    fn set_input_size(&mut self, input_width: usize, input_height: usize) -> u32 {
        if input_width == 0 || input_height == 0 {
            self.width = 0;
            self.height = 0;
            return 0;
        }

        if self.width != input_width || self.height != input_height {
            self.resize_buffers(input_width, input_height);
        }

        pack_dimensions(self.width, self.height)
    }

    fn prepare_rgba(&mut self, len: usize) -> *mut u8 {
        if self.rgba.len() != len {
            self.rgba.resize(len, 0);
        }
        self.rgba.as_mut_ptr()
    }

    fn detect(
        &mut self,
        source_width: usize,
        source_height: usize,
    ) {
        if source_width == 0 || source_height == 0 || self.width == 0 || self.height == 0 {
            self.write_empty_result();
            return;
        }

        if self.rgba.len() < self.width * self.height * 4 {
            self.write_empty_result();
            return;
        }

        let started_at = now();
        self.build_gray_integral();
        let scale_x = source_width as f64 / self.width as f64;
        let scale_y = source_height as f64 / self.height as f64;
        let markers =
            self.detect_markers(scale_x, scale_y, source_width as f64, source_height as f64);
        self.tune_max_dimension(now() - started_at);
        self.write_detection_result(&markers);
    }

    fn write_empty_result(&mut self) {
        self.result.clear();
        self.result.push(0.0);
    }

    fn write_detection_result(&mut self, markers: &[DetectedMarker]) {
        self.result.clear();
        self.result.push(markers.len() as f64);

        for marker in markers.iter().take(MAX_RESULT_MARKERS) {
            self.result.push(marker.id as f64);
            self.result.push(marker.confidence);
            for corner in marker.corners {
                self.result.push(corner.x);
                self.result.push(corner.y);
            }
            self.result.push(marker.axis.origin.x);
            self.result.push(marker.axis.origin.y);
            self.result.push(marker.axis.x.x);
            self.result.push(marker.axis.x.y);
            self.result.push(marker.axis.y.x);
            self.result.push(marker.axis.y.y);
            self.result.push(marker.axis.z.x);
            self.result.push(marker.axis.z.y);
            self.result.push(marker.pose.translation[0]);
            self.result.push(marker.pose.translation[1]);
            self.result.push(marker.pose.translation[2]);
            self.result.push(marker.pose.quaternion[0]);
            self.result.push(marker.pose.quaternion[1]);
            self.result.push(marker.pose.quaternion[2]);
            self.result.push(marker.pose.quaternion[3]);
        }
    }

    fn build_gray_integral(&mut self) {
        #[cfg(target_arch = "wasm32")]
        unsafe {
            self.build_gray_integral_simd();
            return;
        }

        #[cfg(not(target_arch = "wasm32"))]
        self.build_gray_integral_scalar();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn build_gray_integral_scalar(&mut self) {
        let integral_stride = self.width + 1;
        self.integral[..integral_stride].fill(0);

        for y in 0..self.height {
            let mut row_sum = 0u32;
            let integral_row = (y + 1) * integral_stride;
            let previous_integral_row = y * integral_stride;
            let row_offset = y * self.width;
            self.integral[integral_row] = 0;

            for x in 0..self.width {
                let rgba_offset = (row_offset + x) * 4;
                let gray_value = ((self.rgba[rgba_offset] as u32 * 77
                    + self.rgba[rgba_offset + 1] as u32 * 150
                    + self.rgba[rgba_offset + 2] as u32 * 29)
                    >> 8) as u8;
                row_sum += gray_value as u32;
                self.gray[row_offset + x] = gray_value;
                self.integral[integral_row + x + 1] =
                    self.integral[previous_integral_row + x + 1] + row_sum;
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    #[target_feature(enable = "simd128")]
    unsafe fn build_gray_integral_simd(&mut self) {
        let integral_stride = self.width + 1;
        self.integral[..integral_stride].fill(0);
        let weight_r = u8x16_splat(77);
        let weight_g = u8x16_splat(150);
        let weight_b = u8x16_splat(29);

        for y in 0..self.height {
            let mut row_sum = 0u32;
            let integral_row = (y + 1) * integral_stride;
            let previous_integral_row = y * integral_stride;
            let row_offset = y * self.width;
            self.integral[integral_row] = 0;

            let mut x = 0usize;
            while x + 8 <= self.width {
                let rgba_offset = (row_offset + x) * 4;
                let rgba_ptr = self.rgba.as_ptr().add(rgba_offset) as *const v128;
                let pixels_a = v128_load(rgba_ptr);
                let pixels_b = v128_load(rgba_ptr.add(1));

                let r = u8x16_shuffle::<0, 4, 8, 12, 16, 20, 24, 28, 0, 4, 8, 12, 16, 20, 24, 28>(
                    pixels_a, pixels_b,
                );
                let g = u8x16_shuffle::<1, 5, 9, 13, 17, 21, 25, 29, 1, 5, 9, 13, 17, 21, 25, 29>(
                    pixels_a, pixels_b,
                );
                let b = u8x16_shuffle::<2, 6, 10, 14, 18, 22, 26, 30, 2, 6, 10, 14, 18, 22, 26, 30>(
                    pixels_a, pixels_b,
                );

                let weighted_sum = u16x8_add(
                    u16x8_add(
                        u16x8_extmul_low_u8x16(r, weight_r),
                        u16x8_extmul_low_u8x16(g, weight_g),
                    ),
                    u16x8_extmul_low_u8x16(b, weight_b),
                );
                let gray_values = u16x8_shr(weighted_sum, 8);

                macro_rules! write_lane {
                    ($lane:literal) => {{
                        let gray_value = u16x8_extract_lane::<$lane>(gray_values) as u8;
                        let pixel_index = row_offset + x + $lane;
                        row_sum += gray_value as u32;
                        self.gray[pixel_index] = gray_value;
                        self.integral[integral_row + x + $lane + 1] =
                            self.integral[previous_integral_row + x + $lane + 1] + row_sum;
                    }};
                }

                write_lane!(0);
                write_lane!(1);
                write_lane!(2);
                write_lane!(3);
                write_lane!(4);
                write_lane!(5);
                write_lane!(6);
                write_lane!(7);

                x += 8;
            }

            while x < self.width {
                let rgba_offset = (row_offset + x) * 4;
                let gray_value = ((self.rgba[rgba_offset] as u32 * 77
                    + self.rgba[rgba_offset + 1] as u32 * 150
                    + self.rgba[rgba_offset + 2] as u32 * 29)
                    >> 8) as u8;
                row_sum += gray_value as u32;
                self.gray[row_offset + x] = gray_value;
                self.integral[integral_row + x + 1] =
                    self.integral[previous_integral_row + x + 1] + row_sum;
                x += 1;
            }
        }
    }

    fn adaptive_threshold(&mut self, window_size: usize) {
        let radius = (window_size.max(3) | 1) / 2;
        let integral_stride = self.width + 1;
        self.prepare_threshold_bounds(radius, integral_stride);

        for y in 0..self.height {
            let top_row = self.threshold_y_top[y];
            let bottom_row = self.threshold_y_bottom[y];
            let area_height = self.threshold_y_height[y];
            let row_offset = y * self.width;

            for x in 0..self.width {
                let left = self.threshold_x_left[x];
                let right = self.threshold_x_right[x];
                let area = area_height * self.threshold_x_width[x];
                let sum = self.integral[bottom_row + right]
                    - self.integral[top_row + right]
                    - self.integral[bottom_row + left]
                    + self.integral[top_row + left];
                let offset = row_offset + x;
                let threshold = sum.saturating_sub(THRESHOLD_BIAS.saturating_mul(area));
                self.binary[offset] = if (self.gray[offset] as u32).saturating_mul(area) < threshold
                {
                    1
                } else {
                    0
                };
            }
        }
    }

    fn prepare_threshold_bounds(&mut self, radius: usize, integral_stride: usize) {
        for x in 0..self.width {
            let left = x.saturating_sub(radius);
            let right = (x + radius).min(self.width - 1) + 1;
            self.threshold_x_left[x] = left;
            self.threshold_x_right[x] = right;
            self.threshold_x_width[x] = (right - left) as u32;
        }

        for y in 0..self.height {
            let top = y.saturating_sub(radius);
            let bottom = (y + radius).min(self.height - 1) + 1;
            self.threshold_y_top[y] = top * integral_stride;
            self.threshold_y_bottom[y] = bottom * integral_stride;
            self.threshold_y_height[y] = (bottom - top) as u32;
        }
    }

    fn detect_markers(
        &mut self,
        scale_x: f64,
        scale_y: f64,
        source_width: f64,
        source_height: f64,
    ) -> Vec<DetectedMarker> {
        let image_max_dimension = self.width.max(self.height) as f64;
        let min_perimeter = image_max_dimension * MIN_MARKER_PERIMETER_RATE;
        let max_perimeter = image_max_dimension * MAX_MARKER_PERIMETER_RATE;
        let min_area = MIN_CANDIDATE_AREA.max(self.width as f64 * self.height as f64 * 0.0006);
        let min_component_span = (min_area.sqrt() * 0.5).round().max(4.0) as usize;

        for window_index in threshold_window_order(self.preferred_threshold_window_index) {
            let window_size = ADAPTIVE_THRESH_WINDOW_SIZES[window_index];
            self.visited.fill(0);
            self.adaptive_threshold(window_size);
            let candidates = self.collect_candidates(
                min_perimeter,
                max_perimeter,
                min_area,
                min_component_span,
            );
            if candidates.is_empty() {
                continue;
            }

            let markers = self.resolve_detected_markers(
                candidates,
                scale_x,
                scale_y,
                source_width,
                source_height,
            );
            if !markers.is_empty() {
                self.preferred_threshold_window_index = window_index;
                return markers;
            }
        }

        Vec::new()
    }

    fn resolve_detected_markers(
        &self,
        candidates: Vec<Candidate>,
        scale_x: f64,
        scale_y: f64,
        source_width: f64,
        source_height: f64,
    ) -> Vec<DetectedMarker> {
        let selected_candidates =
            filter_candidate_groups(candidates, self.width as f64, self.height as f64);
        let mut marker_candidates = Vec::with_capacity(selected_candidates.len());

        for candidate in selected_candidates {
            let refined_corners = refine_candidate_lines(&candidate.contour, candidate.corners)
                .unwrap_or(candidate.corners);
            let Some((id, confidence, corners)) = self.decode_candidate(refined_corners) else {
                continue;
            };

            let corners = scale_quad(corners, scale_x, scale_y);
            let perimeter = quad_perimeter(corners);
            marker_candidates.push(MarkerCandidate {
                confidence,
                id,
                corners,
                perimeter,
            });
        }

        let deduped_candidates = dedupe_marker_candidates(marker_candidates);
        let mut markers = Vec::with_capacity(deduped_candidates.len());

        for candidate in deduped_candidates {
            let Some(solution) =
                self.solve_marker_pose(candidate.corners, source_width, source_height)
            else {
                continue;
            };

            markers.push(DetectedMarker {
                confidence: candidate.confidence,
                id: candidate.id,
                corners: candidate.corners,
                pose: solution.pose,
                axis: solution.axis,
            });
        }

        markers
    }

    fn collect_candidates(
        &mut self,
        min_perimeter: f64,
        max_perimeter: f64,
        min_area: f64,
        min_component_span: usize,
    ) -> Vec<Candidate> {
        let mut candidates = Vec::new();
        let scan_width = self.width.saturating_sub(1);

        for y in 1..self.height.saturating_sub(1) {
            let row_offset = y * self.width;
            let mut x = 1usize;
            while x < scan_width {
                let offset = row_offset + x;
                if self.binary[offset] == 0 {
                    x += 1;
                    while x < scan_width && self.binary[row_offset + x] == 0 {
                        x += 1;
                    }
                    continue;
                }

                if self.binary[offset - 1] != 0
                    || self.visited[offset] != 0
                    || self.vertical_black_span_below(x, y, min_component_span)
                {
                    x += 1;
                    continue;
                }

                let Some(candidate) = self.trace_candidate(
                    x as i32,
                    y as i32,
                    min_perimeter,
                    max_perimeter,
                    min_area,
                ) else {
                    x += 1;
                    continue;
                };

                if !self.passes_quick_binary_marker_check(candidate.corners) {
                    x += 1;
                    continue;
                }

                candidates.push(candidate);
                x += 1;
            }
        }

        candidates
    }

    fn trace_candidate(
        &mut self,
        start_x: i32,
        start_y: i32,
        min_perimeter: f64,
        max_perimeter: f64,
        min_area: f64,
    ) -> Option<Candidate> {
        let max_steps = self.width * self.height;
        let start_back_x = start_x - 1;
        let start_back_y = start_y;

        let mut current_x = start_x;
        let mut current_y = start_y;
        let mut back_x = start_back_x;
        let mut back_y = start_back_y;
        let mut steps = 0usize;
        let mut point_count = 0usize;
        let mut contour = Vec::with_capacity(64);

        let mut top_left = Point::default();
        let mut top_right = Point::default();
        let mut bottom_right = Point::default();
        let mut bottom_left = Point::default();
        let mut min_sum = f64::INFINITY;
        let mut max_sum = f64::NEG_INFINITY;
        let mut min_diff = f64::INFINITY;
        let mut max_diff = f64::NEG_INFINITY;

        loop {
            let point = Point {
                x: current_x as f64,
                y: current_y as f64,
            };

            point_count += 1;
            contour.push(point);

            let sum = point.x + point.y;
            let diff = point.x - point.y;
            if sum < min_sum {
                min_sum = sum;
                top_left = point;
            }
            if sum > max_sum {
                max_sum = sum;
                bottom_right = point;
            }
            if diff > max_diff {
                max_diff = diff;
                top_right = point;
            }
            if diff < min_diff {
                min_diff = diff;
                bottom_left = point;
            }

            self.visited[current_y as usize * self.width + current_x as usize] = 1;

            let back_direction = direction_index(back_x - current_x, back_y - current_y);
            if back_direction < 0 {
                break;
            }

            let mut found_next = false;
            let mut next_back_x = back_x;
            let mut next_back_y = back_y;
            let mut next_x = current_x;
            let mut next_y = current_y;

            for offset in 1..=8 {
                let direction = ((back_direction + offset) & 7) as usize;
                let candidate_x = current_x + DIRECTION_X[direction];
                let candidate_y = current_y + DIRECTION_Y[direction];

                if candidate_x < 0
                    || candidate_x >= self.width as i32
                    || candidate_y < 0
                    || candidate_y >= self.height as i32
                {
                    continue;
                }

                if self.binary[candidate_y as usize * self.width + candidate_x as usize] == 1 {
                    next_x = candidate_x;
                    next_y = candidate_y;
                    next_back_x = current_x + DIRECTION_X[(direction + 7) & 7];
                    next_back_y = current_y + DIRECTION_Y[(direction + 7) & 7];
                    found_next = true;
                    break;
                }
            }

            if !found_next {
                break;
            }

            current_x = next_x;
            current_y = next_y;
            back_x = next_back_x;
            back_y = next_back_y;
            steps += 1;

            if steps >= max_steps
                || (current_x == start_x
                    && current_y == start_y
                    && back_x == start_back_x
                    && back_y == start_back_y)
            {
                break;
            }
        }

        if point_count < MIN_CONTOUR_POINTS {
            return None;
        }

        let contour_length = point_count as f64;
        if contour_length < min_perimeter || contour_length > max_perimeter {
            return None;
        }

        let mut corners = [top_left, top_right, bottom_right, bottom_left];
        reorder_corners_clockwise(&mut corners);
        if !corners_are_unique(corners) || !is_convex_quad(corners) {
            return None;
        }

        let contour_area = polygon_area_from_quad(corners);
        if contour_area < min_area {
            return None;
        }

        let min_corner_distance = contour_length * MIN_CORNER_DISTANCE_RATE;
        if has_tight_corners(corners, min_corner_distance) {
            return None;
        }

        Some(Candidate {
            corners,
            contour,
            perimeter: quad_perimeter(corners),
        })
    }

    fn vertical_black_span_below(&self, x: usize, y: usize, minimum_span: usize) -> bool {
        let mut span = 1usize;

        let mut cursor = y;
        while cursor > 0 {
            cursor -= 1;
            if self.binary[cursor * self.width + x] == 0 {
                break;
            }
            span += 1;
            if span >= minimum_span {
                return false;
            }
        }

        cursor = y;
        while cursor + 1 < self.height {
            cursor += 1;
            if self.binary[cursor * self.width + x] == 0 {
                break;
            }
            span += 1;
            if span >= minimum_span {
                return false;
            }
        }

        true
    }

    fn solve_marker_pose(
        &self,
        corners: Quad,
        source_width: f64,
        source_height: f64,
    ) -> Option<MarkerPoseSolution> {
        let homography = create_square_homography(corners)?;
        let principal_x = if self.camera_principal_x > 0.0 {
            self.camera_principal_x
        } else {
            source_width * 0.5
        };
        let principal_y = if self.camera_principal_y > 0.0 {
            self.camera_principal_y
        } else {
            source_height * 0.5
        };
        let fallback_focal =
            source_width.max(source_height) * DEFAULT_FOCAL_LENGTH_RATE * self.camera_focal_length_scale;
        let focal_x = if self.camera_focal_x > 0.0 {
            self.camera_focal_x
        } else {
            fallback_focal
        };
        let focal_y = if self.camera_focal_y > 0.0 {
            self.camera_focal_y
        } else {
            fallback_focal
        };
        let column1 = [
            (homography[0] - principal_x * homography[6]) / focal_x,
            (homography[3] - principal_y * homography[6]) / focal_y,
            homography[6],
        ];
        let column2 = [
            (homography[1] - principal_x * homography[7]) / focal_x,
            (homography[4] - principal_y * homography[7]) / focal_y,
            homography[7],
        ];
        let column3 = [
            (homography[2] - principal_x) / focal_x,
            (homography[5] - principal_y) / focal_y,
            1.0,
        ];

        let norm1 = vector_norm(column1);
        let norm2 = vector_norm(column2);
        if norm1 < HOMOGRAPHY_EPSILON || norm2 < HOMOGRAPHY_EPSILON {
            return None;
        }

        let mut scale = 2.0 / (norm1 + norm2);
        if column3[2] < 0.0 {
            scale *= -1.0;
        }

        let scaled_column2 = scale_vector(column2, scale);
        let r1 = normalize_vector(scale_vector(column1, scale));
        let mut r2 = subtract_vectors(
            scaled_column2,
            scale_vector(r1, dot_product(r1, scaled_column2)),
        );
        r2 = normalize_vector(r2);

        if vector_norm(r1) < HOMOGRAPHY_EPSILON || vector_norm(r2) < HOMOGRAPHY_EPSILON {
            return None;
        }

        let r3 = cross_product(r1, r2);
        let translation = scale_vector(column3, scale);
        if translation[2] <= 0.0 {
            return None;
        }

        let origin = project_pose_point(
            [0.5, 0.5, 0.0],
            r1,
            r2,
            r3,
            translation,
            focal_x,
            focal_y,
            principal_x,
            principal_y,
        )?;
        let x_axis = project_pose_point(
            [0.5 + AXIS_LENGTH, 0.5, 0.0],
            r1,
            r2,
            r3,
            translation,
            focal_x,
            focal_y,
            principal_x,
            principal_y,
        )?;
        let y_axis = project_pose_point(
            [0.5, 0.5 + AXIS_LENGTH, 0.0],
            r1,
            r2,
            r3,
            translation,
            focal_x,
            focal_y,
            principal_x,
            principal_y,
        )?;
        let z_axis = project_pose_point(
            [0.5, 0.5, -AXIS_LENGTH * 1.25],
            r1,
            r2,
            r3,
            translation,
            focal_x,
            focal_y,
            principal_x,
            principal_y,
        )?;

        Some(MarkerPoseSolution {
            axis: AxisProjection {
                origin,
                x: x_axis,
                y: y_axis,
                z: z_axis,
            },
            pose: MarkerPose {
                quaternion: quaternion_from_basis(r1, r2, r3),
                translation,
            },
        })
    }

    fn decode_candidate(&self, corners: Quad) -> Option<(usize, f64, Quad)> {
        let white_ratios = self.extract_candidate_white_ratios(corners)?;
        let border_errors = border_errors(&white_ratios);
        let maximum_errors_in_border =
            ((MARKER_SIZE * MARKER_SIZE) as f64 * MAX_ERRONEOUS_BORDER_RATE) as usize;
        if border_errors > maximum_errors_in_border {
            return None;
        }

        let [byte0, byte1] = pack_marker_bits(&marker_bits_from_white_ratios(&white_ratios));
        let mut best_distance = usize::MAX;
        let mut best_id = usize::MAX;
        let mut best_rotation = 0usize;

        for (id, marker) in DICT_4X4_50.iter().enumerate() {
            for (rotation, dictionary_bytes) in marker.iter().enumerate() {
                let distance = ((byte0 ^ dictionary_bytes[0]).count_ones()
                    + (byte1 ^ dictionary_bytes[1]).count_ones())
                    as usize;

                if distance < best_distance {
                    best_distance = distance;
                    best_id = id;
                    best_rotation = rotation;
                }
            }
        }

        if best_distance > MAX_HAMMING_DISTANCE || best_id == usize::MAX {
            return None;
        }

        let confidence =
            marker_confidence_from_white_ratios(&white_ratios, DICT_4X4_50[best_id][best_rotation]);
        if confidence < MIN_MARKER_CONFIDENCE {
            return None;
        }

        Some((
            best_id,
            confidence,
            rotate_quad(corners, (4 - best_rotation) % 4),
        ))
    }

    fn extract_candidate_white_ratios(
        &self,
        corners: Quad,
    ) -> Option<[f32; GRID_SIZE * GRID_SIZE]> {
        let homography = create_square_homography(corners)?;
        let mut canonical = [0u8; CANONICAL_IMAGE_PIXELS];

        for row in 0..CANONICAL_SIZE {
            for column in 0..CANONICAL_SIZE {
                let projected = project_homography(
                    &homography,
                    (column as f64 + 0.5) / CANONICAL_SIZE as f64,
                    (row as f64 + 0.5) / CANONICAL_SIZE as f64,
                )?;
                let x = projected.x.round() as isize;
                let y = projected.y.round() as isize;

                if x < 0 || x >= self.width as isize || y < 0 || y >= self.height as isize {
                    return None;
                }

                canonical[row * CANONICAL_SIZE + column] =
                    self.gray[y as usize * self.width + x as usize];
            }
        }

        let (mean, std_dev) = canonical_mean_stddev(&canonical);
        let mut white_ratios = [0.0f32; GRID_SIZE * GRID_SIZE];
        if std_dev < MIN_OTSU_STD_DEV {
            white_ratios.fill(if mean > 127.0 { 1.0 } else { 0.0 });
            return Some(white_ratios);
        }

        let threshold = otsu_threshold_bytes(&canonical);
        for row in 0..GRID_SIZE {
            for column in 0..GRID_SIZE {
                let start_x = column * CANONICAL_CELL_SIZE + CELL_MARGIN_PIXELS;
                let start_y = row * CANONICAL_CELL_SIZE + CELL_MARGIN_PIXELS;
                let end_x = (column + 1) * CANONICAL_CELL_SIZE - CELL_MARGIN_PIXELS;
                let end_y = (row + 1) * CANONICAL_CELL_SIZE - CELL_MARGIN_PIXELS;

                let mut white = 0usize;
                let mut total = 0usize;
                for y in start_y..end_y.max(start_y + 1) {
                    for x in start_x..end_x.max(start_x + 1) {
                        total += 1;
                        if canonical[y * CANONICAL_SIZE + x] > threshold {
                            white += 1;
                        }
                    }
                }

                white_ratios[row * GRID_SIZE + column] = white as f32 / total.max(1) as f32;
            }
        }

        Some(white_ratios)
    }

    fn passes_quick_binary_marker_check(&self, corners: Quad) -> bool {
        let Some(homography) = create_square_homography(corners) else {
            return false;
        };

        let mut border_errors = 0usize;
        let mut bits = [0u8; MARKER_SIZE * MARKER_SIZE];
        let mut bit_index = 0usize;

        for row in 0..GRID_SIZE {
            for column in 0..GRID_SIZE {
                let Some(projected) = project_homography(
                    &homography,
                    (column as f64 + 0.5) / GRID_SIZE as f64,
                    (row as f64 + 0.5) / GRID_SIZE as f64,
                ) else {
                    return false;
                };

                let x = projected.x.round() as isize;
                let y = projected.y.round() as isize;
                if x < 0 || x >= self.width as isize || y < 0 || y >= self.height as isize {
                    return false;
                }

                let is_black = self.binary[y as usize * self.width + x as usize] == 1;
                if row == 0 || column == 0 || row == GRID_SIZE - 1 || column == GRID_SIZE - 1 {
                    if !is_black {
                        border_errors += 1;
                        if border_errors > QUICK_MAX_BORDER_ERRORS {
                            return false;
                        }
                    }
                    continue;
                }

                bits[bit_index] = if is_black { 0 } else { 1 };
                bit_index += 1;
            }
        }

        let [byte0, byte1] = pack_marker_bits(&bits);
        let mut best_distance = usize::MAX;

        for marker in DICT_4X4_50.iter() {
            for dictionary_bytes in marker.iter() {
                let distance = ((byte0 ^ dictionary_bytes[0]).count_ones()
                    + (byte1 ^ dictionary_bytes[1]).count_ones())
                    as usize;
                best_distance = best_distance.min(distance);
            }
        }

        best_distance <= MAX_HAMMING_DISTANCE + QUICK_DECODE_HAMMING_SLACK
    }

    fn tune_max_dimension(&mut self, detection_ms: f64) {
        const DECREASE_THRESHOLD_MS: f64 = 18.0;
        const INCREASE_THRESHOLD_MS: f64 = 9.0;
        const DECREASE_STEP: usize = 48;
        const INCREASE_STEP: usize = 24;

        if detection_ms > DECREASE_THRESHOLD_MS && self.current_max_dimension > MIN_DIMENSION {
            self.fast_detection_streak = 0;
            self.slow_detection_streak += 1;
            if self.slow_detection_streak >= 2 {
                self.current_max_dimension = self
                    .current_max_dimension
                    .saturating_sub(DECREASE_STEP)
                    .max(MIN_DIMENSION);
                self.slow_detection_streak = 0;
            }
            return;
        }

        if detection_ms < INCREASE_THRESHOLD_MS
            && self.current_max_dimension < MAX_DIMENSION_CAP
        {
            self.slow_detection_streak = 0;
            self.fast_detection_streak += 1;
            if self.fast_detection_streak >= 6 {
                self.current_max_dimension =
                    (self.current_max_dimension + INCREASE_STEP).min(MAX_DIMENSION_CAP);
                self.fast_detection_streak = 0;
            }
            return;
        }

        self.fast_detection_streak = 0;
        self.slow_detection_streak = 0;
    }

    fn ensure_capacity(&mut self, source_width: usize, source_height: usize) {
        if source_width == 0 || source_height == 0 {
            self.width = 0;
            self.height = 0;
            return;
        }

        let scale =
            (self.current_max_dimension as f64 / source_width.max(source_height) as f64).min(1.0);
        let next_width = (source_width as f64 * scale).round().max(64.0) as usize;
        let next_height = (source_height as f64 * scale).round().max(64.0) as usize;

        if self.width == next_width && self.height == next_height {
            return;
        }

        self.resize_buffers(next_width, next_height);
    }

    fn resize_buffers(&mut self, width: usize, height: usize) {
        self.width = width;
        self.height = height;
        let pixel_count = self.width * self.height;
        self.gray.resize(pixel_count, 0);
        self.binary.resize(pixel_count, 0);
        self.visited.resize(pixel_count, 0);
        self.integral
            .resize((self.width + 1) * (self.height + 1), 0);
        self.threshold_x_left.resize(self.width, 0);
        self.threshold_x_right.resize(self.width, 0);
        self.threshold_x_width.resize(self.width, 0);
        self.threshold_y_bottom.resize(self.height, 0);
        self.threshold_y_height.resize(self.height, 0);
        self.threshold_y_top.resize(self.height, 0);
    }
}

fn now() -> f64 {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        wasm_now()
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        0.0
    }
}

fn pack_dimensions(width: usize, height: usize) -> u32 {
    (width as u32) | ((height as u32) << 16)
}

fn threshold_window_order(preferred_index: usize) -> [usize; ADAPTIVE_THRESH_WINDOW_SIZES.len()] {
    let mut order = [0usize; ADAPTIVE_THRESH_WINDOW_SIZES.len()];
    let preferred = preferred_index.min(ADAPTIVE_THRESH_WINDOW_SIZES.len() - 1);
    order[0] = preferred;

    let mut cursor = 1usize;
    for index in 0..ADAPTIVE_THRESH_WINDOW_SIZES.len() {
        if index != preferred {
            order[cursor] = index;
            cursor += 1;
        }
    }

    order
}

fn create_square_homography(corners: Quad) -> Option<[f64; 8]> {
    let [top_left, top_right, bottom_right, bottom_left] = corners;
    let dx1 = top_right.x - bottom_right.x;
    let dx2 = bottom_left.x - bottom_right.x;
    let dx3 = top_left.x - top_right.x + bottom_right.x - bottom_left.x;
    let dy1 = top_right.y - bottom_right.y;
    let dy2 = bottom_left.y - bottom_right.y;
    let dy3 = top_left.y - top_right.y + bottom_right.y - bottom_left.y;

    if dx3.abs() < HOMOGRAPHY_EPSILON && dy3.abs() < HOMOGRAPHY_EPSILON {
        return Some([
            top_right.x - top_left.x,
            bottom_left.x - top_left.x,
            top_left.x,
            top_right.y - top_left.y,
            bottom_left.y - top_left.y,
            top_left.y,
            0.0,
            0.0,
        ]);
    }

    let denominator = dx1 * dy2 - dx2 * dy1;
    if denominator.abs() < HOMOGRAPHY_EPSILON {
        return None;
    }

    let g = (dx3 * dy2 - dx2 * dy3) / denominator;
    let h = (dx1 * dy3 - dx3 * dy1) / denominator;
    Some([
        top_right.x - top_left.x + g * top_right.x,
        bottom_left.x - top_left.x + h * bottom_left.x,
        top_left.x,
        top_right.y - top_left.y + g * top_right.y,
        bottom_left.y - top_left.y + h * bottom_left.y,
        top_left.y,
        g,
        h,
    ])
}

fn cross_product(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn corners_are_unique(corners: Quad) -> bool {
    for left in 0..4 {
        for right in (left + 1)..4 {
            if corners[left].x == corners[right].x && corners[left].y == corners[right].y {
                return false;
            }
        }
    }

    true
}

fn dedupe_marker_candidates(mut markers: Vec<MarkerCandidate>) -> Vec<MarkerCandidate> {
    markers.sort_by(|left, right| {
        right
            .perimeter
            .partial_cmp(&left.perimeter)
            .unwrap_or(Ordering::Equal)
    });
    let mut deduped: Vec<MarkerCandidate> = Vec::new();

    'marker: for marker in markers {
        for candidate in &deduped {
            let average_distance = average_corner_distance(candidate.corners, marker.corners);
            if average_distance
                < candidate.perimeter.min(marker.perimeter) * MIN_MARKER_DISTANCE_RATE
            {
                continue 'marker;
            }
        }

        deduped.push(marker);
        if deduped.len() >= MAX_RESULT_MARKERS {
            break;
        }
    }

    deduped
}

fn direction_index(dx: i32, dy: i32) -> i32 {
    match (dx, dy) {
        (-1, 0) => 0,
        (-1, -1) => 1,
        (0, -1) => 2,
        (1, -1) => 3,
        (1, 0) => 4,
        (1, 1) => 5,
        (0, 1) => 6,
        (-1, 1) => 7,
        _ => -1,
    }
}

fn distance(a: Point, b: Point) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    (dx * dx + dy * dy).sqrt()
}

fn average_corner_distance(left: Quad, right: Quad) -> f64 {
    let mut min_distance = f64::INFINITY;
    for shift in 0..4 {
        let mut distance_sum = 0.0;
        for index in 0..4 {
            distance_sum += distance(left[(index + shift) % 4], right[index]);
        }
        min_distance = min_distance.min(distance_sum * 0.25);
    }

    min_distance
}

fn border_errors(cell_white_ratios: &[f32; GRID_SIZE * GRID_SIZE]) -> usize {
    let mut errors = 0usize;
    for row in 0..GRID_SIZE {
        for column in 0..GRID_SIZE {
            if (row == 0 || column == 0 || row == GRID_SIZE - 1 || column == GRID_SIZE - 1)
                && cell_white_ratios[row * GRID_SIZE + column] as f64 > VALID_BIT_ID_THRESHOLD
            {
                errors += 1;
            }
        }
    }

    errors
}

fn canonical_mean_stddev(canonical: &[u8; CANONICAL_IMAGE_PIXELS]) -> (f64, f64) {
    let inner_start = CANONICAL_CELL_SIZE / 2;
    let inner_end = CANONICAL_SIZE - inner_start;
    let mut total = 0.0;
    let mut total_sq = 0.0;
    let mut count: f64 = 0.0;

    for row in inner_start..inner_end {
        for column in inner_start..inner_end {
            let value = canonical[row * CANONICAL_SIZE + column] as f64;
            total += value;
            total_sq += value * value;
            count += 1.0;
        }
    }

    let mean = total / count.max(1.0);
    let variance = (total_sq / count.max(1.0) - mean * mean).max(0.0);
    (mean, variance.sqrt())
}

fn candidate_near_border(corners: Quad, image_width: f64, image_height: f64) -> bool {
    corners.iter().any(|corner| {
        corner.x < MIN_DISTANCE_TO_BORDER
            || corner.y < MIN_DISTANCE_TO_BORDER
            || corner.x > image_width - 1.0 - MIN_DISTANCE_TO_BORDER
            || corner.y > image_height - 1.0 - MIN_DISTANCE_TO_BORDER
    })
}

fn filter_candidate_groups(
    mut candidates: Vec<Candidate>,
    image_width: f64,
    image_height: f64,
) -> Vec<Candidate> {
    candidates.sort_by(|left, right| {
        right
            .perimeter
            .partial_cmp(&left.perimeter)
            .unwrap_or(Ordering::Equal)
    });

    let mut selected: Vec<Candidate> = Vec::with_capacity(candidates.len());
    'candidate: for candidate in candidates {
        if candidate_near_border(candidate.corners, image_width, image_height) {
            continue;
        }

        for other in &selected {
            let average_distance = average_corner_distance(candidate.corners, other.corners);
            if average_distance
                < candidate.perimeter.min(other.perimeter) * MIN_MARKER_DISTANCE_RATE
            {
                continue 'candidate;
            }
        }

        selected.push(candidate);
    }

    selected
}

fn dot_product(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn is_convex_quad(corners: Quad) -> bool {
    let mut all_positive = true;
    let mut all_negative = true;
    for index in 0..4 {
        let current = corners[index];
        let next = corners[(index + 1) % 4];
        let following = corners[(index + 2) % 4];
        let cross = (next.x - current.x) * (following.y - next.y)
            - (next.y - current.y) * (following.x - next.x);
        all_positive &= cross > 0.0;
        all_negative &= cross < 0.0;
    }
    all_positive || all_negative
}

fn has_tight_corners(corners: Quad, minimum_distance: f64) -> bool {
    for index in 0..4 {
        let current = corners[index];
        let next = corners[(index + 1) % 4];
        if distance(current, next) < minimum_distance {
            return true;
        }
    }

    false
}

fn interpolate_2d_line(points: &[Point]) -> Option<[f64; 3]> {
    if points.len() < 2 {
        return None;
    }

    let mut min_x = points[0].x;
    let mut max_x = points[0].x;
    let mut min_y = points[0].y;
    let mut max_y = points[0].y;
    for point in points {
        min_x = min_x.min(point.x);
        max_x = max_x.max(point.x);
        min_y = min_y.min(point.y);
        max_y = max_y.max(point.y);
    }

    if max_x - min_x > max_y - min_y {
        let (slope, intercept) = fit_linear_line(points, true)?;
        Some([slope, -1.0, intercept])
    } else {
        let (slope, intercept) = fit_linear_line(points, false)?;
        Some([-1.0, slope, intercept])
    }
}

fn fit_linear_line(points: &[Point], fit_y_from_x: bool) -> Option<(f64, f64)> {
    let mut sum_axis = 0.0;
    let mut sum_value = 0.0;
    let mut sum_axis_sq = 0.0;
    let mut sum_axis_value = 0.0;
    let count = points.len() as f64;

    for point in points {
        let axis = if fit_y_from_x { point.x } else { point.y };
        let value = if fit_y_from_x { point.y } else { point.x };
        sum_axis += axis;
        sum_value += value;
        sum_axis_sq += axis * axis;
        sum_axis_value += axis * value;
    }

    let denominator = count * sum_axis_sq - sum_axis * sum_axis;
    if denominator.abs() < HOMOGRAPHY_EPSILON {
        return None;
    }

    let slope = (count * sum_axis_value - sum_axis * sum_value) / denominator;
    let intercept = (sum_value - slope * sum_axis) / count;
    Some((slope, intercept))
}

fn marker_bits_from_white_ratios(
    white_ratios: &[f32; GRID_SIZE * GRID_SIZE],
) -> [u8; MARKER_SIZE * MARKER_SIZE] {
    let mut bits = [0u8; MARKER_SIZE * MARKER_SIZE];
    let mut bit_index = 0usize;

    for row in MARKER_BORDER_BITS..GRID_SIZE - MARKER_BORDER_BITS {
        for column in MARKER_BORDER_BITS..GRID_SIZE - MARKER_BORDER_BITS {
            bits[bit_index] =
                if white_ratios[row * GRID_SIZE + column] as f64 > VALID_BIT_ID_THRESHOLD {
                    1
                } else {
                    0
                };
            bit_index += 1;
        }
    }

    bits
}

fn marker_confidence_from_white_ratios(
    white_ratios: &[f32; GRID_SIZE * GRID_SIZE],
    dictionary_bytes: [u8; 2],
) -> f64 {
    let mut border_uncertainty = 0.0;
    for row in 0..GRID_SIZE {
        for column in 0..GRID_SIZE {
            if row == 0 || column == 0 || row == GRID_SIZE - 1 || column == GRID_SIZE - 1 {
                border_uncertainty += white_ratios[row * GRID_SIZE + column] as f64;
            }
        }
    }

    let expected_bits = unpack_marker_bits(dictionary_bytes);
    let mut inner_uncertainty = 0.0;
    for row in 0..MARKER_SIZE {
        for column in 0..MARKER_SIZE {
            let expected = expected_bits[row * MARKER_SIZE + column] as f64;
            let observed = white_ratios
                [(row + MARKER_BORDER_BITS) * GRID_SIZE + column + MARKER_BORDER_BITS]
                as f64;
            inner_uncertainty += (expected - observed).abs();
        }
    }

    let normalized_uncertainty =
        (border_uncertainty + inner_uncertainty) / (GRID_SIZE * GRID_SIZE) as f64;
    (1.0 - normalized_uncertainty).clamp(0.0, 1.0)
}

fn unpack_marker_bits(dictionary_bytes: [u8; 2]) -> [u8; MARKER_SIZE * MARKER_SIZE] {
    let mut bits = [0u8; MARKER_SIZE * MARKER_SIZE];
    for (index, bit) in bits.iter_mut().enumerate() {
        let byte = dictionary_bytes[index / 8];
        let shift = 7 - (index % 8);
        *bit = (byte >> shift) & 1;
    }
    bits
}

fn normalize_vector(vector: [f64; 3]) -> [f64; 3] {
    let length = vector_norm(vector);
    if length < HOMOGRAPHY_EPSILON {
        [0.0, 0.0, 0.0]
    } else {
        [vector[0] / length, vector[1] / length, vector[2] / length]
    }
}

fn pack_marker_bits(bits: &[u8; MARKER_SIZE * MARKER_SIZE]) -> [u8; 2] {
    let mut current_byte = 0usize;
    let mut current_bit = 0usize;
    let mut bytes = [0u8; 2];

    for bit in bits {
        bytes[current_byte] = (bytes[current_byte] << 1) | *bit;
        current_bit += 1;
        if current_bit == 8 {
            current_byte += 1;
            current_bit = 0;
        }
    }

    bytes
}

fn otsu_threshold_bytes(values: &[u8]) -> u8 {
    let mut histogram = [0u32; 256];
    for &value in values {
        histogram[value as usize] += 1;
    }

    let total = values.len() as f64;
    let mut total_sum = 0.0;
    for (index, count) in histogram.iter().enumerate() {
        total_sum += index as f64 * *count as f64;
    }

    let mut background_weight = 0.0;
    let mut background_sum = 0.0;
    let mut best_threshold = 0u8;
    let mut best_variance = -1.0;

    for (index, count) in histogram.iter().enumerate() {
        background_weight += *count as f64;
        if background_weight <= 0.0 {
            continue;
        }

        let foreground_weight = total - background_weight;
        if foreground_weight <= 0.0 {
            break;
        }

        background_sum += index as f64 * *count as f64;
        let background_mean = background_sum / background_weight;
        let foreground_mean = (total_sum - background_sum) / foreground_weight;
        let between_variance =
            background_weight * foreground_weight * (background_mean - foreground_mean).powi(2);

        if between_variance > best_variance {
            best_variance = between_variance;
            best_threshold = index as u8;
        }
    }

    best_threshold
}

fn polygon_area_from_quad(corners: Quad) -> f64 {
    let mut area = 0.0;
    for index in 0..4 {
        let current = corners[index];
        let next = corners[(index + 1) % 4];
        area += current.x * next.y - next.x * current.y;
    }
    area.abs() * 0.5
}

fn quad_perimeter(corners: Quad) -> f64 {
    distance(corners[0], corners[1])
        + distance(corners[1], corners[2])
        + distance(corners[2], corners[3])
        + distance(corners[3], corners[0])
}

fn project_homography(homography: &[f64; 8], x: f64, y: f64) -> Option<Point> {
    let denominator = homography[6] * x + homography[7] * y + 1.0;
    if denominator.abs() < HOMOGRAPHY_EPSILON {
        return None;
    }

    Some(Point {
        x: (homography[0] * x + homography[1] * y + homography[2]) / denominator,
        y: (homography[3] * x + homography[4] * y + homography[5]) / denominator,
    })
}

fn reorder_corners_clockwise(corners: &mut Quad) {
    let dx1 = corners[1].x - corners[0].x;
    let dy1 = corners[1].y - corners[0].y;
    let dx2 = corners[2].x - corners[0].x;
    let dy2 = corners[2].y - corners[0].y;
    let cross = dx1 * dy2 - dy1 * dx2;
    if cross < 0.0 {
        corners.swap(1, 3);
    }
}

fn refine_candidate_lines(contour: &[Point], corners: Quad) -> Option<Quad> {
    if contour.len() < 8 {
        return None;
    }

    let mut contour_groups: Vec<Vec<Point>> =
        vec![Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut corner_index = [-1isize; 4];
    let mut group = 4usize;

    for (index, &point) in contour.iter().enumerate() {
        for corner in 0..4 {
            if point.x == corners[corner].x && point.y == corners[corner].y {
                corner_index[corner] = index as isize;
                group = corner;
                break;
            }
        }
        contour_groups[group].push(point);
    }

    if corner_index.iter().any(|index| *index < 0) {
        return None;
    }

    if !contour_groups[4].is_empty() {
        let extra = core::mem::take(&mut contour_groups[4]);
        contour_groups[group].extend(extra);
    }

    let mut direction = 1isize;
    if corner_index[0] > corner_index[1] && corner_index[3] > corner_index[0] {
        direction = -1;
    }
    if corner_index[2] > corner_index[3] && corner_index[1] > corner_index[2] {
        direction = -1;
    }

    let mut lines = [[0.0; 3]; 4];
    for index in 0..4 {
        lines[index] = interpolate_2d_line(&contour_groups[index])?;
    }

    let mut refined = [Point::default(); 4];
    for index in 0..4 {
        refined[index] = if direction < 0 {
            cross_point(lines[index], lines[(index + 1) % 4])?
        } else {
            cross_point(lines[index], lines[(index + 3) % 4])?
        };
    }

    reorder_corners_clockwise(&mut refined);
    if !corners_are_unique(refined) || !is_convex_quad(refined) {
        return None;
    }

    Some(refined)
}

fn cross_point(line_a: [f64; 3], line_b: [f64; 3]) -> Option<Point> {
    let determinant = line_a[0] * line_b[1] - line_b[0] * line_a[1];
    if determinant.abs() < HOMOGRAPHY_EPSILON {
        return None;
    }

    Some(Point {
        x: (line_a[1] * line_b[2] - line_b[1] * line_a[2]) / determinant,
        y: (line_b[0] * line_a[2] - line_a[0] * line_b[2]) / determinant,
    })
}

fn project_pose_point(
    point: [f64; 3],
    r1: [f64; 3],
    r2: [f64; 3],
    r3: [f64; 3],
    translation: [f64; 3],
    focal_x: f64,
    focal_y: f64,
    principal_x: f64,
    principal_y: f64,
) -> Option<Point> {
    let camera_x = r1[0] * point[0] + r2[0] * point[1] + r3[0] * point[2] + translation[0];
    let camera_y = r1[1] * point[0] + r2[1] * point[1] + r3[1] * point[2] + translation[1];
    let camera_z = r1[2] * point[0] + r2[2] * point[1] + r3[2] * point[2] + translation[2];

    if camera_z <= HOMOGRAPHY_EPSILON {
        return None;
    }

    Some(Point {
        x: focal_x * (camera_x / camera_z) + principal_x,
        y: focal_y * (camera_y / camera_z) + principal_y,
    })
}

fn quaternion_from_basis(r1: [f64; 3], r2: [f64; 3], r3: [f64; 3]) -> [f64; 4] {
    let m00 = r1[0];
    let m01 = r2[0];
    let m02 = r3[0];
    let m10 = r1[1];
    let m11 = r2[1];
    let m12 = r3[1];
    let m20 = r1[2];
    let m21 = r2[2];
    let m22 = r3[2];
    let trace = m00 + m11 + m22;

    let quaternion = if trace > 0.0 {
        let scale = (trace + 1.0).sqrt() * 2.0;
        [
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
            0.25 * scale,
        ]
    } else if m00 > m11 && m00 > m22 {
        let scale = (1.0 + m00 - m11 - m22).sqrt() * 2.0;
        [
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
            (m21 - m12) / scale,
        ]
    } else if m11 > m22 {
        let scale = (1.0 + m11 - m00 - m22).sqrt() * 2.0;
        [
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
            (m02 - m20) / scale,
        ]
    } else {
        let scale = (1.0 + m22 - m00 - m11).sqrt() * 2.0;
        [
            (m02 + m20) / scale,
            (m12 + m21) / scale,
            0.25 * scale,
            (m10 - m01) / scale,
        ]
    };

    let length = (quaternion[0] * quaternion[0]
        + quaternion[1] * quaternion[1]
        + quaternion[2] * quaternion[2]
        + quaternion[3] * quaternion[3])
        .sqrt();

    if length < HOMOGRAPHY_EPSILON {
        [0.0, 0.0, 0.0, 1.0]
    } else {
        [
            quaternion[0] / length,
            quaternion[1] / length,
            quaternion[2] / length,
            quaternion[3] / length,
        ]
    }
}

fn rotate_quad(corners: Quad, shift: usize) -> Quad {
    [
        corners[shift % 4],
        corners[(shift + 1) % 4],
        corners[(shift + 2) % 4],
        corners[(shift + 3) % 4],
    ]
}

fn scale_quad(corners: Quad, scale_x: f64, scale_y: f64) -> Quad {
    corners.map(|corner| Point {
        x: corner.x * scale_x,
        y: corner.y * scale_y,
    })
}

fn scale_vector(vector: [f64; 3], scalar: f64) -> [f64; 3] {
    [vector[0] * scalar, vector[1] * scalar, vector[2] * scalar]
}

fn subtract_vectors(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn vector_norm(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

unsafe fn detector_mut<'a>(ptr: *mut Detector) -> &'a mut Detector {
    &mut *ptr
}

unsafe fn detector_ref<'a>(ptr: *const Detector) -> &'a Detector {
    &*ptr
}

#[no_mangle]
pub extern "C" fn detector_new() -> *mut Detector {
    Box::into_raw(Box::new(Detector::new()))
}

#[no_mangle]
pub extern "C" fn detector_free(detector: *mut Detector) {
    if detector.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(detector));
    }
}

#[no_mangle]
pub extern "C" fn detector_configure_frame(
    detector: *mut Detector,
    source_width: u32,
    source_height: u32,
) -> u32 {
    if detector.is_null() {
        return 0;
    }
    unsafe { detector_mut(detector).configure_frame(source_width as usize, source_height as usize) }
}

#[no_mangle]
pub extern "C" fn detector_prepare_rgba(detector: *mut Detector, len: u32) -> *mut u8 {
    if detector.is_null() {
        return core::ptr::null_mut();
    }
    unsafe { detector_mut(detector).prepare_rgba(len as usize) }
}

#[no_mangle]
pub extern "C" fn detector_set_camera_intrinsics(
    detector: *mut Detector,
    focal_length_x: f64,
    focal_length_y: f64,
    principal_x: f64,
    principal_y: f64,
    focal_length_scale: f64,
) {
    if detector.is_null() {
        return;
    }
    unsafe {
        detector_mut(detector).set_camera_intrinsics(
            focal_length_x,
            focal_length_y,
            principal_x,
            principal_y,
            focal_length_scale,
        );
    }
}

#[no_mangle]
pub extern "C" fn detector_set_input_size(
    detector: *mut Detector,
    input_width: u32,
    input_height: u32,
) -> u32 {
    if detector.is_null() {
        return 0;
    }
    unsafe { detector_mut(detector).set_input_size(input_width as usize, input_height as usize) }
}

#[no_mangle]
pub extern "C" fn detector_detect(
    detector: *mut Detector,
    source_width: u32,
    source_height: u32,
) {
    if detector.is_null() {
        return;
    }
    unsafe {
        detector_mut(detector).detect(source_width as usize, source_height as usize);
    }
}

#[no_mangle]
pub extern "C" fn detector_result_ptr(detector: *const Detector) -> *const f64 {
    if detector.is_null() {
        return core::ptr::null();
    }
    unsafe { detector_ref(detector).result.as_ptr() }
}

#[no_mangle]
pub extern "C" fn detector_result_len(detector: *const Detector) -> u32 {
    if detector.is_null() {
        return 0;
    }
    unsafe { detector_ref(detector).result.len() as u32 }
}
