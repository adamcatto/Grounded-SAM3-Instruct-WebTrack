# Cross-camera floor registration

## Motivation

Projects commonly contain repeated recordings from a fixed set of stationary
cameras. Each camera observes the same nominally square arena floor, but
perspective makes that floor appear as a different quadrilateral. Consequently,
raw pixel coordinates, distances, areas, headings, and locomotion measurements
are not directly comparable across cameras.

## Registration workflow

We added a project-level **Registration** annotation workflow:

1. The user selects **Register videos** beneath the project video-progress
   legend.
2. The application creates one registration entry per video and always displays
   physical frame 0, independently of the video's chosen tracking start frame.
3. The user point-prompts SAM to segment the common square floor. Positive
   points identify the floor; negative points exclude surrounding structures,
   walls, animals, or other distractors. As a manual alternative, the user can
   draw three or more polygon vertices; the application fills their convex hull
   as the floor mask.
4. After all first frames have a floor mask, the user selects **Generate
   registration parameters**.
5. The largest floor-mask contour is converted to a convex four-corner
   quadrilateral. Its corners are ordered top-left, top-right, bottom-right,
   bottom-left and mapped to a shared square canvas.

Registration prompts and masks are stored separately from animal/object
tracking annotations, so calibration cannot add objects to or overwrite the
tracking masks.

### Mask cleanup and provenance

The floor mask can be refined with binary morphological opening or closing using
an adjustable elliptical kernel up to the smaller video dimension. Opening removes small foreground
islands and narrow protrusions; closing fills small holes and gaps. Each
application records the exact foreground pixels added and removed as a
compressed delta. A persistent 50-operation queue supports undo and redo without
lossy mask re-estimation, and reports added/removed pixel counts for every edit.
Applying a new operation after undo creates a new history branch and discards
the obsolete redo branch. Clearing a camera calibration removes its points,
floor mask, homography, and morphology history.

Polygon vertices are stored as normalized image coordinates and rasterized at
the native video resolution. Replacing a SAM mask with a polygon hull starts a
new mask-edit history and invalidates the prior homography.

### Partial views and lens distortion

For views where only three corners are visible but a portion of every physical
floor edge remains visible, the user can choose **Curved edges** and annotate
four color-coded edge groups (top, right, bottom, left). Each edge requires two
points, with three or more recommended wherever curvature is visible.

The calibration assumes the physical box edges are straight. It jointly fits
two radial lens-distortion coefficients (`k1`, `k2`) by minimizing the
point-to-line residual after undistortion (the plumb-line constraint). A weak
coefficient regularizer and robust loss limit overfitting. Total-least-squares
lines are then fit to the undistorted edge samples, and adjacent line
intersections extrapolate all four corners, including an occluded corner.

Tracked masks are registered in two stages with nearest-neighbor interpolation:

1. undistort using the fitted camera matrix and radial coefficients;
2. warp the undistorted floor quadrilateral to the canonical square.

The fitted straightness RMS in source-image pixels is stored as a calibration
diagnostic. The camera matrix currently fixes the principal point at image
center and uses the larger image dimension as the focal-length scale; radial
coefficients and planar homography are estimated per video.

### Visual quality control

The registration pane includes an arbitrary-frame comparison viewer. A
debounced frame slider, exact frame-number input, arrow controls, and keyboard
left/right navigation request frames on demand without changing annotation or
tracking state. The viewer can show the decoded source frame beside the
registered overhead square, making it possible to inspect calibration quality
at the beginning, middle, and end of a recording. This is useful for detecting
camera movement or preprocessing changes that violate the stationary-camera
assumption.

The same viewer also provides a multi-video grid for cross-camera quality
control. The user selects any subset of videos with completed registration and
requests one shared frame index; the application renders the registered
overhead view for every selection in a responsive grid. The frame control is
bounded by the shortest selected video so that the requested index exists in
every displayed recording. This view makes residual scale, orientation,
cropping, and lens-correction differences apparent without altering project
data.

Registered previews and downstream masks use a bounded full-frame mesh rather
than applying the planar homography infinitely beyond the floor. The warp is
exactly projective within the inferred floor quadrilateral, then transitions
smoothly toward a finite affine mapping near the outer image boundary. This
preserves the entire camera frame—even when a projective horizon crosses the
image—while retaining the shared square floor coordinate system. A single
project-level canvas bound and offset are computed across every camera, so all
registered outputs have identical dimensions and floor origin. The four floor
corners may be extrapolated from sampled edge lines; therefore the method does
not require every corner to be directly visible, although accuracy depends on
having enough visible edge structure to estimate those lines.

## Mathematical transform

Although the visual distortion can look like shear, an oblique camera view of a
plane generally requires a projective transform rather than an affine shear.
For each camera/video, four source floor corners are paired with the corners of
a canonical square (currently 1000 × 1000 pixels). OpenCV solves the 3 × 3
homography \(H\) such that, in homogeneous coordinates,

\[
\lambda
\begin{bmatrix}
x' \\ y' \\ 1
\end{bmatrix}
=
H
\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix}.
\]

The matrix, detected source corners, method name, and canonical output size are
persisted under the project-level `registration.videos[video_id]` record in
`config.json`. This makes calibration portable with the project and auditable.

## Downstream use

The common downstream mask-loading paths apply each video's homography with
nearest-neighbor interpolation before extracting centroids and behavioral
features. Thus:

- each registered camera uses the same square coordinate extent;
- locomotion distances are measured in a common floor-relative pixel space;
- centroid positions, mask areas, contour distances, overlap, body axes, and
  interaction features are computed from overhead-normalized masks;
- unregistered legacy projects retain their previous behavior unchanged.

The four arenas are geometrically identical but do not need to share a global
physical orientation. Downstream behavioral features are intentionally invariant
to the square's dihedral symmetries (90-degree rotations and reflections):
locomotion uses displacement magnitudes; proximity and contour measures use
distances; parallel motion uses normalized dot products; body turning uses
absolute angular changes; and social heading is expressed relative to the other
animal. Absolute global `x/y` and absolute compass heading are not included in
the sequence-level clustering feature vector.

Nearest-neighbor interpolation preserves binary mask semantics. Pixels outside
the mapped camera floor are filled with background.

## Methods considerations

- The calibration assumes the arena floor is planar and stationary relative to
  each camera.
- The segmented floor boundary must include all four visible corners.
- A geometrically perfect square is rotationally and reflectionally symmetric.
  The implementation intentionally leaves physical compass orientation
  unspecified because the downstream measurements are rotation/reflection
  invariant. Any future analysis of absolute corner occupancy or compass
  direction would require an additional shared-corner orientation cue.
- A homography corrects planar perspective but does not model lens distortion.
  If wide-angle distortion is material, intrinsic camera calibration and
  undistortion should precede this floor homography.
- A square canonical space normalizes camera geometry but is not automatically
  a physical unit. Converting pixels to centimeters requires the known floor
  side length; for a 1000-pixel target square, the scale is
  `floor_side_cm / 999` centimeters per pixel.
- Registration should be recomputed if a camera moves, zoom changes, the video
  is cropped, or preprocessing changes the image geometry.
