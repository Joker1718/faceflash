"""
FaceFlash Streamlit Web App — Full-Featured Demo

Run:
    pip install streamlit opencv-python pandas
    streamlit run examples/webapp_streamlit.py

Tabs:
    1. Register — upload or webcam, duplicate warnings
    2. Identify — upload/webcam with optional liveness check
    3. Live Stream — continuous webcam feed, face boxes, real-time analytics
    4. Analytics — event history, crowd stats, visit frequency, attendance
    5. Settings — profile, consent, schedule, webhook config
    6. Gallery — enrolled people, remove, stats, find duplicates
"""

import sys
import os
import time
import json
import sqlite3
import tempfile
import datetime
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import numpy as np
from faceflash.zones import (normalize_polygon, point_in_polygon,
                             polygon_for_frame, rect_to_polygon,
                             zone_counts, zone_payload)
from faceflash.events import EventStore

st.set_page_config(page_title="FaceFlash", page_icon="⚡", layout="wide")
st.markdown("""
<style>
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {
    width: 100% !important;
    max-height: 520px !important;
    object-fit: contain !important;
    background: #0f172a;
    border-radius: 8px;
}
[data-testid="stCameraInput"] {
    max-width: 760px;
}
</style>
""", unsafe_allow_html=True)

DB_PATH = "faceflash_demo.db"


def _init_stream_config():
    st.session_state.setdefault("zones", [])
    st.session_state.setdefault("stream_events_preview", [])
    st.session_state.setdefault("watchlist_names", [])


def _zone_rect_from_state(prefix: str, frame_w: int, frame_h: int):
    x1 = int(st.session_state.get(f"{prefix}_x1", 0))
    y1 = int(st.session_state.get(f"{prefix}_y1", 0))
    x2 = int(st.session_state.get(f"{prefix}_x2", frame_w))
    y2 = int(st.session_state.get(f"{prefix}_y2", frame_h))
    x1, x2 = sorted((max(0, min(frame_w, x1)), max(0, min(frame_w, x2))))
    y1, y2 = sorted((max(0, min(frame_h, y1)), max(0, min(frame_h, y2))))
    return x1, y1, x2, y2


def _rect_to_polygon(x1, y1, x2, y2):
    return rect_to_polygon(x1, y1, x2, y2)


def _normalize_polygon(poly, frame_w: int, frame_h: int):
    return normalize_polygon(poly, frame_w, frame_h)


def _polygon_for_frame(zone, frame_w: int, frame_h: int):
    return polygon_for_frame(zone, frame_w, frame_h)


def _zone_payload(name: str, polygon, secure: bool, frame_w: int, frame_h: int):
    return zone_payload(name, polygon, secure, frame_w, frame_h)


def _canvas_object_to_polygon(obj, scale: float = 1.0):
    """Convert a streamlit-drawable-canvas object to frame pixel points."""
    if not obj:
        return None

    kind = obj.get("type")
    if kind == "rect":
        left = float(obj.get("left", 0)) / scale
        top = float(obj.get("top", 0)) / scale
        width = float(obj.get("width", 0)) * float(obj.get("scaleX", 1)) / scale
        height = float(obj.get("height", 0)) * float(obj.get("scaleY", 1)) / scale
        if width <= 1 or height <= 1:
            return None
        return _rect_to_polygon(left, top, left + width, top + height)

    if kind == "path":
        pts = []
        for cmd in obj.get("path", []):
            if len(cmd) >= 3 and cmd[0] in ("M", "L"):
                # streamlit-drawable-canvas path coordinates are already in
                # canvas space; adding Fabric's object left/top shifts zones.
                pts.append((float(cmd[1]) / scale, float(cmd[2]) / scale))
        return pts if len(pts) >= 3 else None

    if kind == "polygon":
        left = float(obj.get("left", 0))
        top = float(obj.get("top", 0))
        pts = [((left + float(p["x"])) / scale,
                (top + float(p["y"])) / scale)
               for p in obj.get("points", [])]
        return pts if len(pts) >= 3 else None

    return None


def _canvas_object_center(obj, scale: float = 1.0):
    if not obj:
        return None
    kind = obj.get("type")
    if kind == "circle":
        radius = float(obj.get("radius", 0)) * float(obj.get("scaleX", 1))
        x = (float(obj.get("left", 0)) + radius) / scale
        y = (float(obj.get("top", 0)) + radius) / scale
        return x, y
    if kind == "rect":
        x = (float(obj.get("left", 0)) +
             float(obj.get("width", 0)) * float(obj.get("scaleX", 1)) / 2.0) / scale
        y = (float(obj.get("top", 0)) +
             float(obj.get("height", 0)) * float(obj.get("scaleY", 1)) / 2.0) / scale
        return x, y
    return None


def _draw_zones(frame_rgb, zones, draft=None):
    import cv2
    out = frame_rgb.copy()
    palette = [(14, 165, 233), (34, 197, 94), (245, 158, 11), (239, 68, 68),
               (168, 85, 247), (20, 184, 166)]
    overlay = out.copy()

    def draw(poly, label, color, fill_alpha=0.18, dashed=False):
        pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(out, [pts], True, color, 2 if not dashed else 1)
        x, y = pts[:, 0, 0].min(), pts[:, 0, 1].min()
        cv2.rectangle(out, (x, max(0, y - 24)), (x + max(90, 8 * len(label)), y),
                      color, -1)
        cv2.putText(out, label, (x + 6, max(16, y - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1,
                    cv2.LINE_AA)

    h, w = frame_rgb.shape[:2]
    for i, z in enumerate(zones):
        label = z["name"] + (" secure" if z.get("secure") else "")
        draw(_polygon_for_frame(z, w, h), label, palette[i % len(palette)])

    if draft is not None:
        draw(draft["polygon"], "draft", (255, 255, 255), fill_alpha=0.1, dashed=True)

    cv2.addWeighted(overlay, 0.22, out, 0.78, 0, out)
    return out


def _zone_counts(detections, zones, frame_w: int, frame_h: int):
    return zone_counts(detections, zones, frame_w, frame_h)


def _stream_config_snippet(mode, threshold, sample_fps, identify_interval,
                           det_model, det_size, zones, capacity_limit,
                           capacity_zone, webhook_url, watchlist):
    zone_lines = []
    for z in zones:
        zone_lines.append(f'monitor.add_zone({z["name"]!r}, {z["polygon"]!r})')
        if z.get("secure"):
            zone_lines.append(f'monitor.set_secure_zone({z["name"]!r})')
    if capacity_limit:
        zone_arg = None if capacity_zone == "Whole frame" else capacity_zone
        zone_lines.append(f"monitor.set_capacity({int(capacity_limit)}, zone={zone_arg!r})")
    if webhook_url:
        zone_lines.append(f"monitor.on_event('{webhook_url}')")
    body = "\n".join(f"    {line}" for line in zone_lines) or "    # Add zones from the UI first"
    return f"""from faceflash import FaceFlash
from faceflash.stream import StreamMonitor

ff = FaceFlash({DB_PATH!r})
monitor = StreamMonitor(
    ff,
    mode={mode!r},
    watchlist={watchlist!r},
    threshold={threshold},
    sample_fps={sample_fps},
    identify_interval={identify_interval},
    det_model={det_model!r},
    det_size={det_size},
)
{body}
monitor.run(0)
"""


@st.cache_resource
def get_engine(profile=None):
    """Lazy-load FaceFlash engine (cached across reruns).
    profile=None auto-detects from an existing gallery."""
    from faceflash import FaceFlash
    return FaceFlash(DB_PATH, profile=profile)


def get_db_connection():
    """Direct SQLite connection for analytics queries."""
    if not Path(DB_PATH).exists():
        return None
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def main():
    _init_stream_config()
    st.title("⚡ FaceFlash")
    st.caption("Face search that fits in 6 MB — register, identify, stream, analyze.")

    # Load engine with profile from session state
    profile = st.session_state.get("profile", "light")
    ff = get_engine(profile)

    # Sidebar summary
    with st.sidebar:
        st.header("Gallery Summary")
        names = ff.names()
        st.metric("People enrolled", len(names))
        st.metric("Total faces", len(ff))
        if names:
            st.caption(", ".join(names[:20]))
            if len(names) > 20:
                st.caption(f"… and {len(names) - 20} more")

    # Tabs
    tab_register, tab_identify, tab_live, tab_analytics, tab_test, tab_settings, tab_gallery = st.tabs([
        "📷 Register", "🔍 Identify", "📹 Live Stream",
        "📊 Analytics", "🧪 Test Console", "⚙️ Settings", "🖼️ Gallery"
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 1: Register
    # ═══════════════════════════════════════════════════════════════════════
    with tab_register:
        st.subheader("Register a new face")
        name = st.text_input("Name", placeholder="e.g. Alice", key="reg_name")

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Upload a photo**")
            uploaded = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"],
                                        key="reg_upload")
            if uploaded and name:
                if st.button("Register from upload", key="btn_reg_upload"):
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = tmp.name
                    try:
                        result = ff.register(name, tmp_path, check_duplicate=True)
                        st.success(f"Registered **{name}**! Gallery now has {result['index_size']} faces.")
                        if result.get("duplicate_of"):
                            dup = result["duplicate_of"]
                            st.warning(
                                f"⚠️ Possible duplicate: looks like **{dup['name']}** "
                                f"(similarity {dup['confidence']:.4f})")
                    except Exception as e:
                        st.error(f"Registration failed: {e}")
                    finally:
                        os.unlink(tmp_path)

        with col2:
            st.write("**Webcam snapshot**")
            camera_photo = st.camera_input("Take a photo", key="reg_cam")
            if camera_photo and name:
                if st.button("Register from webcam", key="btn_reg_webcam"):
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp.write(camera_photo.read())
                        tmp_path = tmp.name
                    try:
                        result = ff.register(name, tmp_path, check_duplicate=True)
                        st.success(f"Registered **{name}**! Gallery now has {result['index_size']} faces.")
                        if result.get("duplicate_of"):
                            dup = result["duplicate_of"]
                            st.warning(
                                f"⚠️ Possible duplicate: looks like **{dup['name']}** "
                                f"(similarity {dup['confidence']:.4f})")
                    except Exception as e:
                        st.error(f"Registration failed: {e}")
                    finally:
                        os.unlink(tmp_path)

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 2: Identify
    # ═══════════════════════════════════════════════════════════════════════
    with tab_identify:
        st.subheader("Identify a face")

        input_source = st.radio("Input source", ["Upload", "Camera"],
                                horizontal=True, key="id_input_source")
        uploaded_q = None
        camera_q = None
        if input_source == "Upload":
            uploaded_q = st.file_uploader("Upload query image",
                                          type=["jpg", "jpeg", "png"],
                                          key="id_upload")
        else:
            with st.container():
                camera_q = st.camera_input("Take a query photo", key="id_cam")

        check_live = st.checkbox("Enable liveness check (anti-spoofing)", value=False,
                                  key="id_liveness")

        query_image = uploaded_q or camera_q
        if query_image:
            if st.button("Identify", key="btn_identify"):
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    tmp.write(query_image.read())
                    tmp_path = tmp.name
                try:
                    # Liveness check first if requested
                    if check_live:
                        liveness_result = ff.check_liveness(tmp_path)
                        if not liveness_result["live"]:
                            st.error(f"🚫 SPOOF DETECTED — {liveness_result['label']} "
                                     f"(score: {liveness_result['liveness_score']:.4f})")
                            st.stop()
                        else:
                            st.info(f"✅ Live face confirmed (score: {liveness_result['liveness_score']:.4f})")

                    result = ff.search(tmp_path, k=5)

                    if result["matches"]:
                        st.write("**Matches found:**")
                        for m in result["matches"]:
                            confidence_pct = m["confidence"] * 100
                            st.success(f"**{m['name']}** — {confidence_pct:.1f}% confidence")
                        st.caption(f"Search time: {result['time_ms']:.1f}ms | "
                                   f"Embed: {result['embed_time_ms']:.1f}ms | "
                                   f"Index: {result['search_time_ms']:.1f}ms")
                    else:
                        st.warning("No match found in gallery.")
                        if result.get("filtered"):
                            st.info(f"Filtered: {result['filtered']}")

                except Exception as e:
                    st.error(f"Identification failed: {e}")
                finally:
                    os.unlink(tmp_path)

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 3: Live Stream
    # ═══════════════════════════════════════════════════════════════════════
    with tab_live:
        st.subheader("Live identification stream")
        st.caption("Continuous webcam feed with face detection, identification, and real-time analytics.")

        if len(ff) == 0:
            st.warning("Register at least one face first.")
        else:
            st.write("**Camera profile**")
            saved_cameras = ff.index.list_camera_configs()
            saved_names = [c["name"] for c in saved_cameras]
            col_cam1, col_cam2, col_cam3 = st.columns([1.4, 1, 1])
            with col_cam1:
                camera_name = st.text_input("Camera name", value="default",
                                            key="camera_name")
            with col_cam2:
                camera_source = st.text_input("Source", value="0",
                                              key="camera_source",
                                              help="Use 0 for webcam or an RTSP/HTTP URL.")
            with col_cam3:
                selected_camera = st.selectbox(
                    "Load saved", [""] + saved_names, key="load_camera_name")
                if selected_camera and st.button("Load camera", key="btn_load_camera"):
                    cfg = ff.index.get_camera_config(selected_camera)
                    if cfg:
                        st.session_state["camera_name"] = cfg["name"]
                        st.session_state["camera_source"] = cfg.get("source", "0")
                        st.session_state["stream_mode"] = cfg.get("mode", "identify")
                        st.session_state["stream_threshold"] = float(cfg.get("threshold", 0.4))
                        st.session_state["stream_det_model"] = cfg.get("det_model", "500m")
                        st.session_state["stream_det_size"] = int(cfg.get("det_size", 480))
                        st.session_state["stream_sample_fps"] = float(cfg.get("sample_fps", 2.0))
                        st.session_state["stream_liveness"] = bool(cfg.get("check_liveness", False))
                        st.session_state["watchlist_names"] = cfg.get("watchlist", [])
                        zones = []
                        for z in ff.index.list_zone_configs(selected_camera):
                            zones.append({
                                "name": z["name"],
                                "polygon": z["polygon"],
                                "polygon_norm": z["polygon"],
                                "frame_size": [1, 1],
                                "secure": z.get("secure", False),
                            })
                        st.session_state["zones"] = zones
                        st.rerun()

            col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)
            with col_ctrl1:
                stream_mode = st.selectbox(
                    "Mode",
                    ["identify", "anonymous", "watchlist"],
                    key="stream_mode",
                    help="Anonymous counts visitors without identifying. Watchlist only alerts on selected names.")
                enable_liveness = st.checkbox("Liveness check", value=False,
                                              key="stream_liveness")
            with col_ctrl2:
                threshold = st.slider("Match threshold", 0.1, 0.9, 0.4, 0.05,
                                      key="stream_threshold")
                det_model = st.selectbox("Detector", ["500m", "10g"], key="stream_det_model",
                                         help="500m is faster; 10g is more accurate.")
            with col_ctrl3:
                sample_fps = st.slider("Sample fps", 0.5, 10.0, 2.0, 0.5,
                                       key="stream_sample_fps")
                det_size = st.select_slider("Detector size", [320, 480, 640],
                                            value=480, key="stream_det_size")
                run_stream = st.checkbox("Start live detection", value=False,
                                          key="stream_run")

            watchlist = []
            if stream_mode == "watchlist":
                watchlist = st.multiselect("Watchlist names", names,
                                           default=st.session_state.get("watchlist_names", []),
                                           key="stream_watchlist")
                st.session_state["watchlist_names"] = watchlist
                if not watchlist:
                    st.warning("Watchlist mode needs at least one selected name.")

            # ── Zone configuration ──────────────────────────────────────────
            with st.expander("Zone editor", expanded=True):
                st.caption("Capture one frame, draw a rectangle or polygon, then save it as a named zone.")

                # Capture a reference frame to show dimensions
                frame_w, frame_h = st.session_state.get("frame_size", (640, 480))
                col_cap, col_io = st.columns([1, 2])
                with col_cap:
                    capture_zone = st.button("Capture reference frame", key="btn_capture_zone",
                                             use_container_width=True)
                with col_io:
                    imported = st.text_area("Import zones JSON", height=80,
                                            placeholder='[{"name":"door","polygon":[[0,0],[300,0],[300,400],[0,400]],"secure":true}]',
                                            key="zones_import")
                    if st.button("Load zones JSON", key="btn_load_zones"):
                        try:
                            zones = json.loads(imported)
                            clean = []
                            for z in zones:
                                poly = [(float(x), float(y)) for x, y in z["polygon"]]
                                if len(poly) < 3:
                                    raise ValueError(f"zone {z.get('name', '<unnamed>')} needs 3+ points")
                                base_w, base_h = z.get("frame_size", [frame_w, frame_h])
                                item = _zone_payload(
                                    str(z["name"]), poly, bool(z.get("secure", False)),
                                    int(base_w), int(base_h))
                                if z.get("polygon_norm"):
                                    item["polygon_norm"] = [
                                        (float(x), float(y)) for x, y in z["polygon_norm"]]
                                clean.append(item)
                            st.session_state["zones"] = clean
                            st.success(f"Loaded {len(clean)} zone(s).")
                        except Exception as e:
                            st.error(f"Invalid zone JSON: {e}")

                if capture_zone:
                    try:
                        import cv2
                        source = int(camera_source) if str(camera_source).isdigit() else camera_source
                        cap = cv2.VideoCapture(source)
                        ret, frame = cap.read()
                        cap.release()
                        if ret:
                            st.session_state["zone_frame"] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            h, w = frame.shape[:2]
                            st.session_state["frame_size"] = (w, h)
                            st.session_state.setdefault("draft_x2", w)
                            st.session_state.setdefault("draft_y2", h)
                    except Exception as e:
                        st.error(f"Cannot capture: {e}")

                edit_request = st.session_state.pop("zone_edit_request", None)
                if edit_request:
                    pts = edit_request["polygon"]
                    xs = [p[0] for p in pts]
                    ys = [p[1] for p in pts]
                    st.session_state["zone_draw_name"] = edit_request["name"]
                    st.session_state["draft_x1"] = int(min(xs))
                    st.session_state["draft_y1"] = int(min(ys))
                    st.session_state["draft_x2"] = int(max(xs))
                    st.session_state["draft_y2"] = int(max(ys))
                    st.session_state["zone_draw_secure"] = bool(edit_request.get("secure"))

                col_name, col_draw, col_secure = st.columns([2, 1, 1])
                with col_name:
                    zone_name = st.text_input("Zone name", placeholder="e.g. entrance",
                                              key="zone_draw_name")
                with col_draw:
                    draw_mode_label = st.selectbox("Draw mode", ["Rectangle", "Polygon"],
                                                   key="zone_draw_mode")
                with col_secure:
                    is_secure = st.checkbox("Secure zone", key="zone_draw_secure",
                                            help="Alert on identified + unknown together.")

                drawn_polygon = None
                draft_zone = None

                if "zone_frame" in st.session_state:
                    try:
                        from PIL import Image
                        from streamlit_drawable_canvas import st_canvas

                        bg = _draw_zones(
                            st.session_state["zone_frame"],
                            st.session_state.get("zones", []))
                        display_w = min(frame_w, 900)
                        scale = display_w / max(frame_w, 1)
                        display_h = int(round(frame_h * scale))
                        bg_img = Image.fromarray(bg).resize((display_w, display_h))
                        canvas_result = st_canvas(
                            fill_color="rgba(20, 184, 166, 0.24)",
                            stroke_width=3,
                            stroke_color="#14b8a6",
                            background_image=bg_img,
                            height=display_h,
                            width=display_w,
                            drawing_mode="rect" if draw_mode_label == "Rectangle" else "polygon",
                            update_streamlit=True,
                            key="zone_canvas",
                        )
                        objects = (canvas_result.json_data or {}).get("objects", [])
                        if objects:
                            drawn_polygon = _canvas_object_to_polygon(objects[-1], scale=scale)
                            if drawn_polygon:
                                draft_zone = {"name": "draft", "polygon": drawn_polygon}
                                st.caption(
                                    f"Ready to save: {len(drawn_polygon)} point zone on "
                                    f"{frame_w} x {frame_h} frame.")
                    except Exception as e:
                        st.warning(f"Drawing canvas unavailable, use fine-tune sliders below: {e}")
                else:
                    st.info("Capture a reference frame to draw zones.")

                with st.expander("Fallback: fine tune a rectangle with sliders"):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.slider("Left", 0, frame_w,
                                  min(st.session_state.get("draft_x1", 0), frame_w),
                                  key="draft_x1")
                    with col2:
                        st.slider("Top", 0, frame_h,
                                  min(st.session_state.get("draft_y1", 0), frame_h),
                                  key="draft_y1")
                    with col3:
                        st.slider("Right", 0, frame_w,
                                  min(st.session_state.get("draft_x2", frame_w), frame_w),
                                  key="draft_x2")
                    with col4:
                        st.slider("Bottom", 0, frame_h,
                                  min(st.session_state.get("draft_y2", frame_h), frame_h),
                                  key="draft_y2")
                    x1, y1, x2, y2 = _zone_rect_from_state("draft", frame_w, frame_h)
                    slider_zone = {"name": "draft",
                                   "polygon": _rect_to_polygon(x1, y1, x2, y2)}
                    if "zone_frame" in st.session_state:
                        preview = _draw_zones(
                            st.session_state["zone_frame"],
                            st.session_state.get("zones", []),
                            draft=slider_zone)
                        st.image(preview, caption="Slider rectangle preview",
                                 use_container_width=True)

                zone_to_save = draft_zone or slider_zone
                if st.button("Save drawn zone", key="btn_save_zone"):
                    if not zone_name:
                        st.warning("Enter a zone name.")
                    elif not zone_to_save or len(zone_to_save["polygon"]) < 3:
                        st.error("Draw a rectangle or polygon first.")
                    else:
                        zones = st.session_state.get("zones", [])
                        zones = [z for z in zones if z["name"] != zone_name]
                        zones.append(_zone_payload(
                            zone_name, zone_to_save["polygon"], is_secure,
                            frame_w, frame_h))
                        st.session_state["zones"] = zones
                        st.success(f"Zone '{zone_name}' saved.")

                # Show defined zones
                zones = st.session_state.get("zones", [])
                if zones:
                    st.write("**Active zones**")
                    for i, z in enumerate(zones):
                        pts = _polygon_for_frame(z, frame_w, frame_h)
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        col_z1, col_z2, col_z3 = st.columns([3, 1, 1])
                        with col_z1:
                            st.caption(
                                f"{z['name']}: {len(pts)} points, "
                                f"bounds ({int(min(xs))},{int(min(ys))}) "
                                f"to ({int(max(xs))},{int(max(ys))})"
                                + (" [secure]" if z.get("secure") else ""))
                        with col_z2:
                            if st.button("Edit", key=f"edit_zone_{i}"):
                                st.session_state["zone_edit_request"] = z
                                st.rerun()
                        with col_z3:
                            if st.button("Delete", key=f"delete_zone_{i}"):
                                st.session_state["zones"] = [x for j, x in enumerate(zones) if j != i]
                                st.rerun()

                    export_zones = json.dumps(zones, indent=2)
                    st.download_button("Download zones JSON", export_zones,
                                       file_name="faceflash_zones.json",
                                       mime="application/json",
                                       use_container_width=True)
                    if st.button("Clear all zones", key="btn_clear_zones"):
                        st.session_state["zones"] = []
                        st.rerun()

            with st.expander("Advanced stream test config"):
                capacity_limit = st.number_input("Capacity limit", min_value=0, value=0,
                                                 help="0 disables capacity alerts.",
                                                 key="capacity_limit")
                zone_names = ["Whole frame"] + [z["name"] for z in st.session_state.get("zones", [])]
                capacity_zone = st.selectbox("Capacity scope", zone_names, key="capacity_zone")
                webhook_url = st.session_state.get("webhook_url", "")
                snippet = _stream_config_snippet(
                    stream_mode, threshold, sample_fps, 5, det_model, det_size,
                    st.session_state.get("zones", []), capacity_limit,
                    capacity_zone, webhook_url, watchlist)
                st.code(snippet, language="python")
                if st.button("Save camera config", key="btn_save_camera_config"):
                    capacity = None
                    if capacity_limit:
                        capacity = {
                            "limit": int(capacity_limit),
                            "zone": None if capacity_zone == "Whole frame" else capacity_zone,
                        }
                    ff.index.save_camera_config(
                        camera_name,
                        source=camera_source,
                        mode=stream_mode,
                        threshold=float(threshold),
                        sample_fps=float(sample_fps),
                        identify_interval=5,
                        det_model=det_model,
                        det_size=int(det_size),
                        check_liveness=bool(enable_liveness),
                        capacity=capacity,
                        watchlist=watchlist,
                        webhook_url=webhook_url or None,
                    )
                    for z in st.session_state.get("zones", []):
                        norm = z.get("polygon_norm") or _normalize_polygon(
                            z["polygon"], *z.get("frame_size", [frame_w, frame_h]))
                        ff.index.save_zone_config(
                            camera_name, z["name"], norm,
                            secure=bool(z.get("secure", False)))
                    st.success(f"Saved camera profile '{camera_name}'.")

            if run_stream:
                # Layout: video on left, analytics on right
                col_video, col_stats = st.columns([3, 1])

                with col_stats:
                    crowd_placeholder = st.empty()
                    visible_placeholder = st.empty()

                with col_video:
                    stframe = st.empty()

                try:
                    import cv2
                    from faceflash.stream import StreamMonitor

                    source = int(camera_source) if str(camera_source).isdigit() else camera_source
                    cap = cv2.VideoCapture(source)
                    if not cap.isOpened():
                        st.error("Cannot open webcam.")
                    else:
                        monitor = StreamMonitor(
                            ff,
                            sample_fps=float(sample_fps),
                            identify_interval=5,
                            threshold=float(threshold),
                            det_model=det_model,
                            det_size=int(det_size),
                            mode=stream_mode,
                            watchlist=watchlist or None,
                            check_liveness=enable_liveness,
                            camera_name=camera_name,
                            min_sharpness=0.0,
                        )
                        live_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame_h)
                        live_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame_w)
                        for z in st.session_state.get("zones", []):
                            poly = _polygon_for_frame(z, live_w, live_h)
                            monitor.add_zone(z["name"], poly)
                            if z.get("secure"):
                                monitor.set_secure_zone(z["name"])
                        if capacity_limit:
                            monitor.set_capacity(
                                int(capacity_limit),
                                None if capacity_zone == "Whole frame" else capacity_zone)
                        if webhook_url:
                            monitor.on_event(webhook_url)

                        frame_delay = 1.0 / max(1.0, sample_fps * 8)

                        while run_stream:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            snapshot = monitor.process_frame(frame, annotate=True)
                            shown = snapshot["frame"] if snapshot["frame"] is not None else frame
                            stframe.image(cv2.cvtColor(shown, cv2.COLOR_BGR2RGB),
                                          channels="RGB", use_container_width=True)

                            tracks = snapshot["tracks"]
                            known = [t for t in tracks if t["identity"]]
                            crowd_placeholder.metric("Crowd count", len(tracks))
                            if known:
                                names_visible = [t["identity"] for t in known]
                                visible_placeholder.write("**Visible:** " + ", ".join(names_visible))
                            else:
                                visible_placeholder.write("**Visible:** —")
                            counts = snapshot["report"].get("zones", {})
                            if counts:
                                visible_placeholder.write("**Zone counts**")
                                visible_placeholder.json(counts)
                            if snapshot["events"]:
                                visible_placeholder.write("**Latest events**")
                                visible_placeholder.json(snapshot["events"][-5:])

                            time.sleep(frame_delay)

                        cap.release()

                except ImportError:
                    st.error("opencv-python required: `pip install opencv-python`")

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 4: Analytics
    # ═══════════════════════════════════════════════════════════════════════
    with tab_analytics:
        st.subheader("Analytics Dashboard")

        db = get_db_connection()
        if db is None:
            st.info("No database found yet. Run the stream or register faces to generate data.")
        else:
            store = EventStore(db)
            events = store.query(limit=500)

            if not events:
                st.info("No events recorded yet. Use the Live Stream or the StreamMonitor API to generate analytics.")
            else:
                st.write(f"**{len(events)} recent events**")

                # Event type breakdown
                event_types = [e["event_type"] for e in events]
                type_counts = Counter(event_types)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total events", len(events))
                with col2:
                    st.metric("Appearances", type_counts.get("person_appeared", 0))
                with col3:
                    st.metric("Departures", type_counts.get("person_left", 0))

                st.divider()

                # Visit frequency table
                st.write("**Visit Frequency**")
                appeared_events = [e for e in events if e["event_type"] == "person_appeared"]
                if appeared_events:
                    visit_names = []
                    for ev in appeared_events:
                        data = ev.get("data") or {}
                        if "name" in data:
                            visit_names.append(data["name"])

                    if visit_names:
                        freq = Counter(visit_names)
                        import pandas as pd
                        df = pd.DataFrame(
                            sorted(freq.items(), key=lambda x: -x[1]),
                            columns=["Person", "Visits"]
                        )
                        st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.caption("No appearance events yet.")

                st.divider()

                # Attendance summary (who was seen today)
                st.write("**Today's Attendance**")
                today_start = datetime.datetime.now().replace(
                    hour=0, minute=0, second=0).timestamp()
                today_events = store.query(
                    event_type="person_appeared", since=today_start,
                    limit=10000)

                if today_events:
                    today_names = set()
                    for ev in today_events:
                        data = ev.get("data") or {}
                        if "name" in data:
                            today_names.add(data["name"])
                    if today_names:
                        st.write(", ".join(sorted(today_names)))
                        st.metric("Unique people today", len(today_names))
                    else:
                        st.caption("No identified appearances today.")
                else:
                    st.caption("No events today.")

                st.divider()

                # Crowd stats (from person_left events with dwell time)
                st.write("**Dwell Time Stats**")
                left_events = [e for e in events if e["event_type"] == "person_left"]
                if left_events:
                    dwells = []
                    for ev in left_events:
                        data = ev.get("data") or {}
                        if "dwell_seconds" in data:
                            dwells.append({
                                "name": data.get("name", "unknown"),
                                "dwell_seconds": data["dwell_seconds"]
                            })
                    if dwells:
                        import pandas as pd
                        df_dwell = pd.DataFrame(dwells)
                        avg_dwell = df_dwell.groupby("name")["dwell_seconds"].agg(
                            ["mean", "count"]).reset_index()
                        avg_dwell.columns = ["Person", "Avg Dwell (s)", "Visits"]
                        avg_dwell["Avg Dwell (s)"] = avg_dwell["Avg Dwell (s)"].round(1)
                        st.dataframe(avg_dwell, use_container_width=True, hide_index=True)
                    else:
                        st.caption("No dwell time data yet.")
                else:
                    st.caption("No dwell time data yet.")

                st.divider()

                # Raw event log
                with st.expander("Raw event log (last 50)"):
                    for ev in events[:50]:
                        ts = datetime.datetime.fromtimestamp(
                            ev["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
                        st.text(f"[{ts}] {ev['event_type']}: {json.dumps(ev.get('data') or {})}")

            db.close()

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 5: Test Console
    # ═══════════════════════════════════════════════════════════════════════
    with tab_test:
        st.subheader("Feature test console")
        st.caption("Use this as a manual QA board while you iterate on recognition, access, zones, and streaming.")

        stats = ff.stats()
        zones = st.session_state.get("zones", [])
        checks = [
            ("Gallery has enrolled faces", len(ff) > 0, f"{len(ff)} face(s), {len(ff.names())} people"),
            ("Embedding profile is persisted", bool(ff.index.get_meta("embed_profile")),
             ff.index.get_meta("embed_profile") or "missing"),
            ("SQLite backend active", stats.get("backend") == "sqlite", stats.get("backend", "npy")),
            ("Zones configured", len(zones) > 0, f"{len(zones)} zone(s)"),
            ("Secure zone configured", any(z.get("secure") for z in zones),
             "secure" if any(z.get("secure") for z in zones) else "none"),
            ("Webhook configured", bool(st.session_state.get("webhook_url")),
             st.session_state.get("webhook_url", "none")),
        ]

        import pandas as pd
        qa_rows = [{
            "Check": label,
            "Status": "PASS" if ok else "TODO",
            "Detail": detail,
        } for label, ok, detail in checks]
        st.dataframe(pd.DataFrame(qa_rows), use_container_width=True, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.write("**Fast smoke tests**")
            st.code("""pytest -q
python scripts/selftest_webcam.py
streamlit run examples/webapp_streamlit.py""", language="bash")
        with col_b:
            st.write("**Manual scenarios**")
            st.markdown("""
- Register a clear face, then identify the same photo.
- Upload a no-face image and confirm it returns `filtered: no_face`.
- Set expired consent and confirm the person no longer matches.
- Set a schedule outside current time and confirm `access_denied`.
- Add a secure zone, stand inside it with an unknown face, and watch zone counts.
- Switch to anonymous mode and confirm no names are shown.
- Switch to watchlist mode and confirm non-watchlist people are ignored.
""")

        st.divider()
        st.write("**Current stream configuration**")
        st.json({
            "db_path": DB_PATH,
            "profile": ff.profile,
            "stream_mode": st.session_state.get("stream_mode", "identify"),
            "detector": st.session_state.get("stream_det_model", "500m"),
            "det_size": st.session_state.get("stream_det_size", 480),
            "threshold": st.session_state.get("stream_threshold", 0.4),
            "zones": zones,
            "watchlist": st.session_state.get("watchlist_names", []),
        })

        st.divider()
        st.write("**Zone simulation**")
        st.caption("Draw fake face dots to test zone enter/exit, capacity, and secure-zone tailgating without a camera.")
        sim_w, sim_h = st.session_state.get("frame_size", (640, 480))
        if "zone_frame" in st.session_state:
            sim_bg = st.session_state["zone_frame"]
        else:
            sim_bg = np.zeros((sim_h, sim_w, 3), dtype=np.uint8)
            sim_bg[:] = (18, 24, 38)

        sim_points = []
        try:
            import cv2
            from PIL import Image
            from streamlit_drawable_canvas import st_canvas

            sim_preview = _draw_zones(sim_bg, zones)
            display_w = min(sim_w, 900)
            scale = display_w / max(sim_w, 1)
            display_h = int(round(sim_h * scale))
            sim_img = Image.fromarray(sim_preview).resize((display_w, display_h))
            sim_canvas = st_canvas(
                fill_color="rgba(239, 68, 68, 0.45)",
                stroke_width=2,
                stroke_color="#ef4444",
                background_image=sim_img,
                height=display_h,
                width=display_w,
                drawing_mode="circle",
                update_streamlit=True,
                key="zone_sim_canvas",
            )
            for obj in (sim_canvas.json_data or {}).get("objects", []):
                center = _canvas_object_center(obj, scale=scale)
                if center:
                    sim_points.append(center)
        except Exception as e:
            st.warning(f"Simulation canvas unavailable, use sliders: {e}")

        with st.expander("Fallback: simulate with sliders"):
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                sx = st.slider("Known face x", 0, sim_w, sim_w // 2, key="sim_known_x")
                sy = st.slider("Known face y", 0, sim_h, sim_h // 2, key="sim_known_y")
            with col_s2:
                ux = st.slider("Unknown face x", 0, sim_w, min(sim_w, sim_w // 2 + 80),
                               key="sim_unknown_x")
                uy = st.slider("Unknown face y", 0, sim_h, sim_h // 2, key="sim_unknown_y")
            if not sim_points:
                sim_points = [(sx, sy), (ux, uy)]

        from faceflash.stream import _point_in_polygon
        known = sim_points[:1]
        unknown = sim_points[1:]
        rows = []
        events = []
        for zi, z in enumerate(zones):
            poly = _polygon_for_frame(z, sim_w, sim_h)
            known_in = sum(1 for p in known if _point_in_polygon(p[0], p[1], poly))
            unknown_in = sum(1 for p in unknown if _point_in_polygon(p[0], p[1], poly))
            total = known_in + unknown_in
            rows.append({
                "Zone": z["name"],
                "Known": known_in,
                "Unknown": unknown_in,
                "Total": total,
                "Secure": bool(z.get("secure", False)),
            })
            if total:
                events.append({"event_type": "zone_occupied", "zone": z["name"], "count": total})
            if z.get("secure") and known_in and unknown_in:
                events.append({"event_type": "tailgating_risk", "zone": z["name"],
                               "authorized": known_in, "unidentified_count": unknown_in})

        cap_limit = int(st.session_state.get("capacity_limit", 0) or 0)
        cap_zone = st.session_state.get("capacity_zone", "Whole frame")
        if cap_limit:
            if cap_zone == "Whole frame":
                count = len(sim_points)
            else:
                count = next((r["Total"] for r in rows if r["Zone"] == cap_zone), 0)
            if count > cap_limit:
                events.append({"event_type": "capacity_exceeded",
                               "zone": None if cap_zone == "Whole frame" else cap_zone,
                               "count": count, "limit": cap_limit})

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Add zones in the Live Stream tab, then return here to simulate.")
        st.write("**Simulated events**")
        st.json(events or [])

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 6: Settings
    # ═══════════════════════════════════════════════════════════════════════
    with tab_settings:
        st.subheader("Settings")

        # Profile switch
        st.write("**Embedding Profile**")
        current_profile = st.session_state.get("profile", "light")
        new_profile = st.radio(
            "Model profile",
            ["light", "accurate"],
            index=0 if current_profile == "light" else 1,
            help="'light' (MobileFaceNet) is ~8x faster on CPU — best for streaming. "
                 "'accurate' (ArcFace r50) gives higher accuracy for identification.",
            key="profile_radio"
        )
        if new_profile != current_profile:
            st.warning(
                f"Switching from '{current_profile}' to '{new_profile}' requires re-enrollment "
                f"(embedding spaces are incompatible). This will clear the cached engine on next reload."
            )
            if st.button("Switch profile"):
                st.session_state["profile"] = new_profile
                get_engine.clear()
                st.rerun()

        st.divider()

        # Consent management
        st.write("**Consent Management**")
        names = ff.names()
        if names:
            consent_person = st.selectbox("Person", names, key="consent_person")
            col1, col2 = st.columns(2)
            with col1:
                consent_value = st.text_input(
                    "Consent status",
                    placeholder="e.g. granted, revoked",
                    key="consent_value"
                )
            with col2:
                consent_expires = st.date_input(
                    "Expires (optional)",
                    value=None,
                    key="consent_expires"
                )

            if st.button("Update consent", key="btn_consent"):
                if consent_value:
                    expires_str = consent_expires.isoformat() if consent_expires else None
                    ff.set_consent(consent_person, consent_value, expires=expires_str)
                    st.success(f"Consent for **{consent_person}** set to '{consent_value}'"
                               + (f" (expires {expires_str})" if expires_str else ""))
                else:
                    st.warning("Enter a consent status.")

            # Show current consent state
            if consent_person:
                current_consent = ff.index.get_person_meta(consent_person, "consent")
                current_expires = ff.index.get_person_meta(consent_person, "consent_expires")
                if current_consent:
                    st.caption(f"Current: {current_consent}"
                               + (f" (expires {current_expires})" if current_expires else ""))
        else:
            st.info("No people enrolled yet.")

        st.divider()

        # Schedule management
        st.write("**Schedule Management**")
        if names:
            sched_person = st.selectbox("Person", names, key="sched_person")
            sched_value = st.text_input(
                "Allowed hours",
                placeholder="e.g. Mon-Fri 9:00-18:00",
                help="Format: 'Mon-Fri 9:00-18:00' or 'Sat,Sun 10:00-14:00'",
                key="sched_value"
            )

            if st.button("Set schedule", key="btn_sched"):
                if sched_value:
                    try:
                        ff.set_schedule(sched_person, sched_value)
                        st.success(f"Schedule for **{sched_person}** set to '{sched_value}'")
                    except ValueError as e:
                        st.error(str(e))
                else:
                    st.warning("Enter a schedule rule.")

            # Show current schedule
            if sched_person:
                current_sched = ff.index.get_person_meta(sched_person, "schedule")
                if current_sched:
                    st.caption(f"Current schedule: {current_sched}")
        else:
            st.info("No people enrolled yet.")

        st.divider()

        # Webhook config
        st.write("**Webhook Configuration**")
        st.caption("Configure a URL to receive stream events as JSON POST requests.")
        webhook_url = st.text_input(
            "Webhook URL",
            placeholder="https://your-server.com/webhook",
            key="webhook_url"
        )
        if st.button("Save webhook URL", key="btn_webhook"):
            if webhook_url:
                st.session_state["webhook_url"] = webhook_url
                st.success(f"Webhook URL saved: {webhook_url}")
                st.caption("The StreamMonitor will POST events to this URL when running.")
            else:
                st.session_state.pop("webhook_url", None)
                st.info("Webhook URL cleared.")

        if st.session_state.get("webhook_url"):
            st.caption(f"Active webhook: {st.session_state['webhook_url']}")

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 7: Gallery
    # ═══════════════════════════════════════════════════════════════════════
    with tab_gallery:
        st.subheader("Enrolled People")

        names = ff.names()
        if not names:
            st.info("No faces enrolled yet. Use the Register tab to add people.")
        else:
            # Stats
            stats = ff.stats()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("People", len(names))
            with col2:
                st.metric("Total faces", len(ff))
            with col3:
                st.metric("Index entries", stats.get("count", len(ff)))

            st.divider()

            # Person list with remove buttons
            for person_name in sorted(names):
                face_count = sum(1 for l in ff.index.labels if l == person_name)
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"**{person_name}** ({face_count} face{'s' if face_count != 1 else ''})")
                with col2:
                    consent = ff.index.get_person_meta(person_name, "consent")
                    if consent:
                        st.caption(f"Consent: {consent}")
                with col3:
                    if st.button("Remove", key=f"rm_{person_name}"):
                        removed = ff.remove(person_name)
                        st.success(f"Removed **{person_name}** ({removed} face(s))")
                        st.rerun()

            st.divider()

            # Find duplicates
            st.write("**Duplicate Detection**")
            if st.button("Find duplicates", key="btn_dupes"):
                if len(names) < 2:
                    st.info("Need at least 2 people to check for duplicates.")
                else:
                    dupes = ff.find_duplicates(threshold=0.85)
                    if dupes:
                        st.warning(f"Found {len(dupes)} potential duplicate(s):")
                        for d in dupes:
                            st.write(
                                f"  • **{d['a']}** ↔ **{d['b']}** "
                                f"(similarity: {d['similarity']:.4f})")
                    else:
                        st.success("No duplicates found.")


if __name__ == "__main__":
    main()
