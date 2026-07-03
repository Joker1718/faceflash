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

st.set_page_config(page_title="FaceFlash", page_icon="⚡", layout="wide")

DB_PATH = "faceflash_demo.db"


@st.cache_resource
def get_engine(profile="light"):
    """Lazy-load FaceFlash engine (cached across reruns)."""
    from faceflash import FaceFlash
    return FaceFlash(DB_PATH, profile=profile)


def get_db_connection():
    """Direct SQLite connection for analytics queries."""
    if not Path(DB_PATH).exists():
        return None
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def main():
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
    tab_register, tab_identify, tab_live, tab_analytics, tab_settings, tab_gallery = st.tabs([
        "📷 Register", "🔍 Identify", "📹 Live Stream",
        "📊 Analytics", "⚙️ Settings", "🖼️ Gallery"
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

        col1, col2 = st.columns(2)
        with col1:
            uploaded_q = st.file_uploader("Upload query image",
                                          type=["jpg", "jpeg", "png"],
                                          key="id_upload")
        with col2:
            camera_q = st.camera_input("Or use webcam", key="id_cam")

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
            col_ctrl1, col_ctrl2 = st.columns(2)
            with col_ctrl1:
                enable_liveness = st.checkbox("Liveness check on stream", value=False,
                                              key="stream_liveness")
            with col_ctrl2:
                run_stream = st.checkbox("Start live detection", value=False,
                                          key="stream_run")

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
                    from faceflash.detect import detect_faces_boxes

                    cap = cv2.VideoCapture(0)
                    if not cap.isOpened():
                        st.error("Cannot open webcam.")
                    else:
                        frame_count = 0
                        last_detections = []  # persist between frames to avoid flashing

                        while run_stream:
                            ret, frame = cap.read()
                            if not ret:
                                break

                            frame_count += 1

                            # Process every 3rd frame for speed
                            if frame_count % 3 == 0:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                new_detections = []

                                try:
                                    dets = detect_faces_boxes(frame_rgb)
                                    for det in dets:
                                        box = det["box"]
                                        crop = det.get("crop")
                                        x1, y1, x2, y2 = [int(v) for v in box]

                                        label = "Unknown"
                                        color = (0, 0, 255)
                                        is_live = True

                                        if crop is not None:
                                            # Optional liveness
                                            if enable_liveness:
                                                try:
                                                    score, lbl = ff.liveness.check(crop)
                                                    if lbl != "live":
                                                        label = f"SPOOF ({lbl})"
                                                        color = (255, 0, 0)
                                                        is_live = False
                                                except Exception:
                                                    pass

                                            # Identify
                                            if is_live:
                                                emb = ff.embedder.embed(crop)
                                                results = ff.index.search(
                                                    emb, k=1,
                                                    n_candidates=ff.n_candidates)
                                                if results and results[0][1] >= 0.4:
                                                    label = f"{results[0][0]} ({results[0][1]:.2f})"
                                                    color = (0, 255, 0)

                                        new_detections.append((x1, y1, x2, y2, label, color))
                                except Exception:
                                    pass

                                if new_detections:
                                    last_detections = new_detections

                            # Draw last known detections on EVERY frame
                            for (x1, y1, x2, y2, label, color) in last_detections:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                                cv2.putText(frame, label, (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                          channels="RGB", use_container_width=True)

                            # Update real-time analytics sidebar
                            known = [d for d in last_detections if "Unknown" not in d[4] and "SPOOF" not in d[4]]
                            crowd_placeholder.metric("Crowd count", len(last_detections))
                            if known:
                                names_visible = [d[4].split(" (")[0] for d in known]
                                visible_placeholder.write("**Visible:** " + ", ".join(names_visible))
                            else:
                                visible_placeholder.write("**Visible:** —")

                            time.sleep(0.03)  # ~30fps display

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
            # Check if stream_events table exists
            tables = db.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='stream_events'"
            ).fetchone()

            if not tables:
                st.info("No stream events recorded yet. Use the Live Stream or the StreamMonitor API to generate analytics.")
            else:
                events = db.execute(
                    "SELECT ts, event_type, data FROM stream_events ORDER BY ts DESC LIMIT 500"
                ).fetchall()

                if not events:
                    st.info("No events recorded yet.")
                else:
                    st.write(f"**{len(events)} recent events**")

                    # Event type breakdown
                    event_types = [e[1] for e in events]
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
                    appeared_events = [e for e in events if e[1] == "person_appeared"]
                    if appeared_events:
                        visit_names = []
                        for ev in appeared_events:
                            try:
                                data = json.loads(ev[2])
                                if "name" in data:
                                    visit_names.append(data["name"])
                            except (json.JSONDecodeError, KeyError):
                                pass

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
                    today_events = db.execute(
                        "SELECT ts, event_type, data FROM stream_events "
                        "WHERE ts >= ? AND event_type = 'person_appeared' ORDER BY ts",
                        (today_start,)
                    ).fetchall()

                    if today_events:
                        today_names = set()
                        for ev in today_events:
                            try:
                                data = json.loads(ev[2])
                                if "name" in data:
                                    today_names.add(data["name"])
                            except (json.JSONDecodeError, KeyError):
                                pass
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
                    left_events = [e for e in events if e[1] == "person_left"]
                    if left_events:
                        dwells = []
                        for ev in left_events:
                            try:
                                data = json.loads(ev[2])
                                if "dwell_seconds" in data:
                                    dwells.append({
                                        "name": data.get("name", "unknown"),
                                        "dwell_seconds": data["dwell_seconds"]
                                    })
                            except (json.JSONDecodeError, KeyError):
                                pass
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

                    st.divider()

                    # Raw event log
                    with st.expander("Raw event log (last 50)"):
                        for ev in events[:50]:
                            ts = datetime.datetime.fromtimestamp(ev[0]).strftime("%Y-%m-%d %H:%M:%S")
                            st.text(f"[{ts}] {ev[1]}: {ev[2]}")

            db.close()

    # ═══════════════════════════════════════════════════════════════════════
    # Tab 5: Settings
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
    # Tab 6: Gallery
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
                            st.write(f"  • **{d[0]}** ↔ **{d[1]}** (similarity: {d[2]:.4f})")
                    else:
                        st.success("No duplicates found.")


if __name__ == "__main__":
    main()
