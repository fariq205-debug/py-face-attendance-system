from datetime import datetime
import os
import shutil
import sqlite3
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------- CREDENTIALS CONFIG ----------------------
USERS = {
    "admin": {
        "password": "admin123",
        "role": "Admin",
        "name": "System Admin",
    },
    "kiosk": {
        "password": "kiosk123",
        "role": "Operator",
        "name": "Kiosk Operator",
    },
}

CASCADE_FILE = "haarcascade_frontalface_default.xml"


@st.cache_resource
def load_face_cascade():
    cascade_cls = getattr(cv2, "CascadeClassifier")
    if os.path.exists(CASCADE_FILE):
        cascade = cascade_cls(CASCADE_FILE)
        if not cascade.empty():
            return cascade

    cv2_data = getattr(cv2, "data", None)
    if cv2_data and hasattr(cv2_data, "haarcascades"):
        sys_cascade_path = os.path.join(cv2_data.haarcascades, CASCADE_FILE)
        if os.path.exists(sys_cascade_path):
            return cascade_cls(sys_cascade_path)

    return cascade_cls(CASCADE_FILE)


# ---------------------- DATABASE SETUP ----------------------
def init_db():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            roll_no TEXT NOT NULL,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            status TEXT NOT NULL,
            check_in TEXT,
            check_out TEXT,
            UNIQUE(roll_no, date)
        )
    """)

    # Auto-migration: Existing table mein check_in aur check_out columns add karein
    cursor.execute("PRAGMA table_info(attendance)")
    cols = [col[1] for col in cursor.fetchall()]
    if "check_in" not in cols:
        cursor.execute("ALTER TABLE attendance ADD COLUMN check_in TEXT")
    if "check_out" not in cols:
        cursor.execute("ALTER TABLE attendance ADD COLUMN check_out TEXT")

    conn.commit()
    conn.close()


# ---------------------- HELPER FUNCTIONS ----------------------
def calculate_duration(check_in, check_out):
    """Check-In aur Check-Out time ke beech ka total time calculate karta hai."""
    if not check_in or not check_out or pd.isna(check_in) or pd.isna(check_out):
        return "-"
    try:
        t1 = datetime.strptime(str(check_in), "%I:%M:%S %p")
        t2 = datetime.strptime(str(check_out), "%I:%M:%S %p")
        diff = t2 - t1
        total_seconds = int(diff.total_seconds())

        if total_seconds < 0:
            return "0s"

        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        if hours > 0:
            return f"{hours}h {minutes}m"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"
    except Exception:
        return "-"


def get_dashboard_stats():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM students")
    total_students = cursor.fetchone()[0] or 0

    today_date = datetime.now().strftime("%Y-%m-%d")

    cursor.execute(
        "SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'Present'",
        (today_date,),
    )
    present_today = cursor.fetchone()[0] or 0

    cursor.execute(
        "SELECT COUNT(*) FROM attendance WHERE date = ? AND status = 'Absent'",
        (today_date,),
    )
    absent_today = cursor.fetchone()[0] or 0

    conn.close()
    return total_students, present_today, absent_today


def train_recognizer():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT roll_no, name FROM students")
    students = cursor.fetchall()
    conn.close()

    if not students:
        return None, {}, {}

    faces, ids = [], []
    roll_id_map, id_name_map = {}, {}

    for idx, (roll_no, name) in enumerate(students, start=1):
        roll_id_map[idx] = roll_no
        id_name_map[idx] = name
        student_dir = os.path.join("student_images", roll_no)

        if os.path.exists(student_dir):
            for img_name in os.listdir(student_dir):
                img_path = os.path.join(student_dir, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces.append(img)
                    ids.append(idx)

    if not faces:
        return None, {}, {}

    cv2_face = getattr(cv2, "face")
    recognizer = cv2_face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    return recognizer, roll_id_map, id_name_map


def remove_candidate(roll_no: str):
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students WHERE roll_no = ?", (roll_no,))
    cursor.execute("DELETE FROM attendance WHERE roll_no = ?", (roll_no,))
    conn.commit()
    conn.close()

    student_dir = os.path.join("student_images", roll_no)
    if os.path.exists(student_dir):
        shutil.rmtree(student_dir)


def reset_attendance_today():
    today_date = datetime.now().strftime("%Y-%m-%d")
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE date = ?", (today_date,))
    conn.commit()
    conn.close()


def reset_all_attendance():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance")
    conn.commit()
    conn.close()


# ---------------------- LOGIN SCREEN ----------------------
def render_login():
    col1, col2, col3 = st.columns([1, 1.1, 1])
    with col2:
        st.markdown(
            "## 🔒 VisionAuth Portal\n*Enterprise AI Attendance Engine*"
        )
        with st.form("login_form"):
            username_input = st.text_input(
                "Username", placeholder="e.g. admin or kiosk"
            )
            password_input = st.text_input(
                "Password", type="password", placeholder="••••••••"
            )
            submit_btn = st.form_submit_button(
                "Sign In", type="primary", use_container_width=True
            )

            if submit_btn:
                clean_username = (username_input or "").strip().lower()
                user_info = USERS.get(clean_username)

                if user_info and user_info["password"] == password_input:
                    st.session_state.logged_in = True
                    st.session_state.username = clean_username
                    st.session_state.user_role = user_info["role"]
                    st.session_state.display_name = user_info["name"]
                    st.rerun()
                else:
                    st.error("Invalid Username or Password")


# ---------------------- TAB RENDERERS ----------------------
def render_registration_tab(face_cascade):
    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("#### ➕ Register New Candidate")
        roll_no = st.text_input("Roll No", placeholder="e.g. REG-2026-88")
        name = st.text_input("Full Name", placeholder="e.g. John Doe")

        start_capture = st.button(
            "🔴 Begin Facial Sampling",
            use_container_width=True,
            type="primary",
        )

    with col2:
        st.markdown("#### 📋 Registered Database Directory")
        conn = sqlite3.connect("attendance.db")
        df_students = pd.read_sql_query(
            "SELECT roll_no AS 'Roll No', name AS 'Name' FROM students",
            conn,
        )
        conn.close()
        st.dataframe(df_students, use_container_width=True, height=220)

        st.divider()
        st.markdown("#### 🗑️ Remove Registered Candidate")

        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT roll_no, name FROM students")
        student_list = cursor.fetchall()
        conn.close()

        if student_list:
            options = {f"{r[0]} - {r[1]}": r[0] for r in student_list}
            selected_option = st.selectbox(
                "Select Candidate to Remove", list(options.keys())
            )
            selected_roll = options[selected_option]

            if st.button("🗑️ Delete Selected Candidate", use_container_width=True):
                remove_candidate(selected_roll)
                st.success(f"Candidate **{selected_roll}** removed successfully!")
                st.rerun()
        else:
            st.info("No candidate registered to remove.")

    if start_capture:
        clean_roll = (roll_no or "").strip()
        clean_name = (name or "").strip()

        if not clean_roll or not clean_name:
            st.warning("⚠️ Roll No and Name are required.")
        else:
            conn = sqlite3.connect("attendance.db")
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM students WHERE roll_no = ?", (clean_roll,)
            )
            exists = cursor.fetchone()
            conn.close()

            if exists:
                st.error("❌ Candidate Roll Number already exists.")
            else:
                student_dir = os.path.join("student_images", clean_roll)
                if not os.path.exists(student_dir):
                    os.makedirs(student_dir)

                cap = cv2.VideoCapture(0)
                frame_placeholder = st.empty()
                count = 0

                while cap.isOpened() and count < 20:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.3, minNeighbors=5
                    )

                    for x, y, w, h in faces:
                        count += 1
                        face_img = gray[y : y + h, x : x + w]
                        cv2.imwrite(
                            os.path.join(student_dir, f"{count}.jpg"),
                            cv2.resize(face_img, (200, 200)),
                        )
                        cv2.rectangle(
                            frame, (x, y), (x + w, y + h), (59, 130, 246), 2
                        )

                    frame_placeholder.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        caption=f"Sampling... ({count}/20)",
                    )

                cap.release()
                frame_placeholder.empty()

                if count >= 20:
                    conn = sqlite3.connect("attendance.db")
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO students (roll_no, name) VALUES (?, ?)",
                        (clean_roll, clean_name),
                    )
                    conn.commit()
                    conn.close()
                    st.success(f"✅ Registered **{clean_name}** successfully.")
                    st.rerun()


def render_scanner_tab(face_cascade):
    today_date = datetime.now().strftime("%Y-%m-%d")
    col_att1, col_att2 = st.columns([1.3, 1])

    with col_att1:
        run_scanner = st.checkbox("🟢 Activate Optical Scanner Feed")
        camera_placeholder = st.empty()

    with col_att2:
        st.markdown("#### Live Today's Attendance Logs")
        conn = sqlite3.connect("attendance.db")
        df_today = pd.read_sql_query(
            """
            SELECT s.roll_no AS 'Roll No', s.name AS 'Name', 
                   COALESCE(a.status, 'Not Marked') AS 'Status',
                   a.check_in AS 'Check In',
                   a.check_out AS 'Check Out'
            FROM students s
            LEFT JOIN attendance a ON s.roll_no = a.roll_no AND a.date = ?
        """,
            conn,
            params=[today_date],
        )
        conn.close()

        # Calculate Duration
        if not df_today.empty:
            df_today["Duration"] = df_today.apply(
                lambda r: calculate_duration(r["Check In"], r["Check Out"]),
                axis=1,
            )

        st.dataframe(df_today, use_container_width=True, height=280)

    if run_scanner:
        recognizer, roll_id_map, id_name_map = train_recognizer()

        if recognizer is None:
            st.warning("⚠️ No enrolled candidates found in database.")
        else:
            cap = cv2.VideoCapture(0)

            while run_scanner and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray, scaleFactor=1.2, minNeighbors=5
                )

                for x, y, w, h in faces:
                    face_roi = cv2.resize(gray[y : y + h, x : x + w], (200, 200))
                    label_id, confidence = recognizer.predict(face_roi)

                    if confidence < 85:
                        name_str = id_name_map.get(label_id, "Unknown")
                        roll_str = roll_id_map.get(label_id, "")
                        now_time = datetime.now().strftime("%I:%M:%S %p")

                        color = (34, 197, 94)
                        label_text = f"{name_str} ({roll_str})"

                        # CHECK-IN & CHECK-OUT LOGIC
                        conn = sqlite3.connect("attendance.db")
                        cursor = conn.cursor()

                        cursor.execute(
                            "SELECT check_in FROM attendance WHERE roll_no = ? AND date = ?",
                            (roll_str, today_date),
                        )
                        existing_rec = cursor.fetchone()

                        if existing_rec is None:
                            # Pehli baar scan hua -> Record Check-In
                            cursor.execute(
                                """
                                INSERT INTO attendance (roll_no, name, date, status, check_in, check_out)
                                VALUES (?, ?, ?, 'Present', ?, ?)
                            """,
                                (
                                    roll_str,
                                    name_str,
                                    today_date,
                                    now_time,
                                    now_time,
                                ),
                            )
                        else:
                            # Baad mein scan hua -> Update Check-Out Time
                            cursor.execute(
                                """
                                UPDATE attendance 
                                SET check_out = ? 
                                WHERE roll_no = ? AND date = ?
                            """,
                                (now_time, roll_str, today_date),
                            )

                        conn.commit()
                        conn.close()
                    else:
                        color = (239, 68, 68)
                        label_text = "Unverified Identity"

                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(
                        frame,
                        label_text,
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )

                camera_placeholder.image(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB"
                )

            cap.release()
            camera_placeholder.empty()


def render_logs_tab(is_admin: bool):
    st.markdown("#### 📊 Historical Attendance Logs")
    conn = sqlite3.connect("attendance.db")
    df_all = pd.read_sql_query(
        """
        SELECT roll_no AS 'Roll No', 
               name AS 'Name', 
               date AS 'Date', 
               status AS 'Status',
               check_in AS 'Check In',
               check_out AS 'Check Out'
        FROM attendance 
        ORDER BY date DESC, id DESC
    """,
        conn,
    )
    conn.close()

    if not df_all.empty:
        df_all["Duration"] = df_all.apply(
            lambda r: calculate_duration(r["Check In"], r["Check Out"]), axis=1
        )

    st.dataframe(df_all, use_container_width=True, height=280)

    if is_admin:
        st.divider()
        st.markdown("#### 🔄 Reset Operations")
        r_col1, r_col2 = st.columns(2)

        with r_col1:
            if st.button("🧹 Reset Today's Attendance", use_container_width=True):
                reset_attendance_today()
                st.success("Today's attendance records cleared!")
                st.rerun()

        with r_col2:
            if st.button("⚠️ Clear ALL Attendance Logs", use_container_width=True):
                reset_all_attendance()
                st.success("All attendance records cleared!")
                st.rerun()


# ---------------------- MAIN APPLICATION ----------------------
def main():
    st.set_page_config(
        page_title="VisionAuth | Next-Gen AI Portal", page_icon="⚡", layout="wide"
    )

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    init_db()

    if not st.session_state.logged_in:
        render_login()
        return

    if not os.path.exists("student_images"):
        os.makedirs("student_images")

    face_cascade = load_face_cascade()

    # Sidebar Options
    with st.sidebar:
        st.markdown("### 👤 Session Info")
        st.write(f"**User:** {st.session_state.display_name}")
        st.write(f"**Role:** `{st.session_state.user_role}`")
        st.divider()

        if st.button("🔒 Sign Out", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    # Top Header
    st.title("⚡ VisionAuth Executive Dashboard")
    st.caption("Automated Computer Vision Attendance Engine")

    tot_students, pres_today, abs_today = get_dashboard_stats()
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Enrolled", tot_students)
    m2.metric("Present Today", pres_today)
    m3.metric("Absent Today", abs_today)

    st.divider()

    is_admin = st.session_state.user_role == "Admin"

    if is_admin:
        tab1, tab2, tab3 = st.tabs(
            ["👤 Candidate Registration", "🎥 Live Scanner Feed", "📊 Attendance Logs"]
        )
        with tab1:
            render_registration_tab(face_cascade)
        with tab2:
            render_scanner_tab(face_cascade)
        with tab3:
            render_logs_tab(is_admin)
    else:
        tab2, tab3 = st.tabs(["🎥 Live Scanner Feed", "📊 Attendance Logs"])
        with tab2:
            render_scanner_tab(face_cascade)
        with tab3:
            render_logs_tab(is_admin)


if __name__ == "__main__":
    main()