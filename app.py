import streamlit as st
import pandas as pd
import joblib
import os
import sqlite3
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from auth import (
    init_db,
    register_user,
    authenticate_user,
    is_valid_identifier,
    is_valid_reg_number,
    is_valid_email,
    get_user_email_by_identifier,
    update_password_by_email
)
from otp_manager import init_otp_table, generate_otp, verify_otp
from email_utils import send_otp_email

init_db()
init_otp_table()

st.set_page_config(
    page_title="DeKUT Mental Health Early Detector",
    page_icon="🎓",
    layout="centered"
)

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'user_id' not in st.session_state:
    st.session_state.user_id = None
if 'reset_stage' not in st.session_state:
    st.session_state.reset_stage = None
if 'reset_email' not in st.session_state:
    st.session_state.reset_email = None

DB_PATH = os.path.join(os.path.dirname(__file__), "mental_health.db")

def save_assessment(user_id, data, score, mood_comment=""):
    conn = sqlite3.connect(DB_PATH)
    level = "High" if score >= 0.4 else "Low"
    conn.execute('''
        INSERT INTO assessments (user_id, timestamp, sleep, activity, social, stress, academics, mood_comment, risk_score, risk_level)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, datetime.now().isoformat(),
        data['sleep'], data['activity'], data['social'],
        data['stress'], data['academics'], mood_comment,
        score, level
    ))
    conn.commit()
    conn.close()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    model = joblib.load(os.path.join(BASE_DIR, 'best_linear_regression_model.pkl'))
    scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))
except Exception as e:
    st.error(f"❌ Model loading error: {e}")
    st.stop()

DKUT_SUPPORT_EMAIL = "deanofstudents@dkut.ac.ke"
DKUT_MOBILE = "0708 680 879"
DKUT_EXTENSION = "1262"
DKUT_CONTACT_LINE = f"📧 {DKUT_SUPPORT_EMAIL} | 📞 {DKUT_MOBILE} (Ext. {DKUT_EXTENSION})"

if not st.session_state.logged_in:
    st.title("🏛️ DeKUT Student Mental Health Portal")

    if st.session_state.reset_stage == 'initiate':
        st.subheader("🔑 Forgot Password")
        identifier = st.text_input("🆔 Registration Number or 📧 Email", key="fp_input").strip()
        if st.button("Send OTP"):
            if not identifier:
                st.error("❌ Please enter your registration number or email.")
            elif not is_valid_identifier(identifier):
                st.error("❌ Invalid format.")
            else:
                email = get_user_email_by_identifier(identifier)
                if email:
                    st.session_state.reset_email = email
                    otp = generate_otp(email)
                    if send_otp_email(email, otp):
                        st.success("✅ OTP sent!")
                    else:
                        st.error("❌ Failed to send OTP.")
                else:
                    st.success("✅ If account exists, OTP sent.")
                st.session_state.reset_stage = 'verify'
                st.rerun()
        if st.button("← Back to Login"):
            st.session_state.reset_stage = None
            st.rerun()

    elif st.session_state.reset_stage == 'verify':
        st.subheader("🔐 Enter OTP & Set New Password")
        email = st.session_state.get('reset_email', '')
        if not email:
            st.error("❌ Session expired.")
            if st.button("Restart"):
                st.session_state.reset_stage = 'initiate'
                st.rerun()
            st.stop()
        st.info(f"Code sent to **{email}**")
        otp = st.text_input("OTP", max_chars=6).strip()
        new_password = st.text_input("🔒 New Password", type="password", key="new_pass")
        confirm_password = st.text_input("🔒 Confirm Password", type="password", key="confirm_pass")
        if st.button("Reset Password"):
            if len(otp) != 6 or not otp.isdigit():
                st.error("❌ OTP must be 6-digit.")
            elif len(new_password) < 6:
                st.error("❌ Password ≥6 chars.")
            elif new_password != confirm_password:
                st.error("❌ Passwords don't match.")
            else:
                if verify_otp(email, otp):
                    update_password_by_email(email, new_password)
                    st.session_state.reset_stage = 'success'
                    if 'reset_email' in st.session_state:
                        del st.session_state['reset_email']
                    st.rerun()
                else:
                    st.error("🚫 Invalid/expired OTP.")
        if st.button("← Back to Login"):
            if 'reset_email' in st.session_state:
                del st.session_state['reset_email']
            st.session_state.reset_stage = None
            st.rerun()

    elif st.session_state.reset_stage == 'success':
        st.success("✅ Password reset! Log in with new password.")
        if st.button("Go to Login"):
            st.session_state.reset_stage = None
            st.rerun()

    else:
        auth_mode = st.radio("Choose an option", ["🔐 Login", "📝 Sign Up"], horizontal=True)
        is_signup = "Sign Up" in auth_mode
        with st.form("auth_form"):
            if is_signup:
                reg_number = st.text_input("🆔 Registration Number", placeholder="e.g., X123-45-6789/2024")
                email = st.text_input("📧 Student Email", placeholder="e.g., surname.name22@students.dkut.ac.ke")
            else:
                identifier = st.text_input("🆔 Reg Number or 📧 Email", placeholder="e.g., X123-45-6789/2024 or surname.name22@students.dkut.ac.ke")
            password = st.text_input("🔒 Password", type="password")
            if is_signup:
                confirm_password = st.text_input("🔒 Confirm Password", type="password")
            submitted = st.form_submit_button("Submit")
        if submitted:
            if is_signup:
                if not reg_number or not email or not password:
                    st.error("❌ Fill all fields.")
                elif not is_valid_reg_number(reg_number):
                    st.error("❌ Invalid reg number format.")
                elif not is_valid_email(email):
                    st.error("❌ Use format: surname.name22@students.dkut.ac.ke")
                elif password != confirm_password:
                    st.error("❌ Passwords don't match.")
                else:
                    user_id, message = register_user(reg_number, email, password)
                    if user_id:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.success("✅ Account created! Logged in.")
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")
            else:
                if not identifier or not password:
                    st.error("❌ Fill all fields.")
                elif not is_valid_identifier(identifier):
                    st.error("❌ Invalid format.")
                else:
                    user_id = authenticate_user(identifier, password)
                    if user_id:
                        st.session_state.logged_in = True
                        st.session_state.user_id = user_id
                        st.success("✅ Login successful!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials.")
        if not is_signup and st.button("🤔❓ Forgot Password?"):
            st.session_state.reset_stage = 'initiate'
            st.rerun()

    st.stop()

st.title("👨‍⚕️ DeKUT Mental Health Early Detector")

st.caption("ℹ️ This tool simulates a multimodal system:\n"
             "• 📱 **Wearables**: Sleep & activity\n"
             "• 💬 **Social**: Interaction level\n"
             "• 📝 **Surveys**: Stress + mood comments\n"
             "• 🎓 **Academics**: Performance %\n\n"
             "⚠️ **Not a diagnosis.** For support, contact the Dean of Students:\n"
             f"{DKUT_CONTACT_LINE}")

if st.button("➡️ Logout"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

col1, col2 = st.columns(2)
with col1:
    sleep = st.number_input(
        "😴 Sleep Hours/Night",
        min_value=2.0,
        max_value=12.0,
        value=None,
        help="Typical range: 2–12 hours"
    )
    social = st.number_input(
        "👥 Social (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=None,
        help="0 = isolated, 10 = very social"
    )
    academics = st.number_input(
        "🎓 Academics (%)",
        min_value=0.0,
        max_value=100.0,
        value=None,
        help="Current semester performance"
    )
with col2:
    activity = st.number_input(
        "🏃 Activity (hrs/week)",
        min_value=0.0,
        max_value=20.0,
        value=None,
        help="Walking, gym, sports (typical: 0–15 hrs)"
    )
    stress = st.number_input(
        "😰 Stress (0-10)",
        min_value=0.0,
        max_value=10.0,
        value=None,
        help="0 = relaxed, 10 = overwhelmed"
    )

st.markdown("### 💬 How have you been feeling lately?")
mood_comment = st.text_area(
    "Share anything on your mind (optional):",
    placeholder="e.g., 'Overwhelmed with exams', 'Feeling lonely', 'Happy and energized!'",
    height=100,
    label_visibility="collapsed"
)

consent = st.checkbox("✅ I consent to store anonymized data securely.")

inputs = [sleep, activity, social, stress, academics]
if st.button("✨ Check My Well-being"):
    if not consent:
        st.warning("🔒 Please consent to proceed.")
    elif any(x is None for x in inputs):
        st.error("❌ Please fill all fields.")
    else:
        user_data = {
            'sleep': sleep,
            'activity': activity,
            'social': social,
            'stress': stress,
            'academics': academics
        }
        X = pd.DataFrame([list(user_data.values())], columns=user_data.keys())
        X_scaled = scaler.transform(X)
        pred_score = model.predict(X_scaled)[0]
        is_high_risk = pred_score >= 0.4

        if st.session_state.user_id is not None:
            save_assessment(st.session_state.user_id, user_data, pred_score, mood_comment)
        else:
            st.warning("⚠️ Log in to save history.")

        if is_high_risk:
            st.error(f"⚠️ High Risk Detected! (Score: {pred_score:.2f})")
        else:
            st.balloons()
            st.success(f"✅ Low Risk! (Score: {pred_score:.2f})")

        st.subheader("🩺 Personalized Recommendations")
        recs = []
        if sleep < 6:
            recs.append("💤 Sleep: Aim for 7–8 hours.")
        if sleep < 4:
            recs.append("🚨 Severe sleep deprivation: Contact Dean of Students immediately.")
        if activity < 3:
            recs.append("🏃 Activity: Start with 15-min walks daily.")
        if social < 4:
            recs.append("👥 Social: Reach out to a friend or student group.")
        if stress > 7:
            recs.append(f"😰 Stress: Contact Dean of Students — {DKUT_MOBILE} (Ext. {DKUT_EXTENSION}).")
        if academics < 65:
            recs.append("🎓 Academics: Visit Academic Support Centre.")
        if not recs and not is_high_risk:
            recs.append("🎉 Great job! Keep it up.")
        if not recs and is_high_risk:
            recs.append(f"⚠️ Please contact Dean of Students now: {DKUT_CONTACT_LINE}")

        for r in recs:
            st.markdown(
                f'<div style="background:#1a202c; color:white; padding:12px; border-radius:8px; border-left:3px solid #f6ad55;">• {r}</div>',
                unsafe_allow_html=True
            )

        st.subheader("💡 What influenced your score?")
        coef = model.coef_
        features = ['Sleep', 'Activity', 'Social', 'Stress', 'Academics']
        user_vals = [sleep, activity, social, stress, academics]
        max_abs = max(abs(c) for c in coef) or 1
        influence = [c / max_abs for c in coef]

        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ['#10b981' if x <= 0 else '#ef4444' for x in influence]
        y_pos = range(len(features))
        ax.barh(y_pos, influence, color=colors, height=0.6)
        ax.axvline(0, color='#4b5563', linestyle='-')
        for i, (feat, val) in enumerate(zip(features, user_vals)):
            unit = 'h' if feat in ['Sleep', 'Activity'] else '%' if feat == 'Academics' else ''
            label = f"{feat}: {val}{unit}"
            ha = 'left' if influence[i] >= 0 else 'right'
            x_pos = 0.05 if influence[i] >= 0 else -0.05
            ax.text(x_pos, i, label, va='center', ha=ha, color='white', weight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels([''] * len(features))
        ax.set_xlim(-1.3, 1.3)
        ax.set_facecolor('#0f172a')
        fig.patch.set_facecolor('#0f172a')
        st.pyplot(fig)
        plt.close()

        st.info("🔍 **Red** = increases risk | **Green** = reduces risk")

st.markdown("<br><hr>", unsafe_allow_html=True)

if st.session_state.user_id is not None:
    with st.expander("📊 My Assessment History & Trends"):
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql_query("""
            SELECT datetime(timestamp) as "Date", sleep, activity, social, stress, academics,
                   mood_comment, printf("%.2f", risk_score) as "Risk_Score", risk_level as "Level"
            FROM assessments WHERE user_id = ? ORDER BY timestamp ASC
        """, conn, params=(st.session_state.user_id,))
        conn.close()

        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            display_df = df[[
                'Date', 'sleep', 'activity', 'social', 'stress', 'academics', 'Risk_Score', 'Level'
            ]].copy()
            st.dataframe(display_df, hide_index=True, use_container_width=True)

            st.subheader("💭 Your Mood Comments")
            for _, row in df.iterrows():
                comment = str(row['mood_comment']).strip() if row['mood_comment'] else ""
                if comment:
                    st.markdown(f"**{row['Date'].strftime('%Y-%m-%d %H:%M')}**: {comment}")
                else:
                    st.markdown(f"**{row['Date'].strftime('%Y-%m-%d %H:%M')}**: _No comment provided_")

            st.subheader("📈 Risk & Behavior Trends")
            fig, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(df['Date'], df['Risk_Score'].astype(float), color='#f56565', marker='o', linewidth=2, label='Risk Score')
            ax1.axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='High Risk Threshold')
            ax1.set_ylabel('Risk Score', color='#f56565')
            ax1.tick_params(axis='y', labelcolor='#f56565')

            ax2 = ax1.twinx()
            ax2.plot(df['Date'], df['sleep'], color='#4299e1', alpha=0.7, label='Sleep')
            ax2.plot(df['Date'], df['activity'], color='#ed8936', alpha=0.7, label='Activity')
            ax2.plot(df['Date'], df['social'], color='#9f7aea', alpha=0.7, label='Social')
            ax2.plot(df['Date'], df['stress'], color='#f56565', alpha=0.7, label='Stress')
            ax2.plot(df['Date'], df['academics'], color='#48bb78', alpha=0.7, label='Academics')
            ax2.set_ylabel('Behavior Scores', color='white')
            ax2.tick_params(axis='y', labelcolor='white')

            ax1.set_xlabel('Date')
            fig.tight_layout()
            ax1.legend(loc='upper left', bbox_to_anchor=(0, 0.8))
            ax2.legend(loc='upper right', bbox_to_anchor=(1, 0.8))
            st.pyplot(fig)
            plt.close()

            latest = df.iloc[-1]
            earliest = df.iloc[0]
            risk_change = float(latest['Risk_Score']) - float(earliest['Risk_Score'])
            if risk_change < -0.1:
                st.success(f"🌟 Risk improved by {-risk_change:.2f}!")
            elif risk_change > 0.1:
                st.warning(f"⚠️ Risk increased by {risk_change:.2f}.")
            else:
                st.info("🔄 Risk stable.")
        else:
            st.info("📭 No assessments yet.")

    with st.expander("🛡️ Privacy Controls"):
        st.write("Your data is stored securely and only accessible to you.")
        if st.button("🗑️ Delete All History"):
            conn = sqlite3.connect(DB_PATH)
            conn.execute("DELETE FROM assessments WHERE user_id = ?", (st.session_state.user_id,))
            conn.commit()
            conn.close()
            st.success("✅ History deleted.")
            st.rerun()
else:
    st.info("Log in to view history and trends.")