import base64
from io import BytesIO

import streamlit as st
from PIL import Image
from openai import OpenAI

# -----------------------------
# Page + API key
# -----------------------------
st.set_page_config(page_title="Chatbot + PoseMirror", page_icon="🩰", layout="centered")
st.title("💬 Chatbot + 🩰 PoseMirror")

openai_api_key = st.secrets['OPENAI_API_KEY']

client = OpenAI(api_key=openai_api_key)

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    # Each message can optionally contain image_b64
    st.session_state.messages = []
if "pose_notes" not in st.session_state:
    st.session_state.pose_notes = ""

# -----------------------------
# Helpers
# -----------------------------
def uploaded_file_to_b64_png(uploaded_file) -> str:
    img = Image.open(uploaded_file).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def build_pose_eval_messages(user_notes: str, image_b64: str):
    rubric = """
You are a dance technique coach. Evaluate the dancer's pose from the image and the user's request.

Return:
1) Overall score (0-10)
2) 3 strengths (bullets)
3) 3 improvements (bullets) - specific, actionable cues
4) Safety note (1 sentence) if any risky alignment is visible
5) A 10-second drill to improve the #1 issue

Be encouraging but precise. If the image is missing/unclear, ask for a clearer full-body photo and explain what you can't see.
Do NOT identify the person.
"""
    notes = user_notes.strip() if user_notes else "No extra notes."
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{rubric}\n\nUser notes/context: {notes}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }
    ]

def run_pose_eval(image_b64: str, user_notes: str) -> str:
    # Choose a vision-capable model (swap to gpt-5-mini ONLY if your account supports vision on it).
    model = "gpt-4.1-mini"
    resp = client.chat.completions.create(
        model=model,
        messages=build_pose_eval_messages(user_notes, image_b64),
    )
    return resp.choices[0].message.content

# -----------------------------
# 1) PoseMirror input area (posts into chat)
# -----------------------------
st.subheader("🩰 PoseMirror")

c1, c2 = st.columns([1, 1])

with c1:
    pose_file = st.file_uploader(
        "Upload a dance pose photo",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
        key="pose_upload",
        label_visibility="collapsed",
    )

with c2:
    st.session_state.pose_notes = st.text_area(
        "What should I focus on? (optional)",
        value=st.session_state.pose_notes,
        placeholder="e.g., arabesque line, turnout, torso lift, arm placement, balance",
        height=110,
    )

eval_clicked = st.button("Evaluate Pose → post to chat", type="primary", use_container_width=True)

# If user clicks, add to chat + generate assistant reply
if eval_clicked:
    if not pose_file:
        st.warning("Please upload a photo first.")
    else:
        image_b64 = uploaded_file_to_b64_png(pose_file)

        # Add a user chat message that includes the image
        user_text = "Evaluate this pose."
        if st.session_state.pose_notes.strip():
            user_text += f"\n\n**Focus:** {st.session_state.pose_notes.strip()}"

        st.session_state.messages.append(
            {"role": "user", "content": user_text, "image_b64": image_b64}
        )

        # Generate assistant response
        with st.spinner("Evaluating pose…"):
            try:
                result = run_pose_eval(image_b64, st.session_state.pose_notes)
            except Exception as e:
                result = f"Sorry—something went wrong while evaluating the pose: `{e}`"

        st.session_state.messages.append({"role": "assistant", "content": result})

        # Clear the uploader + notes if you want a “new eval” feel
        st.session_state.pose_notes = ""
        st.rerun()

st.divider()

# -----------------------------
# 2) Chat thread (shows both normal chat + pose eval posts)
# -----------------------------
st.subheader("💬 Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message.get("image_b64"):
            st.image(
                base64.b64decode(message["image_b64"]),
                caption="Uploaded pose",
                use_container_width=True,
            )
        st.markdown(message["content"])

# Normal text chat input
if prompt := st.chat_input("Ask anything… (pose evals appear here too)"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            # Keep text chat as-is; you can upgrade to gpt-5-mini (text) if you want
            stream = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                stream=True,
            )
            response = st.write_stream(stream)
        except Exception as e:
            response = f"Sorry—something went wrong: `{e}`"
            st.error(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
