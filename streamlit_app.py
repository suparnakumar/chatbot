import base64
from io import BytesIO

import streamlit as st
from PIL import Image
from openai import OpenAI

# Show title and description.
st.title("💬 Chatbot + 🩰 Pose Evaluator")
st.write(
    "This is Suparna's chatbot that uses OpenAI to generate responses. "
    "Upload a dance pose photo and ask for an evaluation (score + coaching cues)."
)

# Ask user for their OpenAI API key
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
    st.stop()

# Create an OpenAI client.
client = OpenAI(api_key=openai_api_key)

# Session state to store chat messages (persist across reruns).
# We'll optionally store an image per user message as base64.
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---- Helpers ----
def uploaded_file_to_b64_png(uploaded_file) -> str:
    """Convert uploaded image to a base64-encoded PNG data payload."""
    img = Image.open(uploaded_file).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def build_vision_messages(history, latest_prompt: str, latest_image_b64: str | None):
    """
    Convert session history to OpenAI 'content' format.
    For prior messages: keep them as text (simple).
    For the latest user message: include image + rubric so the model evaluates the pose.
    """
    # Keep previous turns as plain text to avoid re-sending old images every time.
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[:-1]]

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

    # Latest user turn: use multimodal content if image exists
    if latest_image_b64:
        latest_content = [
            {"type": "text", "text": rubric + "\n\nUser request: " + (latest_prompt or "Please evaluate this pose.")},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{latest_image_b64}"}},
        ]
        msgs.append({"role": "user", "content": latest_content})
    else:
        # No image: just ask as normal chat (or pose-eval request without image)
        msgs.append({"role": "user", "content": (latest_prompt or "Please evaluate my pose (no image attached).")})

    return msgs

# ---- Display existing chat messages ----
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # If the message has an image, render it
        if message.get("image_b64"):
            st.image(
                base64.b64decode(message["image_b64"]),
                caption="Uploaded pose",
                use_container_width=True,
            )
        st.markdown(message["content"])

# ---- Chat-like upload + input ----
uploaded = st.file_uploader(
    "Upload a dance pose photo (optional)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    label_visibility="collapsed",
    key="pose_upload",
)

if prompt := st.chat_input("Ask anything, or upload a dance pose and ask for a rating…"):
    image_b64 = uploaded_file_to_b64_png(uploaded) if uploaded else None

    # Store and display the user's prompt (+ image if any)
    st.session_state.messages.append({"role": "user", "content": prompt, "image_b64": image_b64})
    with st.chat_message("user"):
        if image_b64:
            st.image(base64.b64decode(image_b64), caption="Uploaded pose", use_container_width=True)
        st.markdown(prompt)

    # Decide which model to use:
    # - If an image is attached, use a vision-capable model.
    # - Otherwise, you can keep gpt-3.5-turbo or upgrade.
    if image_b64:
        model = "gpt-4.1-mini"   # vision-capable (good cost/latency)
    else:
        model = "gpt-3.5-turbo"  # your original choice

    # Build messages payload
    messages_payload = build_vision_messages(
        st.session_state.messages,
        latest_prompt=prompt,
        latest_image_b64=image_b64,
    )

    # Stream the response
    with st.chat_message("assistant"):
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=messages_payload,
                stream=True,
            )
            response = st.write_stream(stream)
        except Exception as e:
            response = f"Sorry—something went wrong: `{e}`"
            st.error(response)

    # Store assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
