import streamlit as st
from aiortc.contrib.media import MediaRecorder
from pathlib import Path
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
import urllib.request
import subprocess


import urllib.request

HERE = Path(__file__).parent

# Remove audio file
def clear():
    # If file exists, delete it
    if os.path.isfile("input.wav"):
        os.remove("input.wav")


# # This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def download_file(url, download_to: Path, expected_size=None):
    subprocess.run(['pip3', 'install', '--editable', './'])

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

    


# st.write()


from infer_engish import s2t
def app():
   
  

    MODEL_URL = "https://drive.google.com/uc?export=download&confirm=no_antivirus&id=17nMz39Apg6XkpSc63W1Be74gc9Kk1Bsa"  # noqa
    # https://drive.google.com/file/d/17nMz39Apg6XkpSc63W1Be74gc9Kk1Bsa/view?usp=sharing
    MODEL_LOCAL_PATH = HERE / "english/checkpoint91.pt"
   

    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=365042194)
  

    st.write("Click start button to record audio.")
    def in_recorder_factory() -> MediaRecorder:
        return MediaRecorder(
            "input.wav", format="wav"
        )

    webrtc_streamer(
        key="loopback",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={
            "video": False,
            "audio": True,
        },
        sendback_audio=False,
        in_recorder_factory=in_recorder_factory,
    )
    try:
        st.audio('input.wav', format="audio/wav")
        if st.button('Compute'):
            output = s2t()
            st.write(output)
            clear()
        
        
    except:
        st.write(" ")
    
   


    





if __name__ == "__main__":
    app()