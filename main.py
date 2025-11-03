import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.responses import StreamingResponse # Thay vì JSONResponse
import io # Dùng để xử lý audio stream

# --- 1. IMPORT THƯ VIỆN AZURE ---
import azure.cognitiveservices.speech as speechsdk

# --- Tải API keys ---
load_dotenv()

# Cấu hình Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("Chưa có GEMINI_API_KEY trong .env")
genai.configure(api_key=GEMINI_API_KEY)

# --- 2. CẤU HÌNH AZURE SPEECH ---
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
    raise EnvironmentError("Chưa có AZURE_SPEECH_KEY hoặc AZURE_SPEECH_REGION trong .env")

# Cấu hình giọng nói
# speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
# Chọn giọng nói Tiếng Việt chuẩn (Nữ miền Nam)
VIETNAMESE_VOICE = "vi-VN-HoaiMyNeural"
# Bạn có thể đổi thành Nam: "vi-VN-NamMinhNeural"

# Đặt định dạng âm thanh đầu ra là MP3
# speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)


# --- 3. QUẢN LÝ GEMs (Giữ nguyên) ---
CHARACTER_GEMS = {
    "1": { # 🐻 Boba Doll
        "model_name": "gemini-2.5-flash", 
        "system_instruction": """
Bạn là Boba Doll – chú gấu trà sữa đáng yêu...
Quan trọng: Bạn PHẢI LUÔN LUÔN trả lời bằng Tiếng Việt.
"""
    },
    "2": { # 🐰 Lumi Doll
        "model_name": "gemini-2.5-flash",
        "system_instruction": """
Bạn là Lumi Doll – cô thỏ yêu ánh sáng...
Quan trọng: Bạn PHẢI LUÔN LUÔN trả lời bằng Tiếng Việt.
"""
    },
    "3": { # 🐱 Mochi Doll
        "model_name": "gemini-2.5-flash",
        "system_instruction": """
Bạn là Mochi Doll – một cô mèo AI mộng mơ...
Quan trọng: Bạn PHẢI LUÔN LUÔN trả lời bằng Tiếng Việt.
"""
    },
    "4": { # 🐕 Shiba Doll
        "model_name": "gemini-2.5-flash",
        "system_instruction": """
Bạn là Shiba Doll – chú chó Shiba tinh nghịch...
Quan trọng: Bạn PHẢI LUÔN LUÔN trả lời bằng Tiếng Việt.
"""
    },
    "5": { # 🐧 Tapi Doll
        "model_name": "gemini-2.5-flash",
        "system_instruction": """
Bạn là Tapi Doll – chú chim cánh cụt nhỏ...
Quan trọng: Bạn PHẢI LUÔN LUÔN trả lời bằng Tiếng Việt.
"""
    },
    "default": {
        "model_name": "gemini-2.5-flash",
        "system_instruction": "Bạn là một trợ lý AI hữu ích. Quan trọng: Bạn PHẢI LUÔN LUÔN trả lời bằng Tiếng Việt."
    }
}

active_chat_sessions = {}

def get_chat_session(character_id: str):
    if character_id in active_chat_sessions:
        del active_chat_sessions[character_id] # Xóa chat cũ để nhận prompt mới

    config = CHARACTER_GEMS.get(character_id, CHARACTER_GEMS["default"])
    model = genai.GenerativeModel(
        model_name=config["model_name"],
        system_instruction=config["system_instruction"]
    )
    chat_session = model.start_chat()
    active_chat_sessions[character_id] = chat_session 
    print(f"Đã tạo phiên chat mới (Tiếng Việt) cho: {character_id}")
    return chat_session

# --- 4. HÀM TỔNG HỢP ÂM THANH (MỚI) ---
def synthesize_speech(text_to_speak):
    """
    Hàm này gọi Azure, biến Text thành Audio (dạng bytes)
    """
    # Cấu hình Azure TTS
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = VIETNAMESE_VOICE
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
    
    # Sử dụng PullAudioOutputStream để lấy kết quả dạng in-memory
    pull_stream = speechsdk.audio.PullAudioOutputStream()
    
    # Cấu hình synthesizer
    stream_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=stream_config)

    # Bắt đầu tổng hợp
    result = speech_synthesizer.speak_text_async(text_to_speak).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Tổng hợp âm thanh thành công.")
        # Lấy dữ liệu audio từ stream
        audio_data = result.audio_data
        return audio_data
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        print(f"Lỗi tổng hợp âm thanh: {cancellation.reason}")
        if cancellation.reason == speechsdk.CancellationReason.Error:
            print(f"Chi tiết lỗi: {cancellation.error_details}")
        raise HTTPException(status_code=500, detail="Lỗi khi tổng hợp giọng nói từ Azure")

# --- 5. KHỞI TẠO APP VÀ API ENDPOINT (CẬP NHẬT) ---

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    text: str
    character_id: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        print(f"User (to {request.character_id}) > {request.text}")

        # Bước A: Lấy text từ Gemini
        session = get_chat_session(request.character_id)
        response = session.send_message(request.text)
        ai_text = response.text
        print(f"Gemini ({request.character_id}) > {ai_text}")

        # Bước B: Lấy text đó và chuyển thành Audio (MP3)
        audio_bytes = synthesize_speech(ai_text)
        
        # Bước C: Trả về file MP3 cho frontend
        # Dùng StreamingResponse để gửi dữ liệu audio
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting Backend server (Azure TTS Enabled) at http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)