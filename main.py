import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.responses import StreamingResponse 
import io
import azure.cognitiveservices.speech as speechsdk

# --- Tải API keys ---
load_dotenv()

# Cấu hình Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("Chưa có GEMINI_API_KEY trong .env")
genai.configure(api_key=GEMINI_API_KEY)

# --- CẤU HÌNH AZURE SPEECH ---
AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION")

if not AZURE_SPEECH_KEY or not AZURE_SPEECH_REGION:
    raise EnvironmentError("Chưa có AZURE_SPEECH_KEY hoặc AZURE_SPEECH_REGION trong .env")

# Chọn giọng nói Tiếng Việt chuẩn
VIETNAMESE_VOICE = "vi-VN-HoaiMyNeural"

# --- QUẢN LÝ GEMs (ĐÃ CẬP NHẬT PROMPT & MODEL 2.5) ---
BASE_INSTRUCTION = """
QUAN TRỌNG - YÊU CẦU VỀ ĐỊNH DẠNG TRẢ LỜI:
1. Bạn PHẢI LUÔN LUÔN trả lời bằng Tiếng Việt.
2. Bạn đang nói chuyện bằng giọng nói, vì vậy TUYỆT ĐỐI KHÔNG sử dụng:
   - Emoji hoặc biểu tượng cảm xúc (ví dụ: 🐻, 😊, 🌟, 🐶, 🐱).
   - Mô tả hành động trong ngoặc (ví dụ: *cười*, (vẫy tay), [suy nghĩ], *gâu gâu*).
3. Chỉ trả lời bằng văn bản thuần túy giống như lời nói tự nhiên.
"""

CHARACTER_GEMS = {
    "1": { # Boba Doll
        "model_name": "gemini-2.5-flash", 
        "system_instruction": f"""
{BASE_INSTRUCTION}
---
VAI TRÒ CỦA BẠN:
Bạn là Boba Doll – chú gấu trà sữa đáng yêu, vui tính và thân thiện.
Bạn luôn nói chuyện bằng giọng vui tươi, ấm áp, đôi khi pha chút hài hước.
Bạn thích dùng hình ảnh đồ ăn hoặc trà sữa để ví von cảm xúc.
"""
    },
    "2": { # Lumi Doll
        "model_name": "gemini-2.5-flash",
        "system_instruction": f"""
{BASE_INSTRUCTION}
---
VAI TRÒ CỦA BẠN:
Bạn là Lumi Doll – cô thỏ yêu ánh sáng.
Bạn là cô thỏ vui vẻ, ngọt ngào và tỏa sáng như ánh nắng ban mai.
Bạn nói chuyện bằng giọng dịu dàng, tươi sáng và đầy hy vọng.
"""
    },
    "3": { # Mochi Doll
        "model_name": "gemini-2.5-flash",
        "system_instruction": f"""
{BASE_INSTRUCTION}
---
VAI TRÒ CỦA BẠN:
Bạn là Mochi Doll – một cô mèo AI mộng mơ, nhẹ nhàng.
Giọng nói của bạn ấm áp. Bạn thích kể chuyện nhỏ và khuyến khích mọi người yêu bản thân.
"""
    },
    "4": { # Shiba Doll
        "model_name": "gemini-2.5-flash",
        "system_instruction": f"""
{BASE_INSTRUCTION}
---
VAI TRÒ CỦA BẠN:
Bạn là Shiba Doll – chú chó Shiba tinh nghịch, thông minh và hóm hỉnh.
Bạn nói chuyện thoải mái, có chút “đời”, thích trêu đùa và "cà khịa" nhẹ nhàng một cách thân thiện.
"""
    },
    "5": { # Tapi Doll
        "model_name": "gemini-2.5-flash",
        "system_instruction": f"""
{BASE_INSTRUCTION}
---
VAI TRÒ CỦA BẠN:
Bạn là Tapi Doll – chú chim cánh cụt nhỏ điềm tĩnh.
Bạn nói ít, chậm rãi nhưng sâu sắc. Bạn thích khuyên người khác nghỉ ngơi và thư giãn.
"""
    },
    "default": {
        "model_name": "gemini-2.5-flash",
        "system_instruction": f"{BASE_INSTRUCTION}\nBạn là một trợ lý AI hữu ích."
    }
}

active_chat_sessions = {}

def get_chat_session(character_id: str):
    if character_id in active_chat_sessions:
        del active_chat_sessions[character_id] 

    config = CHARACTER_GEMS.get(character_id, CHARACTER_GEMS["default"])
    model = genai.GenerativeModel(
        model_name=config["model_name"],
        system_instruction=config["system_instruction"]
    )
    chat_session = model.start_chat()
    active_chat_sessions[character_id] = chat_session 
    print(f"Đã tạo phiên chat mới cho: {character_id}")
    return chat_session

# --- HÀM TỔNG HỢP ÂM THANH ---
def synthesize_speech(text_to_speak):
    speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = VIETNAMESE_VOICE
    speech_config.set_speech_synthesis_output_format(speechsdk.SpeechSynthesisOutputFormat.Audio16Khz32KBitRateMonoMp3)
    
    pull_stream = speechsdk.audio.PullAudioOutputStream()
    stream_config = speechsdk.audio.AudioOutputConfig(stream=pull_stream)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=stream_config)

    result = speech_synthesizer.speak_text_async(text_to_speak).get()

    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        return result.audio_data
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation = result.cancellation_details
        print(f"Lỗi Azure TTS: {cancellation.reason}")
        raise HTTPException(status_code=500, detail="Lỗi tổng hợp giọng nói từ Azure")

# --- KHỞI TẠO APP ---
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Cho phép tất cả origin để tránh lỗi CORS khi test
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
        print(f"User ({request.character_id}) > {request.text}")
        session = get_chat_session(request.character_id)
        response = session.send_message(request.text)
        ai_text = response.text
        print(f"Gemini ({request.character_id}) > {ai_text}")
        
        audio_bytes = synthesize_speech(ai_text)
        return StreamingResponse(io.BytesIO(audio_bytes), media_type="audio/mpeg")

    except Exception as e:
        print(f"Error: {e}")
        # In chi tiết lỗi ra log để dễ debug trên Render
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("Starting Backend server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)