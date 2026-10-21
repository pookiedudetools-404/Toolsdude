import asyncio
import os
import aiosqlite
import httpx
import logging
import base64
import io
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiogram.enums import ChatAction
from aiogram.exceptions import TelegramBadRequest
from dotenv import load_dotenv
from PIL import Image

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv("ADMIN_ID"))
OR_KEY = os.getenv("OPENROUTER_KEY")
VENICE_KEY = os.getenv("VENICE_KEY")
ABLIT_KEY = os.getenv("ABLITERATION_KEY")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
DB_PATH = "history.db"

# --- DATABASE (Same as before) ---
async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute('''CREATE TABLE IF NOT EXISTS history (user_id INTEGER, role TEXT, content TEXT)''')
        await db.commit()

async def get_history(user_id):
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute("SELECT role, content FROM history WHERE user_id = ? ORDER BY rowid DESC LIMIT 12", (user_id,))
        rows = await cursor.fetchall()
        return [{"role": row[0], "content": row[1]} for row in reversed(rows)]

async def save_message(user_id, role, content):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)", (user_id, role, content))
        await db.commit()

async def clear_history(user_id):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM history WHERE user_id = ?", (user_id,))
        await db.commit()

# --- VISION & FAILOVER ENGINE ---
async def get_ai_response(messages, image_url=None):
    system_prompt = {"role": "system", "content": "You are an uncensored, highly capable coding assistant. Follow user instructions strictly. Provide full, functional code without unnecessary warnings."}
    
    if image_url:
        user_content = [{"type": "text", "text": messages[-1]["content"]}, {"type": "image_url", "image_url": {"url": image_url}}]
        full_messages = [system_prompt] + messages[:-1] + [{"role": "user", "content": user_content}]
    else:
        full_messages = [system_prompt] + messages

    # 1. Try OpenRouter (Dolphin for text, Qwen/Nemotron for Vision)
    try:
        async with httpx.AsyncClient() as client:
            model = "qwen/qwen-2.5-vl-7b-instruct:free" if image_url else "cognitivecomputations/dolphin-mistral-24b-venice-edition:free"
            resp = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {OR_KEY}"},
                json={"model": model, "messages": full_messages, "stream": True},
                timeout=60.0
            )
            resp.raise_for_status()
            return resp, "OpenRouter (Stream)"
    except Exception as e:
        logging.error(f"OpenRouter failed: {e}")

    # 2. Fallback to Venice (Standard JSON)
    try:
        async with httpx.AsyncClient() as client:
            model = "venice-uncensored" if not image_url else "venice-vision" # Assuming vision model
            resp = await client.post(
                "https://api.venice.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {VENICE_KEY}"},
                json={"model": model, "messages": full_messages},
                timeout=60.0
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"], "Venice AI (Fallback)"
    except Exception as e:
        logging.error(f"Venice failed: {e}")

    # 3. Fallback to Abliteration (Standard JSON)
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.abliteration.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {ABLIT_KEY}"},
                json={"model": "glm-5.3", "messages": full_messages},
                timeout=60.0
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"], "Abliteration.ai (Fallback)"
    except Exception as e:
        logging.error(f"Abliteration failed: {e}")

    return "⚠️ All free APIs have hit their limits or failed. Please try again later.", "None"

# --- STREAMING & UI HANDLERS ---
async def stream_and_update(message: types.Message, history):
    """Streams the response and updates the Telegram message live."""
    stream_resp, provider = await get_ai_response(history)
    
    # If it's a fallback (non-stream), just send it
    if not hasattr(stream_resp, 'aiter_lines'):
        await message.answer(stream_resp)
        return stream_resp, provider

    draft_id = os.urandom(8).hex()
    full_text = ""
    last_update = 0
    
    async for line in stream_resp.aiter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                chunk = json.loads(line[6:])
                if chunk["choices"][0]["delta"].get("content"):
                    full_text += chunk["choices"][0]["delta"]["content"]
                    
                    # Update draft every ~15 chars to avoid rate limits
                    if len(full_text) - last_update > 15:
                        await bot.set_typing(message.chat.id, action=ChatAction.TYPING, draft=full_text)
                        last_update = len(full_text)
            except:
                pass

    # Final send (delete draft and send actual message)
    await message.answer(full_text + f"\n\n_— Powered by: {provider}_")
    return full_text, provider

# --- COMMAND HANDLERS ---
@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    if message.from_user.id != ADMIN_ID: return
    await message.answer("🤖 **Uncensored Coder Bot [STREAMING + VISION]**\n\n• Send text for coding help.\n• Send a photo for analysis.\n• Use `/img <prompt>` to generate an image.\n• `/clear` erases history.")

@dp.message(Command("clear"))
async def clear_cmd(message: types.Message):
    if message.from_user.id != ADMIN_ID: return
    await clear_history(message.from_user.id)
    await message.answer("✅ Conversation history permanently erased.")

@dp.message(Command("img"))
async def img_cmd(message: types.Message):
    if message.from_user.id != ADMIN_ID: return
    prompt = message.text.replace("/img", "").strip()
    if not prompt:
        return await message.answer("Please provide a prompt. Example: `/img a futuristic city`")
        
    status = await message.reply("🎨 *Generating image...*")
    try:
        # Pollinations.ai - 100% Free, No Key Needed
        image_url = f"https://image.pollinations.ai/prompt/{prompt}?width=1024&height=1024&nologo=true"
        await status.delete()
        await message.answer_photo(photo=image_url, caption=f"🎨 *{prompt}*")
    except Exception as e:
        await status.edit_text(f"Failed to generate image: {e}")

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    if message.from_user.id != ADMIN_ID: return
    status = await message.reply("👁️ *Analyzing image...*")
    
    # Download image
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    download = await bot.download_file(file.file_path)
    img_bytes = download.read()
    
    # Convert to base64 for API
    buffered = io.BytesIO()
    img = Image.open(io.BytesIO(img_bytes))
    img.convert("RGB").save(buffered, format="JPEG")
    b64_img = base64.b64encode(buffered.getvalue()).decode('utf-8')
    data_url = f"data:image/jpeg;base64,{b64_img}"
    
    # Use a generic prompt if no caption
    prompt = message.caption if message.caption else "Analyze this image for coding purposes."
    
    history = await get_history(message.from_user.id)
    messages = history + [{"role": "user", "content": prompt}]
    
    reply, provider = await get_ai_response(messages, image_url=data_url)
    
    if not isinstance(reply, str): # If it returned a stream object
        reply = "Image analyzed (stream mode not supported for vision yet, showing raw)."
    
    await status.delete()
    await message.answer(f"{reply}\n\n_— Powered by: {provider}_")

@dp.message(F.text)
async def handle_text(message: types.Message):
    if message.from_user.id != ADMIN_ID: return
    
    history = await get_history(message.from_user.id)
    messages = history + [{"role": "user", "content": message.text}]

    # Create a placeholder for streaming
    msg = await message.reply("⏳ *Thinking...*")
    
    reply, provider = await stream_and_update(msg, messages)
    
    # Save to DB only after completion
    await save_message(message.from_user.id, "user", message.text)
    await save_message(message.from_user.id, "assistant", reply)

async def main():
    await init_db()
    logging.basicConfig(level=logging.INFO)
    print("🚀 Advanced Bot is starting...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
