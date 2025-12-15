from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
import os
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

# 1. 設定 DeepSeek API (使用 OpenAI 兼容模式)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 驗證 API Key
if not DEEPSEEK_API_KEY:
    raise ValueError(
        "❌ 錯誤：未設置 DEEPSEEK_API_KEY。\n"
        "請在 .env 檔案中設置：DEEPSEEK_API_KEY=your-api-key\n"
        "或複製 .env.example 為 .env 並填入你的 API Key"
    )

# 2. 初始化 DeepSeek LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0.1
)

# 3. 定義提示詞模板
system_prompt = """你是一個專業的對話分析助手。
請分析給定的對話內容，按照以下格式進行總結：

1. 参与者 - 对话中的个体或角色。
2. 时间 - 对话发生的时间点或时间段。
3. 地点 - 对话发生的地理位置或环境。
4. 主题 - 对话讨论的主要话题或中心思想。
5. 关键词 - 对话中反复出现或重要的词汇。
6. 行动项 - 对话中提到的具体行动或待办事项。
7. 情感基调 - 对话过程中表达的情感色彩或氛围。

請用結構化的方式回答，每一項都要清晰列出。
"""

# 4. 讀取 JSON 檔案
print("正在讀取對話記錄...")
with open("1__formatted_template.json", "r", encoding="utf-8") as f:
    messages_data = json.load(f)

# 5. 取前 20 句對話並格式化為文字
conversation_lines = []
for i, msg in enumerate(messages_data[:20]):
    role = msg["role"]
    content = msg["content"]
    speaker = "User" if role == "user" else "Assistant"
    conversation_lines.append(f"{speaker}: {content}")

conversation = "\n".join(conversation_lines)
print(f"已讀取 {len(conversation_lines)} 條對話\n")

# 6. 構建訊息
messages = [
    SystemMessage(content=system_prompt),
    HumanMessage(content=f"請分析以下對話並進行總結：\n\n{conversation}")
]

# 7. 呼叫 LLM
print("正在分析對話並生成總結...\n")
response = llm.invoke(messages)

# 8. 保存結果到 JSON
output_data = {
    "summary": response.content,
    "analyzed_messages_count": len(conversation_lines)
}

with open("summary.json", "w", encoding="utf-8") as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2)

print("總結已保存至 summary.json")
