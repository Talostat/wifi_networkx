from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
import re
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
system_prompt = """你是一個知識圖譜構建專家。任務是從文字內容中提取三元組，只關注三個維度：

===三個提取維度===
1. 話題（Topic）：對話中討論的主要話題、事件、活動
   例如：出門遊玩、購物、散步、直播、討論

2. 地點（Location）：提及的具體地點、地區、場所
   例如：飲料店、公園、家中、馬路、車站前、戶外

3. 物品（Object）：提及的具體物品、事物、物體
   例如：狐狸耳朵、貼圖、貓、飲料、手

===三種關係類型===
1. 話題-話題關聯：(話題A, 關係詞, 話題B)
   例如：(出門遊玩, 包括, 購物)、(散步, 進行到, 購飲料)

2. 話題-地點關聯：(話題, 發生地, 地點) 或 (話題, 地點, 地點)
   例如：(購物, 發生於, 飲料店)、(散步, 經過, 公園)

3. 話題-物品關聯：(話題, 涉及, 物品) 或 (物品, 出現於, 話題)
   例如：(購飲料, 涉及, 飲料)、(直播, 收到, 貼圖)

===嚴格規則===
A. 只提取以下三種關係，其他一律排除：
   - 人物、人名、身份、性格、情感、行為動作 → 排除
   - 只保留話題、地點、物品三者的明確關聯

B. 實體來源：
   - 話題：明確的活動、事件、行為（必須是名詞形式，如「購物」而非「買」）
   - 地點：具體的地名、場所名稱
   - 物品：具體的物品名稱、事物

C. 排除項目：
   - 所有人名和人物相關信息
   - 情感、性格、形容詞
   - 對話內容、言論
   - 時間條件
   - 推斷或自行添加的實體

===良好範例===
✓ (出門遊玩, 包括, 購物) - 話題-話題
✓ (購物, 發生於, 飲料店) - 話題-地點
✓ (直播, 收到, 貼圖) - 話題-物品
✓ (散步, 經過, 公園) - 話題-地點

===不良範例===
✗ (Kiwi, 結伴, MANUKA) - 人物排除
✗ (MANUKA, 感到, 興奮) - 情感排除
✗ (貓, 很, 可愛) - 形容詞排除
✗ (兩人, 牽手, 過馬路) - 人物行為排除，應改為(散步, 發生於, 馬路)

===輸出格式===
JSON格式，每個話題包含：
- event_id: E1, E2等
- event_name: 話題簡述
- main_topics: 該話題涉及的主要事項列表
- triples: 三元組數組，格式為 [subject, relation, object, entity_type]
  其中 entity_type 為：topic_topic、topic_location、topic_object

===範例輸出===
{
  "events": [
    {
      "event_id": "E1",
      "event_name": "規劃出門",
      "main_topics": ["出門遊玩", "購物規劃"],
      "triples": [
        ["出門遊玩", "包括", "購物", "topic_topic"],
        ["購物", "發生於", "飲料店", "topic_location"],
        ["購物", "涉及", "飲料", "topic_object"]
      ]
    }
  ]
}

===提取指南===
1. 先識別所有話題、地點、物品
2. 只連接這三種實體之間的明確關係
3. 排除所有人物相關信息
4. 每個三元組應清晰且獨立成立
5. 嚴格遵循三種關係類型
6. 謝詞要簡潔（發生於、包括、涉及、經過、前往等）"""

def extract_triples_from_text(text, source_name):
    """從文本中提取三元組"""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"請分析以下內容，提取三元組並輸出為JSON格式：\n\n{text}")
    ]

    response = llm.batch([messages])[0]

    return response.content


def parse_json_response(response_text):
    """解析LLM的JSON響應"""
    try:
        # 嘗試直接解析
        data = json.loads(response_text)
        return data
    except json.JSONDecodeError:
        # 嘗試提取JSON塊
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                return data
            except Exception:
                pass

    return None

def convert_json_to_text_format(data):
    """將JSON格式轉換為文本格式[E1][類別] (主體, 關係, 客體)"""
    lines = []

    if not isinstance(data, dict) or 'events' not in data:
        return ""

    for event in data['events']:
        event_id = event.get('event_id', 'E1')
        event_name = event.get('event_name', '')

        # 主事件
        subjects = event.get('main_subjects', [])
        if subjects:
            main_subject = '和'.join(subjects)
            lines.append(f"[{event_id}][事件] ({main_subject}, 進行, {event_name})")

        # 子三元組 - 支持數組格式 [subject, relation, object, category]
        triples = event.get('triples', [])
        for i, triple in enumerate(triples, 1):
            if isinstance(triple, list) and len(triple) >= 4:
                # 數組格式: [subject, relation, object, category]
                subject, relation, obj, category = triple[0], triple[1], triple[2], triple[3]
            elif isinstance(triple, dict):
                # 字典格式: {"subject": ..., "relation": ..., ...}
                subject = triple.get('subject', '')
                relation = triple.get('relation', '')
                obj = triple.get('object', '')
                category = triple.get('category', '行為')
            else:
                continue

            sub_id = f"{event_id}.{i}"
            lines.append(f"[{sub_id}][{category}] ({subject}, {relation}, {obj})")

    return "\n".join(lines)

# 4. 選擇要處理的數據源
print("請選擇要提取三元組的數據源：")
print("1. messages.json (對話記錄)")
print("2. summary.json (摘要)")
print("3. 兩者都提取")

choice = input("請輸入選擇 (1/2/3): ").strip()

# 初始化變數
messages_json = None
summary_json = None
messages_text_output = ""
summary_text_output = ""

# 根據選擇處理數據
if choice in ["1", "3"]:
    # 讀取並處理 messages.json
    try:
        with open("messages.json", "r", encoding="utf-8") as f:
            messages_data = json.load(f)

        # 取前 20 句對話並格式化為文字
        conversation_lines = []
        for i, msg in enumerate(messages_data[:20]):
            role = msg["role"]
            content = msg["content"]
            speaker = "User" if role == "user" else "Assistant"
            conversation_lines.append(f"{speaker}: {content}")

        messages_text = "\n".join(conversation_lines)
        print("\n正在提取 messages.json 的三元組...")
        response = extract_triples_from_text(messages_text, "messages.json")
        print(response)
        # 解析JSON
        messages_json = parse_json_response(response)
        if messages_json:
            messages_text_output = convert_json_to_text_format(messages_json)
            print("✓ messages.json 提取完成")
        else:
            print("⚠ 無法解析 messages.json 響應")
    except FileNotFoundError:
        print("錯誤：找不到 messages.json")
    except Exception as e:
        print(f"錯誤：{e}")

if choice in ["2", "3"]:
    # 讀取並處理 summary.json
    try:
        with open("summary.json", "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        summary_text = summary_data.get("summary", "")

        print("\n正在提取 summary.json 的三元組...")
        response = extract_triples_from_text(summary_text, "summary.json")
        print(response)

        # 解析JSON
        summary_json = parse_json_response(response)
        if summary_json:
            summary_text_output = convert_json_to_text_format(summary_json)
            print("✓ summary.json 提取完成")
        else:
            print("⚠ 無法解析 summary.json 響應")
    except FileNotFoundError:
        print("錯誤：找不到 summary.json")
    except Exception as e:
        print(f"錯誤：{e}")

# 5. 儲存結果

# 保存文本格式
output_lines = []
if messages_text_output:
    output_lines.append(messages_text_output)
if summary_text_output:
    if output_lines:
        output_lines.append("")
    output_lines.append(summary_text_output)

with open("triples_comparison_categorized.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

# 保存JSON格式
json_output = {
    "metadata": {
        "sources": []
    },
    "data": {}
}

if messages_json:
    json_output['metadata']['sources'].append('messages.json')
    json_output['data']['messages'] = messages_json

if summary_json:
    json_output['metadata']['sources'].append('summary.json')
    json_output['data']['summary'] = summary_json

with open("triples_comparison_categorized.json", "w", encoding="utf-8") as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)

print("\n✓ 結果已儲存至:")
print("  - triples_comparison_categorized.txt (文本格式)")
print("  - triples_comparison_categorized.json (JSON格式)")

