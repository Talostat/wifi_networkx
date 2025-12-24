from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
import re
import os

# 1. 設定 DeepSeek API (使用 OpenAI 兼容模式)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 驗證 API Key
if not DEEPSEEK_API_KEY:
    raise ValueError("❌ 錯誤：未設置 DEEPSEEK_API_KEY 環境變數。請設置後重新執行。")

# 2. 初始化 DeepSeek LLM
llm = ChatOpenAI(
    model="deepseek-chat",
    openai_api_key=DEEPSEEK_API_KEY,
    openai_api_base=DEEPSEEK_BASE_URL,
    temperature=0.1
)

# 3. 定義提示詞模板
system_prompt = """你是一個專業的信息抽取專家。

任務：從提供的文本中提取結構化的三元組（主體、關係、客體）。

重要規則：
1. 每個三元組的主體必須是單數實體（如：Kiwi、MANUKA、公園等），不允許在主體中同時出現兩個人名
2. 優先提取主要參與者的個人行為、特徵和狀態
3. 提取群體活動時，應該分別為每個參與者建立三元組

輸出格式為JSON，包含以下結構：
{
  "events": [
    {
      "event_id": "E1",
      "event_name": "事件名稱",
      "main_subjects": ["主體1", "主體2"],
      "triples": [
        ["主體", "關係", "客體", "類別"],
        ...
      ]
    }
  ]
}

類別可以包括：行為、特徵、狀態、關係等。
"""


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

