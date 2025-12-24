from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
import re
import os
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()

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
system_prompt = """你是一個專業的知識圖譜構建助手。你的目標是從文本中提取實體（Entities）和關係（Relationships）。

請嚴格遵守以下 JSON 格式輸出，不要包含任何 Markdown 標記或其他文本：

{
  "entities": [
    {
      "entity_name": "實體名稱",
      "entity_type": "實體類型 (如: 人物, 地點, 事件, 物品, 組織, 概念等)",
      "entity_description": "實體的詳細描述"
    }
  ],
  "relationships": [
    {
      "source_entity": "源實體名稱 (必須在 entities 中出現)",
      "target_entity": "目標實體名稱 (必須在 entities 中出現)",
      "relationship_description": "關係描述",
      "relationship_strength": 1-10 的整數 (表示關係強度)
    }
  ]
}

注意：
1. 確保 JSON 格式合法。
2. 實體名稱應保持原文。
3. 盡可能提取所有重要的實體和關係。
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
        # 1. 嘗試直接解析
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    # 2. 嘗試提取 Markdown 代碼塊 (```json ... ```)
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 嘗試提取最外層的 {}
    try:
        # 尋找第一個 {
        start = response_text.find('{')
        # 尋找最後一個 }
        end = response_text.rfind('}')

        if start != -1 and end != -1 and end > start:
            json_str = response_text[start:end+1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    return None

def convert_json_to_text_format(data):
    """將JSON格式轉換為文本格式"""
    lines = []

    # 新格式處理 (entities & relationships)
    if 'entities' in data or 'relationships' in data:
        if 'entities' in data:
            lines.append("【實體】")
            for e in data['entities']:
                name = e.get('entity_name', 'Unknown')
                etype = e.get('entity_type', 'Unknown')
                desc = e.get('entity_description', '')
                lines.append(f"[{etype}] {name}: {desc}")

        if 'relationships' in data:
            if lines: lines.append("")
            lines.append("【關係】")
            for r in data['relationships']:
                src = r.get('source_entity', '')
                tgt = r.get('target_entity', '')
                desc = r.get('relationship_description', '')
                strength = r.get('relationship_strength', '')
                lines.append(f"({src} -> {tgt}) [{strength}] {desc}")

        return "\n".join(lines)

    # 舊格式處理 (events)
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

def process_json_file(file_path, source_name, is_conversation=False):
    """處理JSON文件並提取三元組

    Args:
        file_path: JSON文件路徑
        source_name: 數據源名稱
        is_conversation: 是否為對話格式（messages.json）

    Returns:
        tuple: (all_parsed_jsons, all_text_outputs)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        all_parsed_jsons = []
        all_text_outputs = []

        # 準備文本內容
        if is_conversation:
            # messages.json: 每 20 句對話分批處理
            batch_size = 20
            total_messages = len(data)
            num_batches = (total_messages + batch_size - 1) // batch_size

            print(f"\n正在處理 {source_name}，共 {total_messages} 條消息，分 {num_batches} 批次...")

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_messages)
                batch_messages = data[start_idx:end_idx]

                # 構建對話文本
                conversation_lines = []
                for msg in batch_messages:
                    role = msg["role"]
                    content = msg["content"]
                    speaker = "User" if role == "user" else "Assistant"
                    conversation_lines.append(f"{speaker}: {content}")
                text_content = "\n".join(conversation_lines)

                print(f"  [批次 {batch_idx + 1}/{num_batches}] 處理第 {start_idx + 1}-{end_idx} 條消息...")
                response = extract_triples_from_text(text_content, f"{source_name} (批次 {batch_idx + 1})")
                print(response)

                # 解析JSON
                parsed_json = parse_json_response(response)
                if parsed_json:
                    text_output = convert_json_to_text_format(parsed_json)
                    all_parsed_jsons.append(parsed_json)
                    all_text_outputs.append(text_output)
                    print(f"  ✓ 批次 {batch_idx + 1} 提取完成")
                else:
                    print(f"  ⚠ 無法解析批次 {batch_idx + 1} 的響應")

            print(f"\n✓ {source_name} 所有批次處理完成")

        else:
            # summary.json: 取 summary 欄位（不分批）
            text_content = data.get("summary", "")
            print(f"\n正在提取 {source_name} 的三元組...")
            response = extract_triples_from_text(text_content, source_name)
            print(response)

            # 解析JSON
            parsed_json = parse_json_response(response)
            if parsed_json:
                text_output = convert_json_to_text_format(parsed_json)
                all_parsed_jsons.append(parsed_json)
                all_text_outputs.append(text_output)
                print(f"✓ {source_name} 提取完成")
            else:
                print(f"⚠ 無法解析 {source_name} 響應")

        return all_parsed_jsons, all_text_outputs

    except FileNotFoundError:
        print(f"錯誤：找不到 {file_path}")
        return [], []
    except Exception as e:
        print(f"錯誤：{e}")
        return [], []

# 4. 選擇要處理的數據源
print("請選擇要提取三元組的數據源：")
print("1. messages.json (對話記錄)")
print("2. summary.json (摘要)")
print("3. 兩者都提取")

choice = input("請輸入選擇 (1/2/3): ").strip()

# 初始化變數
results = {}
output_lines = []

# 根據選擇處理數據
if choice in ["1", "3"]:
    messages_jsons, messages_text_outputs = process_json_file(
        "messages.json", "messages.json", is_conversation=True
    )
    if messages_jsons:
        results['messages'] = messages_jsons
        output_lines.extend(messages_text_outputs)

if choice in ["2", "3"]:
    summary_jsons, summary_text_outputs = process_json_file(
        "summary.json", "summary.json", is_conversation=False
    )
    if summary_jsons:
        results['summary'] = summary_jsons
        if output_lines:
            output_lines.append("")
        output_lines.extend(summary_text_outputs)

# 5. 儲存結果

# 保存文本格式
with open("triples_comparison_categorized.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

# 保存JSON格式
json_output = {
    "metadata": {
        "sources": list(results.keys())
    },
    "data": results
}

with open("triples_comparison_categorized.json", "w", encoding="utf-8") as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)

print("\n✓ 結果已儲存至:")
print("  - triples_comparison_categorized.txt (文本格式)")
print("  - triples_comparison_categorized.json (JSON格式)")

