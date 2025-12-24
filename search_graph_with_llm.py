#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
圖譜搜索工具
功能：
1. 讀取 graph_analysis_data.json
2. 提供交互式命令行搜索界面
3. 顯示實體詳細信息及其關係網絡
"""

import json
import networkx as nx
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import re

# 載入 .env 檔案
load_dotenv()

# 設定 DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

llm = None
if DEEPSEEK_API_KEY:
    llm = ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=DEEPSEEK_API_KEY,
        openai_api_base=DEEPSEEK_BASE_URL,
        temperature=0.1
    )
else:
    print("⚠️ Warning: DEEPSEEK_API_KEY not found. LLM features will be disabled.")

class GraphSearcher:
    def __init__(self, json_file="graph_analysis_data.json"):
        self.json_file = json_file
        self.G = None
        self.load_graph()

    def load_graph(self):
        """從 JSON 文件加載圖譜"""
        if not os.path.exists(self.json_file):
            print(f"❌ 錯誤: 找不到數據文件 {self.json_file}")
            print("請先運行 visualize_ms.py 生成數據文件。")
            return False

        try:
            print(f"📂 正在加載圖譜數據: {self.json_file} ...")
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 重建 NetworkX 圖
            self.G = nx.node_link_graph(data)
            print(f"✅ 圖譜加載成功!")
            print(f"   - 節點數: {self.G.number_of_nodes()}")
            print(f"   - 邊數: {self.G.number_of_edges()}")
            return True
        except Exception as e:
            print(f"❌ 加載失敗: {e}")
            return False

    def search(self, query, entity_type=None):
        """搜索節點並顯示相關信息"""
        if not self.G:
            return False

        query = query.strip()
        if not query:
            return False

        print(f"\n🔍 搜索結果: '{query}' (類型: {entity_type})")
        print("=" * 50)

        matches_name = []
        matches_type = []

        # 遍歷所有節點
        for node, attrs in self.G.nodes(data=True):
            node_name = str(node)
            node_group = str(attrs.get('group', ''))

            # 1. 名稱匹配
            if query.lower() in node_name.lower():
                # 修正: 先找到名稱, 然後只保留相關type
                if entity_type and entity_type != "Unknown":
                    if entity_type.lower() in node_group.lower() or node_group.lower() in entity_type.lower():
                        matches_name.append(node)
                else:
                    matches_name.append(node)

            # 2. 類型匹配 (檢查 query 是否為類型)
            # 情況 A: 用戶搜 "零食"，圖中有 group="零食" 的節點
            if query.lower() in node_group.lower():
                if node not in matches_name and node not in matches_type:
                    matches_type.append(node)

        if not matches_name and not matches_type:
            print(f"❌ 未找到匹配的實體: {query}")
            return False

        # 顯示名稱匹配結果
        if matches_name:
            print(f"✅ 找到 {len(matches_name)} 個名稱匹配實體:\n")
            for node_name in matches_name:
                self._print_node_details(node_name)

        # 顯示類型匹配結果
        if matches_type:
            print(f"🏷️ 找到 {len(matches_type)} 個類型相關實體 (匹配 '{query}' 或 '{entity_type}'):\n")
            # 如果數量太多，只顯示前 5 個詳細信息，其他的只列出名字
            for i, node_name in enumerate(matches_type):
                if i < 3: # 只詳細顯示前 3 個
                    self._print_node_details(node_name)
                else:
                    print(f"   • {node_name} (類型: {self.G.nodes[node_name].get('group')})")

            if len(matches_type) > 3:
                print(f"\n   ... (共 {len(matches_type)} 個，僅顯示前 3 個詳細信息)")

        return True

    def _print_node_details(self, node_name):
        """打印單個節點的詳細信息和關係"""
        attrs = self.G.nodes[node_name]
        current_community = attrs.get('community')

        print(f"📍 實體: {node_name}")
        print(f"   類型: {attrs.get('group', '未知')}")

        # 顯示社區信息
        if current_community is not None:
            print(f"   社區: #{current_community}")

        # 顯示描述
        desc = attrs.get('description', '').replace('\n', '\n         ')
        if desc:
            print(f"   描述: {desc}")

        print("-" * 30)

        # --- 新增: 顯示同社區內的強關聯實體 ---
        if current_community is not None:
            # 1. 找出同社區的所有成員
            community_members = []
            for node in self.G.nodes():
                if self.G.nodes[node].get('community') == current_community:
                    community_members.append(node)

            print(f"   🏘️  所屬社區: #{current_community} (共 {len(community_members)} 個成員)")

            # 2. 找出與當前節點有直接連接的同社區成員 (核心關聯)
            community_neighbors = []

            # 收集所有鄰居 (不分出入)
            all_neighbors = set(self.G.successors(node_name)) | set(self.G.predecessors(node_name))

            for neighbor in all_neighbors:
                neighbor_attrs = self.G.nodes[neighbor]
                if neighbor_attrs.get('community') == current_community:
                    # 獲取邊的權重 (取最大值如果有多條邊)
                    weight = 0
                    # 檢查出邊
                    if self.G.has_edge(node_name, neighbor):
                        weight = max(weight, self.G[node_name][neighbor].get('weight', 1))
                    # 檢查入邊
                    if self.G.has_edge(neighbor, node_name):
                        weight = max(weight, self.G[neighbor][node_name].get('weight', 1))

                    community_neighbors.append((neighbor, weight))

            # 按權重排序
            community_neighbors.sort(key=lambda x: x[1], reverse=True)

            if community_neighbors:
                print(f"      🔥 社區內的核心關聯 (Top 5):")
                for neighbor, weight in community_neighbors[:5]:
                    print(f"         ★ {neighbor} (強度: {weight})")

            # 3. 列出社區內的其他重要成員 (按度數排序，展示社區全貌)
            # 計算社區內每個節點的度數
            member_degrees = []
            for member in community_members:
                if member == node_name: continue # 跳過自己
                degree = self.G.degree(member)
                member_degrees.append((member, degree))

            member_degrees.sort(key=lambda x: x[1], reverse=True)

            print(f"      👀 社區內的其他重要成員:")
            shown_count = 0
            for member, degree in member_degrees:
                # 避免重複顯示已經在核心關聯裡顯示過的
                if member in [n for n, w in community_neighbors[:5]]:
                    continue
                print(f"         • {member}")
                shown_count += 1
                if shown_count >= 5: # 最多顯示5個
                    break

            print("-" * 30)
        # 出度 (主動關係)
        out_edges = list(self.G.out_edges(node_name, data=True))
        if out_edges:
            print("   ➡️  主動關係 (Out):")
            for _, target, edge_attrs in out_edges:
                rel_desc = edge_attrs.get('description', edge_attrs.get('label', '相關'))
                # 簡化描述顯示
                rel_desc = rel_desc.split('\n')[0] if rel_desc else "相關"
                weight = edge_attrs.get('weight', 1)
                print(f"      -> {target} : {rel_desc} (強度: {weight})")

        # 入度 (被動關係)
        in_edges = list(self.G.in_edges(node_name, data=True))
        if in_edges:
            print("   ⬅️  被動關係 (In):")
            for source, _, edge_attrs in in_edges:
                rel_desc = edge_attrs.get('description', edge_attrs.get('label', '相關'))
                rel_desc = rel_desc.split('\n')[0] if rel_desc else "相關"
                weight = edge_attrs.get('weight', 1)
                print(f"      <- {source} : {rel_desc} (強度: {weight})")

        print("=" * 50 + "\n")

def analyze_query_with_llm(query):
    """使用 LLM 分析用戶查詢，提取關鍵詞"""
    if not llm:
        return [{"entity_name": query, "entity_type": "Unknown"}]

    system_prompt = """你是一個知識圖譜搜索助手。你的任務是分析用戶的自然語言問題，提取出可能存在於圖譜中的關鍵實體名稱及其類型。

    請輸出一個 JSON 對象，格式如下：
    {
        "entities": [
            {
                "entity_name": "實體名稱",
                "entity_type": "實體類型 (如: 人物, 地點, 事件, 物品, 組織, 概念等)"
            }
        ]
    }

    規則：
    1. 提取問題中的核心實體。
    2. 推斷實體的類型。
    3. 只返回 JSON，不要包含其他文本。
    """

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]

    try:
        response = llm.invoke(messages)
        content = response.content.strip()

        # 嘗試解析 JSON
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if match:
            content = match.group(1)
        elif '{' in content:
             # 簡單的提取
             start = content.find('{')
             end = content.rfind('}') + 1
             content = content[start:end]

        data = json.loads(content)
        return data.get("entities", [{"entity_name": query, "entity_type": "Unknown"}])
    except Exception as e:
        print(f"⚠️ LLM 分析失敗，將使用原始查詢: {e}")
        return [{"entity_name": query, "entity_type": "Unknown"}]

def main():
    searcher = GraphSearcher()

    if not searcher.G:
        return

    print("\n💡 提示: 輸入實體名稱或自然語言問題進行搜索")
    print("👉 輸入 'q' 或 'exit' 退出程序。\n")

    while True:
        try:
            user_input = input("Search (輸入問題或實體) > ")
            if user_input.lower() in ['q', 'exit', 'quit']:
                print("👋 再見!")
                break

            # 使用 LLM 分析
            print("🤖 正在分析問題...")
            entities = analyze_query_with_llm(user_input)
            print(f"🔍 提取實體: {json.dumps(entities, ensure_ascii=False)}")

            found_count = 0
            for entity in entities:
                name = entity.get('entity_name')
                etype = entity.get('entity_type')
                print(f"\n--- 搜索: {name} ({etype}) ---")
                if searcher.search(name, etype):
                    found_count += 1

            if found_count == 0:
                print("❌ 所有關鍵詞均未找到匹配實體。")

        except KeyboardInterrupt:
            print("\n👋 再見!")
            break
        except Exception as e:
            print(f"發生錯誤: {e}")

if __name__ == "__main__":
    main()
