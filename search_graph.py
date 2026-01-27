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
from pyvis.network import Network
import os
import difflib
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import re

# 載入 .env 檔案
load_dotenv()

# 設定 DeepSeek API
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

llm = None
if DEEPSEEK_API_KEY:
    try:
        llm = ChatOpenAI(
            model="deepseek-chat",
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_BASE_URL,
            temperature=0.1
        )
    except Exception as e:
        print(f"⚠️ Warning: Failed to initialize LLM: {e}")
else:
    print("⚠️ Warning: DEEPSEEK_API_KEY not found. LLM features will be disabled.")

# ==========================================
# 配置
# ==========================================
STYLE_CONFIG = {
    "groups": {
        "人物": {"color": "#FF6B6B", "shape": "dot"},
        "参与者": {"color": "#FF6B6B", "shape": "dot"},
        "User": {"color": "#FF6B6B", "shape": "dot"},
        "Assistant": {"color": "#FF8C00", "shape": "dot"},
        "事件": {"color": "#FFD700", "shape": "diamond"},
        "主题": {"color": "#FFD700", "shape": "diamond"},
        "地点": {"color": "#48C9B0", "shape": "triangle"},
        "物品": {"color": "#5DADE2", "shape": "box"},
        "行为": {"color": "#AF7AC5", "shape": "star"},
        "特征": {"color": "#D5D8DC", "shape": "ellipse"},
        "default": {"color": "#999999", "shape": "dot"}
    }
}

PHYSICS_CONFIG = """
{
  "physics": {
    "enabled": false
  },
  "interaction": {
    "dragNodes": true,
    "dragView": true,
    "hideEdgesOnDrag": false,
    "hideNodesOnDrag": false,
    "hover": true,
    "navigationButtons": true,
    "keyboard": true,
    "multiselect": true
  }
}
"""

class GraphSearcher:
    def __init__(self, json_file="graph_messages_data.json"):
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
            self.G = nx.node_link_graph(data, edges="links")
            print(f"✅ 圖譜加載成功!")
            print(f"   - 節點數: {self.G.number_of_nodes()}")
            print(f"   - 邊數: {self.G.number_of_edges()}")
            return True
        except Exception as e:
            print(f"❌ 加載失敗: {e}")
            return False

    def visualize_subgraph(self, subgraph, output_file="search_result.html", title="Search Result"):
        """生成子圖的 HTML 可視化"""
        print(f"🎨 正在生成可視化文件: {output_file} ...")

        try:
            net = Network(height="800px", width="100%", bgcolor="#ffffff", font_color="black", select_menu=True)
            net.set_options(PHYSICS_CONFIG)

            # 使用靜態佈局
            pos = nx.spring_layout(subgraph, seed=42, k=2, scale=1000)

            # 添加節點
            for node, attrs in subgraph.nodes(data=True):
                group = attrs.get('group', 'default')
                # 這裡做簡化處理，因為 search_graph 的 STYLE_CONFIG 可能不完整
                style = STYLE_CONFIG['groups'].get(group, STYLE_CONFIG['groups']['default'])

                size = 20 + subgraph.degree(node) * 2
                x, y = pos[node]

                # Title
                node_title = attrs.get('title', node)

                net.add_node(
                    node,
                    label=node,
                    title=node_title,
                    group=group,
                    color=attrs.get('community_color', style['color']),
                    shape=style['shape'],
                    size=size,
                    x=x, y=y,
                    borderWidth=2,
                    shadow=True,
                    font={'size': 14, 'color': 'black', 'face': 'Microsoft YaHei'}
                )

            # 添加邊
            for u, v, attrs in subgraph.edges(data=True):
                weight = attrs.get('weight', 1)
                width = 1 + (weight * 0.5)
                color = "#FF6B6B" if weight >= 8 else "#AAB7B8"

                net.add_edge(
                    u, v,
                    title=attrs.get('title', ''),
                    width=width,
                    color={'color': color, 'opacity': 0.8},
                    arrows={'to': {'enabled': True}},
                    font={'size': 10, 'color': 'black', 'align': 'middle', 'background': 'rgba(255,255,255,0.7)'},
                    smooth={'type': 'curvedCW', 'roundness': 0.2}
                )

            net.save_graph(output_file)
            print(f"✅ 可視化文件已保存: {os.path.abspath(output_file)}")
            # import webbrowser
            # webbrowser.open(output_file)

        except Exception as e:
            print(f"❌ 可視化生成失敗: {e}")

    def analyze_query_with_llm(self, json_file="messages_input_analysis.json"):
        """使用 LLM 分析對話歷史 (JSON)，提取關鍵詞"""
        if not llm:
            print("⚠️ LLM 未初始化，無法分析。")
            return []

        if not os.path.exists(json_file):
            print(f"⚠️ 找不到文件: {json_file}")
            return []

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                messages_data = json.load(f)


        except Exception as e:
            print(f"⚠️ 讀取或解析文件失敗: {e}")
            return []

        system_prompt = """你是一個知識圖譜搜索助手。你的任務是分析對話內容，提取出對話內容在於關鍵實體名稱及其類型。

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
        1. 提取對話中提到的核心實體，不限於一個。
        2. 推斷實體的類型。
        3. 只返回 JSON，不要包含其他文本。
        4. 忽略MANUKA, Kiwi
        """

        messages = [
            SystemMessage(content=system_prompt)
        ]

        for msg in messages_data:
            role = msg.get('role', 'unknown').lower()
            content = msg.get('content', '')

            if role == 'user':
                messages.append(HumanMessage(content=content))
            elif role == 'assistant':
                messages.append(AIMessage(content=content))

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
            return data.get("entities", [])
        except Exception as e:
            print(f"⚠️ LLM 分析失敗: {e}")
            return []


    def search(self, query, entity_type=None):
        """搜索節點並顯示相關信息"""
        if not self.G:
            return

        query = query.strip()
        if not query:
            return

        # 1. 處理 batch 搜索 (格式: batch:1 或 b:1) - 優先級最高，不需要 LLM 分析
        if query.lower().startswith("batch:") or query.lower().startswith("b:"):
            try:
                batch_part = query.split(':', 1)[1].strip()
                if not batch_part:
                    print("❌ 請指定 Batch Index，例如: batch:1")
                    return
                batch_index = int(batch_part)
                self._search_by_batch(batch_index)
                return
            except ValueError:
                print(f"❌ 格式錯誤。無法解析 Batch Index。請確認輸入的是整數 (例如: batch:1)。")
                return

        print(f"\n🔍 搜索結果: '{query}' (類型限制: {entity_type if entity_type else '無'})")
        print("=" * 50)

        matches = []

        # 2. 實體搜索 (使用 difflib.get_close_matches 優化)
        all_nodes = list(self.G.nodes())
        matched_nodes = []
        query_lower = query.lower()

        # 1. 模糊匹配
        close_matches = difflib.get_close_matches(query_lower, all_nodes, n=10, cutoff=0.5)
        matched_nodes.extend(close_matches)
        print(close_matches)
        # 2. 子串匹配
        for node in all_nodes:
            if query_lower in node:
                matched_nodes.append(node)

        # for node in matched_nodes:
        #     attrs = self.G.nodes[node]
        #     node_group = str(attrs.get('group', ''))

        #     # 如果有指定類型，則進行過濾
        #     if entity_type and entity_type != "Unknown":
        #         if entity_type.lower() in node_group.lower() or node_group.lower() in entity_type.lower():
        #             matched_nodes.append(node)
        #     else:
        #         matched_nodes.append(node)

        if not matched_nodes:
            print("❌ 未找到匹配的實體。")
            return False

        print(f"找到 {len(matched_nodes)} 個相關實體:\n")

        # 收集所有相關節點及其鄰居構建子圖
        self.generate_searched_graph(self.G, matched_nodes, query)

        return True
    def generate_searched_graph(self, G, matched_nodes, query):
        # 收集所有相關節點及其鄰居構建子圖
        subgraph_nodes = set(matched_nodes)
        for node in matched_nodes:
            subgraph_nodes.update(G.successors(node))
            subgraph_nodes.update(G.predecessors(node))

        if subgraph_nodes:
            subgraph = self.G.subgraph(subgraph_nodes)
            # 文件名加入類型區分，避免覆蓋
            safe_query = re.sub(r'[\\/*?:"<>|]', "", query)
            self.visualize_subgraph(subgraph, f"search_entity_{safe_query}.html", f"Search: {query}")

        # 顯示每個匹配節點的詳細信息
        for node_name in matched_nodes:
            self._print_node_details(node_name)

    def _search_by_batch(self, batch_index):
        """根據 batch_source 搜索並顯示關係"""
        print(f"\n🔍 搜索 Batch Source: {batch_index}")
        print("=" * 50)

        found_edges = []
        # 遍歷所有邊尋找匹配的 batch_source
        for u, v, attrs in self.G.edges(data=True):
            # 注意：JSON 加載進來後屬性名稱保持不變
            if attrs.get('batch_source') == batch_index:
                found_edges.append((u, v, attrs))

        if not found_edges:
            print(f"❌ 找不到 batch_source 為 {batch_index} 的關係數據。")
            # 嘗試列出可用的 batches
            avail_batches = set()
            for _, _, a in self.G.edges(data=True):
                if 'batch_source' in a:
                    avail_batches.add(a['batch_source'])
            if avail_batches:
                print(f"💡 目前可用的 Batch Index: {sorted(list(avail_batches))}")
            return

        print(f"找到 {len(found_edges)} 條關係:\n")

        # 收集涉及的節點以構建子圖
        subgraph_edges = []
        subgraph_nodes = set()

        # 格式化輸出
        for u, v, attrs in found_edges:
            subgraph_edges.append((u, v))
            subgraph_nodes.add(u)
            subgraph_nodes.add(v)

            desc = attrs.get('description', '')
            # 只取第一行描述，避免太長
            short_desc = desc.split('\n')[0] if desc else "無描述"
            if len(short_desc) > 30:
                short_desc = short_desc[:30] + "..."

            weight = attrs.get('weight', 0)

            print(f"   🔗 {u} -> {v}")
            print(f"      📝 描述: {short_desc}")
            print("-" * 30)

        # 生成可視化文件
        if subgraph_nodes:
             # 使用 edge_subgraph 保持原有的邊屬性
            subgraph = self.G.edge_subgraph(subgraph_edges)
            self.visualize_subgraph(subgraph, f"search_batch_{batch_index}.html", f"Batch Source: {batch_index}")

        print("=" * 50 + "\n")

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

        # 入度 (被動關係)
        in_edges = list(self.G.in_edges(node_name, data=True))
        if in_edges:
            print("   ⬅️  被動關係 (In):")
            for _, edge_attrs in in_edges:
                rel_desc = edge_attrs.get('description', edge_attrs.get('label', '相關'))
                rel_desc = rel_desc.split('\n')[0] if rel_desc else "相關"
                weight = edge_attrs.get('weight', 1)

        print("=" * 50 + "\n")

def main():
    searcher = GraphSearcher()

    if not searcher.G:
        return
    print("� 提示: 輸入 'b:1' 或 'batch:1' 查看特定批次的關係")
    print("�👉 輸入 'q' 或 'exit' 退出程序。\n")

    while True:
        try:
            mode = input("選擇查看特定批次輸入 'b'，用使llm分析輸入 'l' (輸入 'q' 退出): ").strip().lower()

            if mode in ['q', 'exit', 'quit']:
                print("👋 再見!")
                break

            if mode == 'b':
                user_input = input("Search Batch > ").strip()
                # 自動添加前綴，如果用戶只輸入數字
                if user_input.isdigit():
                    user_input = f"batch:{user_input}"
                searcher.search(user_input)
            elif mode == 'l':
                print(f"🤖 正在分析 messages_input_analysis.json ...")
                entities = searcher.analyze_query_with_llm()

                found_count = 0
                if not entities:
                    print("⚠️ 未提取到任何實體。")

                for entity in entities:
                    name = entity.get('entity_name')
                    etype = entity.get('entity_type')
                    print(f"\n--- 搜索: {name} ({etype}) ---")
                    if searcher.search(name, etype):
                        found_count += 1
            else:
                print("⚠️ 無效的輸入，請重試。")

        except KeyboardInterrupt:
            print("\n👋 再見!")
            break
        except Exception as e:
            print(f"發生錯誤: {e}")

if __name__ == "__main__":
    main()
