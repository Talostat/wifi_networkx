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

    def search(self, query):
        """搜索節點並顯示相關信息"""
        if not self.G:
            return

        query = query.strip()
        if not query:
            return

        print(f"\n🔍 搜索結果: '{query}'")
        print("=" * 50)

        matches = []

        # 普通實體搜索
        for node in self.G.nodes():
            # 僅搜索節點名稱
            if query.lower() in str(node).lower():
                matches.append(node)

        if not matches:
            print("❌ 未找到匹配的實體。")
            return

        print(f"找到 {len(matches)} 個相關實體:\n")

        # 2. 顯示每個匹配節點的詳細信息
        for node_name in matches:
            self._print_node_details(node_name)

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

def main():
    searcher = GraphSearcher()

    if not searcher.G:
        return

    print("\n💡 提示: 輸入實體名稱進行搜索 (例如: Kiwi)")
    print("👉 輸入 'q' 或 'exit' 退出程序。\n")

    while True:
        try:
            user_input = input("Search > ")
            if user_input.lower() in ['q', 'exit', 'quit']:
                print("👋 再見!")
                break

            searcher.search(user_input)

        except KeyboardInterrupt:
            print("\n👋 再見!")
            break
        except Exception as e:
            print(f"發生錯誤: {e}")

if __name__ == "__main__":
    main()
