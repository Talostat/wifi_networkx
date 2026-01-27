#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高級知識圖譜可視化工具 (Refactored)
功能：
1. 支持多種數據源格式解析
2. 自動化節點樣式（顏色、形狀）映射
3. 基於權重的動態視覺效果（節點大小、邊粗細）
4. 交互式 HTML 輸出
"""

import json
import networkx as nx
from pyvis.network import Network
import os
import math

# ==========================================
# 配置區域
# ==========================================

# 實體類型樣式配置
# shape 可選: dot, diamond, star, triangle, triangleDown, square, box, ellipse, text
STYLE_CONFIG = {
    "groups": {
        # 人物/參與者
        "人物": {"color": "#FF6B6B", "shape": "dot", "icon": "👤"},
        "参与者": {"color": "#FF6B6B", "shape": "dot"},
        "User": {"color": "#FF6B6B", "shape": "dot"},
        "Assistant": {"color": "#FF8C00", "shape": "dot"},

        # 事件/主題
        "事件": {"color": "#FFD700", "shape": "diamond"},
        "主题": {"color": "#FFD700", "shape": "diamond"},
        "计划目标": {"color": "#FFD700", "shape": "diamond"},

        # 地點/空間
        "地点": {"color": "#48C9B0", "shape": "triangle"},
        "位置": {"color": "#48C9B0", "shape": "triangle"},
        "时空信息": {"color": "#48C9B0", "shape": "triangle"},

        # 時間
        "时间": {"color": "#F7DC6F", "shape": "square"},

        # 物品
        "物品": {"color": "#5DADE2", "shape": "box"},
        "物品事物": {"color": "#5DADE2", "shape": "box"},

        # 行為/動作
        "行为": {"color": "#AF7AC5", "shape": "star"},
        "行动项": {"color": "#AF7AC5", "shape": "star"},
        "行为动作": {"color": "#AF7AC5", "shape": "star"},

        # 特徵/情感
        "特征": {"color": "#D5D8DC", "shape": "ellipse"},
        "特徵": {"color": "#D5D8DC", "shape": "ellipse"},
        "情感基调": {"color": "#F1948A", "shape": "heart"}, # heart 形狀在某些版本可能回退為 ellipse
        "属性": {"color": "#D5D8DC", "shape": "ellipse"},

        # 默認
        "default": {"color": "#999999", "shape": "dot"}
    }
}

# 物理引擎配置
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

# ==========================================
# 核心類定義
# ==========================================

class KnowledgeGraphVisualizer:
    def __init__(self, input_file):
        self.input_file = input_file
        self.G = nx.DiGraph()
        self.raw_data = self._load_data()

    def _load_data(self):
        """加載 JSON 數據"""
        if not os.path.exists(self.input_file):
            print(f"❌ 錯誤: 找不到文件 {self.input_file}")
            return {}
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ 讀取 JSON 失敗: {e}")
            return {}

    def build_graph(self):
        """構建 NetworkX 圖"""
        print("🔄 正在構建圖譜結構...")
        data_content = self.raw_data.get('data', {})

        # 遍歷所有數據源 (如 'messages', 'summary')
        for source_name, source_data in data_content.items():
            self._process_source_data(source_data)

        # 計算節點中心性以調整大小
        self._calculate_node_metrics()

        # 應用 Louvain 社區檢測
        self._apply_louvain_communities()

        print(f"✓ 圖譜構建完成: {self.G.number_of_nodes()} 節點, {self.G.number_of_edges()} 邊")

    def _apply_louvain_communities(self):
        """應用 Louvain 算法進行社區檢測並著色"""
        print("🔍 正在應用 Louvain 算法進行社區檢測...")
        try:
            # Louvain 需要無向圖
            G_undirected = self.G.to_undirected()

            # 使用 NetworkX 內置的 Louvain 算法
            # resolution 參數控制社區的大小，默認 1.0
            communities = nx.community.louvain_communities(G_undirected, seed=42)

            print(f"✓ 檢測到 {len(communities)} 個社區")

            # 為每個社區生成顏色
            # 使用一些預定義的鮮豔顏色
            palette = [
                "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEEAD",
                "#D4A5A5", "#9B59B6", "#3498DB", "#E67E22", "#2ECC71",
                "#F1C40F", "#E74C3C", "#1ABC9C", "#8E44AD", "#2C3E50"
            ]

            for i, community in enumerate(communities):
                color = palette[i % len(palette)]
                for node in community:
                    # 保存社區信息
                    self.G.nodes[node]['community'] = i
                    self.G.nodes[node]['community_color'] = color

        except Exception as e:
            print(f"⚠ Louvain 社區檢測失敗: {e}")
            # 如果失敗，確保沒有殘留的 community_color 屬性影響顯示
            for node in self.G.nodes():
                if 'community_color' in self.G.nodes[node]:
                    del self.G.nodes[node]['community_color']

    def _process_source_data(self, source_data):
        """處理單個數據源的數據（支持字典或列表格式）"""
        if isinstance(source_data, list):
            for item in source_data:
                self._process_single_data_block(item)
        elif isinstance(source_data, dict):
            self._process_single_data_block(source_data)

    def _process_single_data_block(self, source_data):
        """處理單個數據塊 - 層級聚合方式"""
        keys = {'entities', 'relationships', 'batch_index'}
        missing = keys - source_data.keys()
        if missing:
            raise ValueError(f"數據塊缺少：{', '.join(missing)}")

        batch_index = source_data.get('batch_index')

        # 1. 處理實體 (Entities)
        for entity in source_data['entities']:
            name = entity.get('entity_name')
            etype = entity.get('entity_type', '未知')
            desc = entity.get('entity_description', '')

            if name:
                # 創建或更新節點
                if not self.G.has_node(name):
                    self.G.add_node(name,
                                    group=etype,
                                    description=desc)
                else:
                    # 更新現有節點的描述和標題
                    if desc:
                        self.G.nodes[name]['description'] = desc

        # 2. 處理關係 (Relationships)
        for rel in source_data['relationships']:
            src = rel.get('source_entity')
            tgt = rel.get('target_entity')
            desc = rel.get('relationship_description', '')
            #strength = int(rel.get('relationship_strength', 1))

            if src and tgt:
                # 確保節點存在
                if not self.G.has_node(src):
                    self.G.add_node(src, group='未知')

                if not self.G.has_node(tgt):
                    self.G.add_node(tgt, group='未知')

                # # 添加關係邊
                # if self.G.has_edge(src, tgt):
                #     # 邊已存在，累加權重
                #     edge = self.G[src][tgt]
                #     edge['weight'] += strength
                #     edge['source'] = batch_index
                #     current_desc = edge.get('description', '')
                #     if desc and desc not in current_desc:
                #         edge['description'] = f"{current_desc}\n• {desc}"
                #         edge['title'] = f"強度: {edge['weight']}\n描述: {edge['description']}"
                # else:
                    # 新邊
                self.G.add_edge(src, tgt,
                                description=desc,
                                batch_source=batch_index)


    def _format_tooltip(self, name, etype, desc):
        """格式化 HTML Tooltip"""
        html = f"<b>{name}</b><br>"
        html += f"類型: {etype}<br>"
        if desc:
            # 處理換行並使用 div 限制寬度
            safe_desc = desc.replace('\n', '<br>')
            html += f'<div style="max-width: 300px; white-space: pre-wrap; margin-top: 5px;">描述: {safe_desc}</div>'
        return html

    def _calculate_node_metrics(self):
        """計算節點指標並存儲在節點屬性中"""
        degrees = dict(self.G.degree())
        for node in self.G.nodes():
            self.G.nodes[node]['degree'] = degrees.get(node, 0)

    def export_graph_data(self, output_prefix="graph_analysis"):
        """導出圖譜數據為標準格式 (JSON, GraphML)"""
        print(f"💾 正在導出圖譜數據...")

        # 1. 導出為 NetworkX Node-Link JSON
        # 這是最通用的 "純數據" 格式，方便其他程序讀取或重新加載
        try:
            data = nx.node_link_data(self.G)
            json_file = f"{output_prefix}_data.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"✅ JSON 數據已保存至: {json_file}")
        except Exception as e:
            print(f"❌ JSON 導出失敗: {e}")

    def generate_html(self, output_file="graph_analysis.html"):
        """生成 Pyvis HTML"""
        print(f"🎨 正在生成可視化文件: {output_file}...")

        net = Network(height="900px", width="100%", bgcolor="#ffffff", font_color="black", select_menu=True, filter_menu=True)

        # 應用物理配置
        net.set_options(PHYSICS_CONFIG)

        # 預計算佈局 (因為物理引擎已關閉)
        print("📐 正在計算靜態佈局...")
        # scale參數調整節點間距
        pos = nx.spring_layout(self.G, seed=42, k=2, scale=500)

        # 添加節點
        for node, attrs in self.G.nodes(data=True):
            group = attrs.get('group', 'default')
            style = STYLE_CONFIG['groups'].get(group, STYLE_CONFIG['groups']['default'])

            # 動態大小: 基礎大小 + 度數 * 係數
            base_size = 20
            degree_factor = 2
            size = base_size + (attrs.get('degree', 0) * degree_factor)

            # 優先使用社區顏色，如果沒有則使用組顏色
            node_color = attrs.get('community_color', style['color'])

            # 獲取佈局座標
            x, y = pos[node]

            net.add_node(
                node,
                label=node,
                # title=attrs.get('title', node),
                group=group,
                color=node_color,
                shape=style['shape'],
                size=size,
                x=x, # 設置靜態座標
                y=y, # 設置靜態座標
                borderWidth=2,
                shadow=True,
                font={'size': 14, 'color': 'black', 'face': 'Microsoft YaHei'}
            )

        # 添加邊
        for u, v, attrs in self.G.edges(data=True):
            weight = attrs.get('weight', 1)

            # 動態寬度
            width = 1 + (weight * 0.5)
            # 邊的顏色根據強度
            if weight >= 8:
                color = "#FF6B6B"  # 強關係用紅色
            else:
                color = "#AAB7B8"  # 默認灰藍色

            net.add_edge(
                u, v,
                # label=attrs.get('label', ''),
                width=width,
                color={'color': color, 'opacity': 0.8},
                arrows={'to': {'enabled': True, 'scaleFactor': 0.5}},
                font={'size': 10, 'color': 'black', 'strokeWidth': 0, 'align': 'middle', 'background': 'rgba(255,255,255,0.7)'},
                smooth={'type': 'curvedCW', 'roundness': 0.2}
            )

        # 保存
        try:
            net.save_graph(output_file)
            print(f"✅ 成功保存至 {output_file}")

            # 嘗試自動打開（可選）
            # import webbrowser
            # webbrowser.open(output_file)

        except Exception as e:
            print(f"❌ 保存文件失敗: {e}")

def main():
    input_file = "triples_comparison_categorized_messages.json"
    output_file_name = "graph_messages"
    output_file = f"{output_file_name}.html"

    visualizer = KnowledgeGraphVisualizer(input_file)
    visualizer.build_graph()
    visualizer.export_graph_data(output_file_name)
    visualizer.generate_html(output_file)

if __name__ == "__main__":
    main()
