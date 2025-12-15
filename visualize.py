#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
統一的知識圖譜可視化工具 - 支持完整圖和event高亮
"""

import json
import networkx as nx
from pyvis.network import Network
from collections import defaultdict

COLOR_SCHEME = {
    "人物": "#FF6B6B",
    "事件": "#FFD700",
    "物品": "#87CEEB",
    "地點": "#98D8C8",
    "時間": "#F7DC6F",
    "行為": "#BB8FCE",
    "特徵": "#CCCCCC",
    "關聯": "#999999",
}

CATEGORY_MAPPING = {
    "事件": "事件",
    "行為": "行為",
    "對象": "人物",
    "位置": "地點",
    "時間": "時間",
    "目標": "事件",
    "屬性": "特徵",
    "人物關係": "人物",
    "行為動作": "行為",
    "情感態度": "特徵",
    "屬性特徵": "特徵",
    "時空信息": "地點",
    "計劃目標": "事件",
    "物品事物": "物品",
}


def parse_triples(filename):
    """從JSON文件解析三元組"""
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    triples = []
    event_mapping = {}

    for source_name, source_data in data.get('data', {}).items():
        events = source_data.get('events', [])

        for event in events:
            event_id = event.get('event_id', 'E1')
            event_name = event.get('event_name', '')
            main_subjects = event.get('main_subjects', [])

            event_id_num = event_id.replace('E', '')
            event_mapping[event_id_num] = event_name

            if main_subjects:
                main_subject = '和'.join(main_subjects)
                triples.append({
                    'event_seq': event_id_num,
                    'event_name': event_name,
                    'category': '事件',
                    'subject': main_subject,
                    'relation': '進行',
                    'object': event_name
                })

            event_triples = event.get('triples', [])
            for i, triple in enumerate(event_triples, 1):
                if isinstance(triple, list) and len(triple) >= 4:
                    subject, relation, obj, category = triple[0], triple[1], triple[2], triple[3]
                    sub_event_id = f"{event_id_num}.{i}"
                    triples.append({
                        'event_seq': sub_event_id,
                        'event_name': event_name,
                        'category': category,
                        'subject': subject,
                        'relation': relation,
                        'object': obj
                    })

    return triples, event_mapping


def classify_entity(entity_name, category):
    """根據類別推斷實體類型"""
    if entity_name.startswith('E') and entity_name[1:].replace('.', '').isdigit():
        return "事件"

    mapped = CATEGORY_MAPPING.get(category, None)
    if mapped:
        return mapped

    return "關聯"


def create_graph(triples):
    """建立知識圖"""
    G = nx.DiGraph()
    entity_types = {}

    for triple in triples:
        subject = triple['subject']
        relation = triple['relation']
        obj = triple['object']
        category = triple['category']
        event_seq = triple['event_seq']
        event_name = triple.get('event_name', '')

        subject_type = classify_entity(subject, category)
        obj_type = classify_entity(obj, category)

        entity_types[subject] = subject_type
        entity_types[obj] = obj_type

        if not G.has_node(subject):
            G.add_node(subject, entity_type=subject_type, events={event_seq.split('.')[0]}, event_names={event_name} if event_name else set(), connections=0)
        else:
            G.nodes[subject]['events'].add(event_seq.split('.')[0])
            if event_name:
                G.nodes[subject]['event_names'].add(event_name)

        if not G.has_node(obj):
            G.add_node(obj, entity_type=obj_type, events={event_seq.split('.')[0]}, event_names={event_name} if event_name else set(), connections=0)
        else:
            G.nodes[obj]['events'].add(event_seq.split('.')[0])
            if event_name:
                G.nodes[obj]['event_names'].add(event_name)

        G.nodes[subject]['connections'] = G.nodes[subject].get('connections', 0) + 1
        G.nodes[obj]['connections'] = G.nodes[obj].get('connections', 0) + 1

        if G.has_edge(subject, obj):
            G[subject][obj]['relations'].append(relation)
            G[subject][obj]['weight'] += 1
        else:
            G.add_edge(subject, obj, relations=[relation], categories={category}, weight=1, label=relation)

    return G, entity_types


def visualize_graph(G, entity_types, focus_event=None, output_file='graph_analysis.html'):
    """
    生成知識圖視覺化

    Args:
        G: NetworkX 圖
        entity_types: 實體類型字典
        focus_event: 要高亮的事件ID（如'1'代表E1），None表示顯示全部
        output_file: 輸出文件名
    """
    net = Network(height="1000px", width="100%", bgcolor="#1a1a1a", font_color="#ffffff", directed=True, notebook=False)

    net.set_options("""
    {
        "physics": {
            "enabled": true,
            "barnesHut": {
                "gravitationalConstant": -18000,
                "centralGravity": 0.08,
                "springLength": 380,
                "springConstant": 0.008,
                "damping": 0.25,
                "avoidOverlap": 0.9
            },
            "stabilization": {
                "enabled": true,
                "iterations": 2000,
                "fit": true
            }
        },
        "edges": {
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.6}},
            "smooth": {"type": "continuous", "roundness": 0.5},
            "font": {"color": "white", "size": 11, "align": "middle", "strokeWidth": 2, "background": {"enabled": true, "color": "rgba(0,0,0,0.8)"}}
        },
        "nodes": {
            "font": {"size": 16, "face": "Microsoft YaHei", "color": "white"},
            "margin": 5,
            "shadow": {"enabled": true, "color": "rgba(0,0,0,0.8)", "size": 9, "x": 2, "y": 2}
        },
        "interaction": {"navigationButtons": true, "keyboard": true}
    }
    """)

    # 確定高亮節點
    highlighted_nodes = set()
    if focus_event:
        for node in G.nodes():
            if focus_event in G.nodes[node].get('events', set()):
                highlighted_nodes.add(node)

    # 添加節點
    for node in G.nodes():
        entity_type = entity_types.get(node, "關聯")
        connections = G.nodes[node].get('connections', 0)
        events = G.nodes[node].get('events', set())
        event_names = G.nodes[node].get('event_names', set())

        is_highlighted = node in highlighted_nodes

        if is_highlighted:
            color = COLOR_SCHEME.get(entity_type, COLOR_SCHEME["關聯"])
            size = 50
            border = "#FF0000"
            border_width = 5
        elif focus_event:
            color = "#444444"
            size = 20
            border = "#666666"
            border_width = 2
        else:
            color = COLOR_SCHEME.get(entity_type, COLOR_SCHEME["關聯"])
            size = 30
            border = "#FFFFFF"
            border_width = 2

        events_str = ", ".join(sorted(events)) if events else "N/A"
        event_names_str = " | ".join(sorted(event_names)) if event_names else "N/A"

        hover_text = f"<b>{node}</b><br>類型: {entity_type}<br>關聯度: {connections}<br>事件ID: {events_str}<br>事件名稱: {event_names_str}"
        if is_highlighted:
            hover_text += "<br><b style='color: #FF0000'>★ 高亮節點</b>"

        net.add_node(node, label=node, color={'background': color, 'border': border, 'highlight': {'background': '#FFFF00', 'border': '#FF0000'}},
                    size=size, title=hover_text, borderWidth=border_width, font={'color': 'white', 'size': 14})

    # 添加邊
    for source, target in G.edges():
        if focus_event and (source not in highlighted_nodes and target not in highlighted_nodes):
            continue

        edge_data = G[source][target]
        relations = edge_data.get('relations', [])
        weight = edge_data.get('weight', 1)

        label = relations[0] if len(relations) == 1 else f"{relations[0]}(+{len(relations)-1})"
        color = "#FF6B6B" if weight > 2 else "#66D9EF"
        width = 3 if weight > 2 else 2

        net.add_edge(source, target, label=label, color=color, width=width, font={'color': 'white', 'size': 11})

    net.save_graph(output_file)
    print(f"✓ 圖表已生成: {output_file}")
    if focus_event:
        print(f"  高亮節點: {len(highlighted_nodes)} 個")
    return highlighted_nodes if focus_event else None


def main():
    print("📊 知識圖譜可視化系統")
    print("=" * 70)

    # 解析三元組
    triples, event_mapping = parse_triples("triples_comparison_categorized.json")
    print(f"✓ 已解析 {len(triples)} 個三元組")

    # 建立圖
    G, entity_types = create_graph(triples)
    print(f"✓ 圖表構建完成: {G.number_of_nodes()} 個實體, {G.number_of_edges()} 條邊")

    # 列出事件
    print(f"\n【可用事件】")
    for eid in sorted(event_mapping.keys()):
        print(f"  E{eid}: {event_mapping[eid]}")

    # 選擇顯示模式
    print(f"\n選擇模式:")
    print(f"  1. 顯示完整圖")
    print(f"  2. 高亮特定事件")
    choice = input("輸入 1 或 2: ").strip()

    if choice == "2":
        print(f"\n輸入事件號碼 (1-{max(map(int, event_mapping.keys()))}):")
        eid = input("").strip()
        if eid in event_mapping:
            visualize_graph(G, entity_types, focus_event=eid, output_file=f"graph_event.html")
            print(f"✅ 已生成 graph_event_E{eid}.html")
        else:
            print("❌ 無效的事件號碼")
    else:
        visualize_graph(G, entity_types, focus_event=None, output_file="graph_analysis.html")
        print(f"✅ 已生成 graph_analysis.html")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
