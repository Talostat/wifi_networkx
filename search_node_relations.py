import json
import networkx as nx
import os

class GraphRelationExplorer:
    def __init__(self, json_file="graph_analysis_data.json"):
        self.json_file = json_file
        self.G = None
        self.load_graph()

    def load_graph(self):
        """加載圖譜數據"""
        if not os.path.exists(self.json_file):
            print(f"❌ 找不到文件: {self.json_file}")
            return

        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.G = nx.node_link_graph(data)
            print(f"✅ 圖譜加載成功! 節點數: {self.G.number_of_nodes()}, 邊數: {self.G.number_of_edges()}")
        except Exception as e:
            print(f"❌ 加載失敗: {e}")

    def search_similar_nodes(self, query):
        """搜索名稱類似的節點並列出關係"""
        if not self.G:
            return

        print(f"\n🔍 正在搜索包含 '{query}' 的節點...\n")

        found_nodes = []
        for node, attrs in self.G.nodes(data=True):
            # 簡單的子字串匹配，忽略大小寫
            if query.lower() in str(node).lower():
                found_nodes.append(node)

        if not found_nodes:
            print(f"❌ 未找到包含 '{query}' 的節點")
            return

        print(f"找到 {len(found_nodes)} 個相關節點:\n")

        for node in found_nodes:
            self._print_node_relations(node)

    def _print_node_relations(self, node):
        """打印單個節點及其關係"""
        attrs = self.G.nodes[node]
        print(f"📍 節點: {node}")
        print(f"   類型: {attrs.get('group', 'N/A')}")

        # 顯示描述 (如果有)
        desc = attrs.get('description', '')
        if desc:
            print(f"   描述: {desc[:100]}..." if len(desc) > 100 else f"   描述: {desc}")

        print("   關係列表:")

        # Outgoing (主動關係: Node -> Target)
        out_edges = list(self.G.out_edges(node, data=True))
        # Incoming (被動關係: Source -> Node)
        in_edges = list(self.G.in_edges(node, data=True))

        if not out_edges and not in_edges:
            print("      (無連接關係)")

        # 顯示主動關係
        for _, target, edge_data in out_edges:
            label = edge_data.get('description')
            label = label.replace('\n', ' ')  # 清理 label 中的換行符，保持整潔
            # 格式化為 LLM 易讀的三元組形式
            print(f"      - Triple: (Subject: {node}, Predicate: {label}, Object: {target})")

        # 顯示被動關係
        for source, _, edge_data in in_edges:
            label = edge_data.get('description')
            label = label.replace('\n', ' ')  # 清理 label 中的換行符，保持整潔
            # 格式化為 LLM 易讀的三元組形式
            print(f"      - Triple: (Subject: {source}, Predicate: {label}, Object: {node})")

        print("-" * 50)

def main():
    explorer = GraphRelationExplorer()
    if not explorer.G:
        return

    print("💡 提示: 輸入關鍵詞搜索節點及其關係")

    while True:
        try:
            query = input("\n請輸入搜索關鍵詞 (輸入 'q' 退出): ").strip()
            if query.lower() in ['q', 'exit', 'quit']:
                break

            if query:
                explorer.search_similar_nodes(query)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"發生錯誤: {e}")

if __name__ == "__main__":
    main()
