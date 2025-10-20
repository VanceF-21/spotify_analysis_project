import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, HeteroData
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List
import networkx as nx


class MusicGraphBuilder:
    """构建艺人合作网络图"""
    
    def __init__(self, artists_df: pd.DataFrame, collaborations_df: pd.DataFrame, 
                 artist_to_idx: Dict):
        """
        初始化图构建器
        
        Args:
            artists_df: 艺人特征DataFrame
            collaborations_df: 合作边DataFrame
            artist_to_idx: 艺人名到索引的映射
        """
        self.artists_df = artists_df
        self.collaborations_df = collaborations_df
        self.artist_to_idx = artist_to_idx
        self.num_artists = len(artist_to_idx)
        
        self.scaler = StandardScaler()
        
    def build_node_features(self) -> torch.Tensor:
        """
        构建节点特征矩阵
        
        Returns:
            node_features: [num_artists, num_features]
        """
        print("\nBuilding node features...")
        
        # 选择数值特征
        feature_columns = [
            'song_count',
            'avg_popularity',
            'collaboration_count',
            'collaboration_rate',
            'avg_danceability',
            'avg_energy',
            'avg_loudness',
            'avg_speechiness',
            'avg_acousticness',
            'avg_instrumentalness',
            'avg_valence',
            'std_danceability',
            'std_energy'
        ]
        
        # 确保艺人按索引排序
        artists_sorted = self.artists_df.copy()
        artists_sorted['idx'] = artists_sorted['artist'].map(self.artist_to_idx)
        artists_sorted = artists_sorted.sort_values('idx')
        
        # 提取特征
        features = artists_sorted[feature_columns].values
        
        # 标准化
        features = self.scaler.fit_transform(features)
        
        # 转换为tensor
        node_features = torch.FloatTensor(features)
        
        print(f"Node feature shape: {node_features.shape}")
        return node_features
    
    def build_edge_index_and_weights(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        构建边索引和边权重
        
        Returns:
            edge_index: [2, num_edges] 边的起点和终点索引
            edge_attr: [num_edges, num_edge_features] 边特征
            edge_label: [num_edges] 边的标签（流行度）
        """
        print("\nBuilding edges...")
        
        edge_list = []
        edge_features = []
        edge_labels = []
        
        for _, row in self.collaborations_df.iterrows():
            artist_1 = row['artist_1']
            artist_2 = row['artist_2']
            
            # 获取索引
            idx_1 = self.artist_to_idx.get(artist_1)
            idx_2 = self.artist_to_idx.get(artist_2)
            
            if idx_1 is None or idx_2 is None:
                continue
            
            # 无向图：添加双向边
            edge_list.append([idx_1, idx_2])
            edge_list.append([idx_2, idx_1])
            
            # 边特征：合作次数、歌曲音频特征
            edge_feat = [
                row['collab_count'],
                row['danceability'],
                row['energy'],
                row['valence']
            ]
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)  # 双向边使用相同特征
            
            # 边标签：流行度
            edge_labels.append(row['points'])
            edge_labels.append(row['points'])
        
        edge_index = torch.LongTensor(edge_list).t().contiguous()
        edge_attr = torch.FloatTensor(edge_features)
        edge_label = torch.FloatTensor(edge_labels)
        
        print(f"Number of edges: {edge_index.shape[1]}")
        print(f"Edge feature shape: {edge_attr.shape}")
        
        return edge_index, edge_attr, edge_label
    
    def build_negative_edges(self, num_negatives: int = None) -> torch.Tensor:
        """
        构建负样本边（未合作的艺人对）
        
        Args:
            num_negatives: 负样本数量，默认与正样本相同
            
        Returns:
            negative_edge_index: [2, num_negative_edges]
        """
        print("\nGenerating negative samples...")
        
        # 获取已存在的边
        existing_edges = set()
        for _, row in self.collaborations_df.iterrows():
            idx_1 = self.artist_to_idx.get(row['artist_1'])
            idx_2 = self.artist_to_idx.get(row['artist_2'])
            if idx_1 is not None and idx_2 is not None:
                existing_edges.add((min(idx_1, idx_2), max(idx_1, idx_2)))
        
        # 生成负样本
        if num_negatives is None:
            num_negatives = len(existing_edges)
        
        negative_edges = []
        attempts = 0
        max_attempts = num_negatives * 10
        
        while len(negative_edges) < num_negatives and attempts < max_attempts:
            idx_1 = np.random.randint(0, self.num_artists)
            idx_2 = np.random.randint(0, self.num_artists)
            
            if idx_1 != idx_2:
                edge = (min(idx_1, idx_2), max(idx_1, idx_2))
                if edge not in existing_edges and edge not in negative_edges:
                    negative_edges.append([idx_1, idx_2])
            
            attempts += 1
        
        negative_edge_index = torch.LongTensor(negative_edges).t().contiguous()
        print(f"Generated {negative_edge_index.shape[1]} negative edges")
        
        return negative_edge_index
    
    def calculate_similarity_features(self) -> pd.DataFrame:
        """
        计算艺人对之间的相似度特征
        用于链接预测
        """
        print("\nCalculating artist pair similarity features...")
        
        # 获取艺人特征矩阵
        artists_sorted = self.artists_df.copy()
        artists_sorted['idx'] = artists_sorted['artist'].map(self.artist_to_idx)
        artists_sorted = artists_sorted.sort_values('idx')
        
        # 音乐风格特征
        style_features = [
            'avg_danceability', 'avg_energy', 'avg_loudness',
            'avg_speechiness', 'avg_acousticness', 'avg_valence'
        ]
        
        style_matrix = artists_sorted[style_features].values
        
        # 计算余弦相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarity_matrix = cosine_similarity(style_matrix)
        
        return similarity_matrix
    
    def build_graph_data(self, include_negative: bool = True) -> Data:
        """
        构建完整的PyTorch Geometric图数据对象
        
        Args:
            include_negative: 是否包含负样本
            
        Returns:
            data: PyTorch Geometric Data对象
        """
        print("\n" + "="*50)
        print("Building PyTorch Geometric graph...")
        print("="*50)
        
        # 构建节点特征
        x = self.build_node_features()
        
        # 构建边
        edge_index, edge_attr, edge_label = self.build_edge_index_and_weights()
        
        # 创建Data对象
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_label=edge_label,
            num_nodes=self.num_artists
        )
        
        # 添加负样本
        if include_negative:
            data.negative_edge_index = self.build_negative_edges()
        
        print("\nGraph statistics:")
        print(f"  Number of nodes: {data.num_nodes}")
        print(f"  Number of edges: {data.edge_index.shape[1]}")
        print(f"  Node feature dim: {data.x.shape[1]}")
        print(f"  Edge feature dim: {data.edge_attr.shape[1]}")
        print("="*50)
        
        return data
    
    def build_networkx_graph(self) -> nx.Graph:
        """
        构建NetworkX图用于网络分析
        
        Returns:
            G: NetworkX无向图
        """
        G = nx.Graph()
        
        # 添加节点
        for artist, idx in self.artist_to_idx.items():
            artist_data = self.artists_df[self.artists_df['artist'] == artist].iloc[0]
            G.add_node(idx, 
                      artist=artist,
                      popularity=artist_data['avg_popularity'],
                      nationality=artist_data['nationality'])
        
        # 添加边
        for _, row in self.collaborations_df.iterrows():
            idx_1 = self.artist_to_idx.get(row['artist_1'])
            idx_2 = self.artist_to_idx.get(row['artist_2'])
            
            if idx_1 is not None and idx_2 is not None:
                G.add_edge(idx_1, idx_2, 
                          weight=row['points'],
                          collab_count=row['collab_count'])
        
        print(f"\nNetworkX graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    
    def compute_network_features(self, G: nx.Graph) -> pd.DataFrame:
        """
        计算网络拓扑特征
        
        Args:
            G: NetworkX图
            
        Returns:
            network_features_df: 包含网络特征的DataFrame
        """
        print("\nComputing network features...")
        
        # 计算各种中心性
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        pagerank = nx.pagerank(G, weight='weight')
        
        # 聚类系数
        clustering = nx.clustering(G)
        
        # 转换为DataFrame
        network_features = []
        for node in G.nodes():
            network_features.append({
                'artist_idx': node,
                'degree_centrality': degree_centrality[node],
                'betweenness_centrality': betweenness_centrality[node],
                'closeness_centrality': closeness_centrality[node],
                'pagerank': pagerank[node],
                'clustering_coefficient': clustering[node],
                'degree': G.degree(node)
            })
        
        network_df = pd.DataFrame(network_features)
        print(f"Computed network features for {len(network_df)} nodes")
        
        return network_df


if __name__ == "__main__":
    # 测试代码
    from data_loader import MusicDataLoader
    
    loader = MusicDataLoader("your_data.csv")
    songs, artists, collabs = loader.process_all()
    artist_to_idx, idx_to_artist = loader.get_artist_mapping()
    
    builder = MusicGraphBuilder(artists, collabs, artist_to_idx)
    graph_data = builder.build_graph_data()
    
    print("\nGraph data summary:")
    print(graph_data)