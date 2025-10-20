import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from itertools import combinations


class CollaborationRecommender:
    """艺人合作推荐系统"""
    
    def __init__(self, model, data, artist_to_idx, idx_to_artist, device='cpu'):
        """
        Args:
            model: 训练好的GNN模型
            data: 图数据
            artist_to_idx: 艺人名到索引映射
            idx_to_artist: 索引到艺人名映射
            device: 设备
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.artist_to_idx = artist_to_idx
        self.idx_to_artist = idx_to_artist
        self.device = device
        
        self.model.eval()
        
        # 预计算所有节点嵌入
        with torch.no_grad():
            self.node_embeddings = self.model.encode(
                self.data.x, 
                self.data.edge_index
            )
        
        # 获取已存在的合作
        self.existing_collabs = self._get_existing_collaborations()
        
    def _get_existing_collaborations(self) -> set:
        """获取已存在的合作关系"""
        existing = set()
        edge_index = self.data.edge_index.cpu().numpy()
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            if src < dst:  # 只存储一个方向
                existing.add((src, dst))
            else:
                existing.add((dst, src))
        
        return existing
    
    def predict_collaboration_score(self, artist1: str, artist2: str) -> float:
        """
        预测两个艺人合作的成功度
        
        Args:
            artist1: 艺人1名称
            artist2: 艺人2名称
            
        Returns:
            预测得分（0-1之间）
        """
        # 获取索引
        idx1 = self.artist_to_idx.get(artist1)
        idx2 = self.artist_to_idx.get(artist2)
        
        if idx1 is None or idx2 is None:
            print(f"Warning: Artist not found in database")
            return 0.0
        
        # 构建边索引
        edge_index = torch.LongTensor([[idx1], [idx2]]).to(self.device)
        
        # 预测
        with torch.no_grad():
            score = self.model.decode(self.node_embeddings, edge_index)
            score = torch.sigmoid(score).item()  # 转换为概率
        
        return score
    
    def recommend_for_artist(self, artist: str, top_k: int = 10,
                            exclude_existing: bool = True) -> List[Dict]:
        """
        为指定艺人推荐合作伙伴
        
        Args:
            artist: 艺人名称
            top_k: 返回top K个推荐
            exclude_existing: 是否排除已有合作
            
        Returns:
            推荐列表，每项包含 {'artist': 艺人名, 'score': 得分}
        """
        artist_idx = self.artist_to_idx.get(artist)
        if artist_idx is None:
            print(f"Artist '{artist}' not found")
            return []
        
        print(f"\nGenerating recommendations for {artist}...")
        
        # 创建所有可能的边
        all_indices = list(range(len(self.artist_to_idx)))
        all_indices.remove(artist_idx)  # 移除自己
        
        candidate_edges = torch.LongTensor([
            [artist_idx] * len(all_indices),
            all_indices
        ]).to(self.device)
        
        # 批量预测
        with torch.no_grad():
            scores = self.model.decode(self.node_embeddings, candidate_edges)
            scores = torch.sigmoid(scores).cpu().numpy()
        
        # 构建推荐列表
        recommendations = []
        for i, candidate_idx in enumerate(all_indices):
            # 检查是否已存在合作
            edge = tuple(sorted([artist_idx, candidate_idx]))
            if exclude_existing and edge in self.existing_collabs:
                continue
            
            recommendations.append({
                'artist': self.idx_to_artist[candidate_idx],
                'artist_idx': candidate_idx,
                'score': float(scores[i])
            })
        
        # 按得分排序
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:top_k]
    
    def recommend_all_pairs(self, top_k: int = 100, 
                           min_score: float = 0.5) -> pd.DataFrame:
        """
        推荐所有可能的艺人合作对
        
        Args:
            top_k: 返回top K个推荐
            min_score: 最低得分阈值
            
        Returns:
            推荐DataFrame
        """
        print("\nGenerating all pair recommendations...")
        print("This may take a while...")
        
        recommendations = []
        num_artists = len(self.artist_to_idx)
        
        # 生成所有可能的艺人对（不包括已合作的）
        batch_size = 1000
        candidate_pairs = []
        
        for idx1 in range(num_artists):
            for idx2 in range(idx1 + 1, num_artists):
                edge = (idx1, idx2)
                if edge not in self.existing_collabs:
                    candidate_pairs.append(edge)
        
        print(f"Evaluating {len(candidate_pairs)} potential collaborations...")
        
        # 批量处理
        for i in range(0, len(candidate_pairs), batch_size):
            batch = candidate_pairs[i:i + batch_size]
            
            edge_index = torch.LongTensor([
                [pair[0] for pair in batch],
                [pair[1] for pair in batch]
            ]).to(self.device)
            
            with torch.no_grad():
                scores = self.model.decode(self.node_embeddings, edge_index)
                scores = torch.sigmoid(scores).cpu().numpy()
            
            for j, (idx1, idx2) in enumerate(batch):
                score = float(scores[j])
                
                if score >= min_score:
                    recommendations.append({
                        'artist_1': self.idx_to_artist[idx1],
                        'artist_2': self.idx_to_artist[idx2],
                        'predicted_score': score
                    })
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i + batch_size}/{len(candidate_pairs)} pairs...")
        
        # 转换为DataFrame并排序
        rec_df = pd.DataFrame(recommendations)
        if len(rec_df) > 0:
            rec_df = rec_df.sort_values('predicted_score', ascending=False)
            rec_df = rec_df.head(top_k)
        
        print(f"\nFound {len(rec_df)} high-potential collaborations")
        return rec_df
    
    def recommend_by_similarity(self, artist: str, top_k: int = 10) -> List[Dict]:
        """
        基于嵌入相似度推荐合作伙伴
        
        Args:
            artist: 艺人名称
            top_k: 返回数量
            
        Returns:
            推荐列表
        """
        artist_idx = self.artist_to_idx.get(artist)
        if artist_idx is None:
            print(f"Artist '{artist}' not found")
            return []
        
        # 获取该艺人的嵌入
        artist_embedding = self.node_embeddings[artist_idx]
        
        # 计算与所有其他艺人的余弦相似度
        similarities = torch.nn.functional.cosine_similarity(
            artist_embedding.unsqueeze(0),
            self.node_embeddings,
            dim=1
        ).cpu().numpy()
        
        # 排序并获取top-k（排除自己）
        similar_indices = np.argsort(similarities)[::-1]
        
        recommendations = []
        for idx in similar_indices:
            if idx == artist_idx:
                continue
            
            # 检查是否已合作
            edge = tuple(sorted([artist_idx, idx]))
            if edge in self.existing_collabs:
                continue
            
            recommendations.append({
                'artist': self.idx_to_artist[idx],
                'similarity': float(similarities[idx])
            })
            
            if len(recommendations) >= top_k:
                break
        
        return recommendations
    
    def explain_recommendation(self, artist1: str, artist2: str, 
                              artists_df: pd.DataFrame) -> Dict:
        """
        解释推荐原因
        
        Args:
            artist1: 艺人1
            artist2: 艺人2
            artists_df: 艺人特征DataFrame
            
        Returns:
            解释信息字典
        """
        score = self.predict_collaboration_score(artist1, artist2)
        
        # 获取艺人特征
        features1 = artists_df[artists_df['artist'] == artist1].iloc[0]
        features2 = artists_df[artists_df['artist'] == artist2].iloc[0]
        
        # 计算音乐风格相似度
        style_features = [
            'avg_danceability', 'avg_energy', 'avg_valence'
        ]
        
        style_sim = 1 - np.abs(
            features1[style_features].values - features2[style_features].values
        ).mean()
        
        explanation = {
            'predicted_score': score,
            'artist_1': {
                'name': artist1,
                'avg_popularity': features1['avg_popularity'],
                'song_count': features1['song_count'],
                'collaboration_rate': features1['collaboration_rate'],
                'nationality': features1['nationality']
            },
            'artist_2': {
                'name': artist2,
                'avg_popularity': features2['avg_popularity'],
                'song_count': features2['song_count'],
                'collaboration_rate': features2['collaboration_rate'],
                'nationality': features2['nationality']
            },
            'style_similarity': float(style_sim),
            'same_nationality': features1['nationality'] == features2['nationality'],
            'avg_popularity': (features1['avg_popularity'] + features2['avg_popularity']) / 2
        }
        
        return explanation
    
    def batch_predict(self, artist_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        批量预测多对艺人合作的成功度
        
        Args:
            artist_pairs: 艺人对列表 [(artist1, artist2), ...]
            
        Returns:
            预测结果DataFrame
        """
        results = []
        
        for artist1, artist2 in artist_pairs:
            score = self.predict_collaboration_score(artist1, artist2)
            results.append({
                'artist_1': artist1,
                'artist_2': artist2,
                'predicted_score': score
            })
        
        return pd.DataFrame(results)
    
    def find_optimal_collaborator_for_new_song(self, 
                                               target_features: Dict[str, float],
                                               top_k: int = 5) -> List[Dict]:
        """
        为一首新歌寻找最佳合作艺人
        基于目标音乐特征
        
        Args:
            target_features: 目标歌曲特征 {'danceability': 0.8, 'energy': 0.7, ...}
            top_k: 返回数量
            
        Returns:
            推荐列表
        """
        print("\nFinding optimal collaborators for target song features...")
        
        # 计算每个艺人与目标特征的匹配度
        from sklearn.metrics.pairwise import cosine_similarity
        
        feature_keys = ['avg_danceability', 'avg_energy', 'avg_valence']
        target_vec = np.array([target_features.get(k.replace('avg_', ''), 0.5) 
                               for k in feature_keys]).reshape(1, -1)
        
        recommendations = []
        
        for artist, idx in self.artist_to_idx.items():
            # 获取艺人嵌入
            embedding = self.node_embeddings[idx].cpu().numpy()
            
            recommendations.append({
                'artist': artist,
                'artist_idx': idx
            })
        
        # 简单排序（这里可以加入更复杂的匹配逻辑）
        # 基于艺人流行度
        recommendations.sort(
            key=lambda x: self.node_embeddings[x['artist_idx']].norm().item(),
            reverse=True
        )
        
        return recommendations[:top_k]
    
    def visualize_top_recommendations(self, recommendations: pd.DataFrame, 
                                     save_path: str = 'recommendations.png'):
        """
        可视化推荐结果
        
        Args:
            recommendations: 推荐DataFrame
            save_path: 保存路径
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plt.figure(figsize=(12, 8))
            
            # 准备数据
            top_20 = recommendations.head(20)
            labels = [f"{row['artist_1']} x {row['artist_2']}" 
                     for _, row in top_20.iterrows()]
            scores = top_20['predicted_score'].values
            
            # 绘制条形图
            plt.barh(range(len(labels)), scores)
            plt.yticks(range(len(labels)), labels)
            plt.xlabel('Predicted Success Score')
            plt.title('Top 20 Recommended Artist Collaborations')
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualization saved to {save_path}")
            plt.close()
            
        except ImportError:
            print("Matplotlib not available for visualization")


def generate_collaboration_report(recommender: CollaborationRecommender,
                                  artist: str,
                                  artists_df: pd.DataFrame,
                                  top_k: int = 10) -> str:
    """
    生成艺人合作推荐报告
    
    Args:
        recommender: 推荐器
        artist: 目标艺人
        artists_df: 艺人特征数据
        top_k: 推荐数量
        
    Returns:
        报告文本
    """
    # 获取推荐
    recommendations = recommender.recommend_for_artist(artist, top_k)
    
    # 生成报告
    report = f"\n{'='*60}\n"
    report += f"Collaboration Recommendations for: {artist}\n"
    report += f"{'='*60}\n\n"
    
    for i, rec in enumerate(recommendations, 1):
        report += f"{i}. {rec['artist']}\n"
        report += f"   Predicted Success Score: {rec['score']:.4f}\n"
        
        # 获取详细解释
        explanation = recommender.explain_recommendation(
            artist, rec['artist'], artists_df
        )
        
        report += f"   Style Similarity: {explanation['style_similarity']:.2f}\n"
        report += f"   Same Nationality: {explanation['same_nationality']}\n"
        report += f"   Combined Avg Popularity: {explanation['avg_popularity']:.1f}\n"
        report += "\n"
    
    return report


if __name__ == "__main__":
    # 测试代码
    from model import CollaborationGNN
    from data_loader import MusicDataLoader
    from graph_builder import MusicGraphBuilder
    
    # 加载数据
    loader = MusicDataLoader("/Users/vancefeng/Desktop/ords/AML/spotify_analysis/data/Spotify_Dataset_V3.csv")
    songs, artists, collabs = loader.process_all()
    artist_to_idx, idx_to_artist = loader.get_artist_mapping()
    
    # 构建图
    builder = MusicGraphBuilder(artists, collabs, artist_to_idx)
    graph_data = builder.build_graph_data()
    
    # 加载训练好的模型
    model = CollaborationGNN(
        in_channels=graph_data.x.size(1),
        hidden_channels=128,
        out_channels=64
    )
    model.load_state_dict(torch.load('/Users/vancefeng/Desktop/ords/AML/spotify_analysis/collaboration_prediction/output_2/best_model.pt'))
    
    # 创建推荐器
    recommender = CollaborationRecommender(
        model, graph_data, artist_to_idx, idx_to_artist
    )
    
    # 测试推荐
    test_artist = "Bad Bunny"
    recommendations = recommender.recommend_for_artist(test_artist, top_k=10)
    
    print(f"\nTop 10 recommendations for {test_artist}:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['artist']} (score: {rec['score']:.4f})")
    
    # 生成完整报告
    report = generate_collaboration_report(
        recommender, test_artist, artists, top_k=5
    )
    print(report)