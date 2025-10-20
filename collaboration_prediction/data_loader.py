import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class MusicDataLoader:
    """音乐数据加载和预处理类"""
    
    def __init__(self, csv_path: str):
        """
        初始化数据加载器
        
        Args:
            csv_path: CSV文件路径
        """
        self.csv_path = csv_path
        self.raw_data = None
        self.songs_df = None
        self.artists_df = None
        self.collaborations_df = None
        
    def load_data(self) -> pd.DataFrame:
        """加载原始CSV数据"""
        print("Loading data from CSV...")
        self.raw_data = pd.read_csv(
            self.csv_path, 
            sep=';',
            encoding='utf-8'
        )
        print(f"Loaded {len(self.raw_data)} rows")
        return self.raw_data
    
    def preprocess_songs(self) -> pd.DataFrame:
        """
        预处理歌曲数据，去重并聚合
        每首歌可能有多行（每个艺人一行），需要聚合成一行
        """
        print("\nPreprocessing songs data...")
        
        # 按歌曲ID分组，聚合艺人信息
        songs_aggregated = []
        
        for song_id, group in self.raw_data.groupby('id'):
            song_info = {
                'song_id': song_id,
                'title': group['Title'].iloc[0],
                'date': group['Date'].iloc[0],
                'rank': group['Rank'].iloc[0],
                'points_total': group['Points (Total)'].iloc[0],
                'num_artists': group['# of Artist'].nunique(),
                # 音频特征
                'danceability': group['Danceability'].iloc[0],
                'energy': group['Energy'].iloc[0],
                'loudness': group['Loudness'].iloc[0],
                'speechiness': group['Speechiness'].iloc[0],
                'acousticness': group['Acousticness'].iloc[0],
                'instrumentalness': group['Instrumentalness'].iloc[0],
                'valence': group['Valence'].iloc[0],
                # 艺人列表
                'artists': group['Artist (Ind.)'].tolist(),
                'nationalities': group['Nationality'].tolist(),
                'continents': group['Continent'].tolist()
            }
            songs_aggregated.append(song_info)
        
        self.songs_df = pd.DataFrame(songs_aggregated)
        
        # 转换日期格式
        self.songs_df['date'] = pd.to_datetime(
            self.songs_df['date'], 
            format='%d/%m/%Y',
            errors='coerce'
        )
        
        print(f"Processed {len(self.songs_df)} unique songs")
        return self.songs_df
    
    def build_artist_features(self) -> pd.DataFrame:
        """
        构建艺人特征
        包括：历史平均流行度、歌曲数量、音乐风格向量等
        """
        print("\nBuilding artist features...")
        
        artist_stats = {}
        
        for _, song in self.songs_df.iterrows():
            points_per_artist = song['points_total'] / song['num_artists']
            
            for i, artist in enumerate(song['artists']):
                if artist not in artist_stats:
                    artist_stats[artist] = {
                        'song_count': 0,
                        'total_points': 0,
                        'points_list': [],
                        'collaboration_count': 0,
                        'nationality': song['nationalities'][i],
                        'continent': song['continents'][i],
                        # 音频特征累积
                        'danceability': [],
                        'energy': [],
                        'loudness': [],
                        'speechiness': [],
                        'acousticness': [],
                        'instrumentalness': [],
                        'valence': []
                    }
                
                stats = artist_stats[artist]
                stats['song_count'] += 1
                stats['total_points'] += points_per_artist
                stats['points_list'].append(points_per_artist)
                
                if song['num_artists'] > 1:
                    stats['collaboration_count'] += 1
                
                # 累积音频特征
                stats['danceability'].append(song['danceability'])
                stats['energy'].append(song['energy'])
                stats['loudness'].append(song['loudness'])
                stats['speechiness'].append(song['speechiness'])
                stats['acousticness'].append(song['acousticness'])
                stats['instrumentalness'].append(song['instrumentalness'])
                stats['valence'].append(song['valence'])
        
        # 转换为DataFrame
        artists_list = []
        for artist, stats in artist_stats.items():
            artist_features = {
                'artist': artist,
                'song_count': stats['song_count'],
                'avg_popularity': stats['total_points'] / stats['song_count'],
                'total_points': stats['total_points'],
                'collaboration_count': stats['collaboration_count'],
                'collaboration_rate': stats['collaboration_count'] / stats['song_count'],
                'nationality': stats['nationality'],
                'continent': stats['continent'],
                # 平均音频特征（艺人的音乐风格）
                'avg_danceability': np.mean(stats['danceability']),
                'avg_energy': np.mean(stats['energy']),
                'avg_loudness': np.mean(stats['loudness']),
                'avg_speechiness': np.mean(stats['speechiness']),
                'avg_acousticness': np.mean(stats['acousticness']),
                'avg_instrumentalness': np.mean(stats['instrumentalness']),
                'avg_valence': np.mean(stats['valence']),
                # 标准差（音乐风格多样性）
                'std_danceability': np.std(stats['danceability']),
                'std_energy': np.std(stats['energy'])
            }
            artists_list.append(artist_features)
        
        self.artists_df = pd.DataFrame(artists_list)
        print(f"Built features for {len(self.artists_df)} artists")
        return self.artists_df
    
    def build_collaboration_edges(self) -> pd.DataFrame:
        """
        构建艺人合作边
        每条边表示两个艺人的合作及其成功度
        """
        print("\nBuilding collaboration edges...")
        
        collaborations = []
        
        for _, song in self.songs_df.iterrows():
            if song['num_artists'] > 1:
                artists = song['artists']
                points_per_artist = song['points_total'] / song['num_artists']
                
                # 生成所有艺人对
                for i in range(len(artists)):
                    for j in range(i + 1, len(artists)):
                        collaborations.append({
                            'artist_1': artists[i],
                            'artist_2': artists[j],
                            'song_id': song['song_id'],
                            'song_title': song['title'],
                            'points': song['points_total'],
                            'rank': song['rank'],
                            'date': song['date'],
                            # 歌曲音频特征
                            'danceability': song['danceability'],
                            'energy': song['energy'],
                            'valence': song['valence']
                        })
        
        self.collaborations_df = pd.DataFrame(collaborations)
        
        # 聚合多次合作的艺人对
        collab_aggregated = self.collaborations_df.groupby(
            ['artist_1', 'artist_2']
        ).agg({
            'points': 'mean',  # 平均流行度
            'rank': 'mean',
            'song_id': 'count',  # 合作次数
            'danceability': 'mean',
            'energy': 'mean',
            'valence': 'mean'
        }).reset_index()
        
        collab_aggregated.rename(columns={'song_id': 'collab_count'}, inplace=True)
        
        self.collaborations_df = collab_aggregated
        print(f"Found {len(self.collaborations_df)} unique collaborations")
        return self.collaborations_df
    
    def get_artist_mapping(self) -> Tuple[Dict, Dict]:
        """
        创建艺人到索引的映射
        
        Returns:
            artist_to_idx: 艺人名 -> 索引
            idx_to_artist: 索引 -> 艺人名
        """
        unique_artists = sorted(self.artists_df['artist'].unique())
        artist_to_idx = {artist: idx for idx, artist in enumerate(unique_artists)}
        idx_to_artist = {idx: artist for artist, idx in artist_to_idx.items()}
        
        return artist_to_idx, idx_to_artist
    
    def process_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        执行完整的数据处理流程
        
        Returns:
            songs_df: 歌曲数据
            artists_df: 艺人特征
            collaborations_df: 合作边数据
        """
        self.load_data()
        self.preprocess_songs()
        self.build_artist_features()
        self.build_collaboration_edges()
        
        print("\n" + "="*50)
        print("Data processing completed!")
        print(f"Songs: {len(self.songs_df)}")
        print(f"Artists: {len(self.artists_df)}")
        print(f"Collaborations: {len(self.collaborations_df)}")
        print("="*50)
        
        return self.songs_df, self.artists_df, self.collaborations_df


if __name__ == "__main__":
    # 测试代码
    loader = MusicDataLoader("your_data.csv")
    songs, artists, collabs = loader.process_all()
    
    print("\nArtist features sample:")
    print(artists.head())
    
    print("\nCollaboration edges sample:")
    print(collabs.head())