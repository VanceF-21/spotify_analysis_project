import argparse
import torch
import pandas as pd
import json
from pathlib import Path

from data_loader import MusicDataLoader
from graph_builder import MusicGraphBuilder
from model import CollaborationGNN, CollaborationGNNWithFeatures
from trainer import CollaborationTrainer
from recommender import CollaborationRecommender, generate_collaboration_report


# 训练模型
# python collaboration_prediction/main.py \
#     --data_path /Users/vancefeng/Desktop/ords/AML/spotify_analysis/data/Spotify_Dataset_V3.csv \
#     --mode train \
#     --encoder sage \
#     --epochs 100

# 生成推荐
# python collaboration_prediction/main.py \
#     --data_path /Users/vancefeng/Desktop/ords/AML/spotify_analysis/data/Spotify_Dataset_V3.csv \
#     --mode recommend \
#     --model_path /Users/vancefeng/Desktop/ords/AML/spotify_analysis/collaboration_prediction/output_2/best_model.pt \
#     --target_artist "Bad Bunny" \
#     --top_k 10


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(
        description='Artist Collaboration Prediction with GNN'
    )
    
    # 数据参数
    parser.add_argument('--data_path', type=str, default='/Users/vancefeng/Desktop/ords/AML/spotify_analysis/data/Spotify_Dataset_V3.csv', required=True,
                       help='Path to the CSV data file')
    parser.add_argument('--output_dir', type=str, default='/Users/vancefeng/Desktop/ords/AML/spotify_analysis/collaboration_prediction/output_3',
                       help='Output directory for results')
    
    # 模型参数
    parser.add_argument('--encoder', type=str, default='sage',
                       choices=['sage', 'gat'],
                       help='GNN encoder type')
    parser.add_argument('--hidden_dim', type=int, default=128,
                       help='Hidden layer dimension')
    parser.add_argument('--embed_dim', type=int, default=64,
                       help='Node embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of GNN layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=200,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                       help='Weight decay')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Validation set ratio')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                       help='Test set ratio')
    
    # 推荐参数
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of recommendations to generate')
    parser.add_argument('--min_score', type=float, default=0.5,
                       help='Minimum score threshold for recommendations')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='full',
                       choices=['train', 'recommend', 'full'],
                       help='Running mode')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to pre-trained model (for recommend mode)')
    parser.add_argument('--target_artist', type=str, default=None,
                       help='Target artist for recommendations')
    
    return parser.parse_args()


def train_pipeline(args, data_loader, graph_builder, graph_data):
    """训练流程"""
    print("\n" + "="*60)
    print("TRAINING PIPELINE")
    print("="*60)
    
    # 创建模型
    model = CollaborationGNN(
        in_channels=graph_data.x.size(1),
        hidden_channels=args.hidden_dim,
        out_channels=args.embed_dim,
        encoder_type=args.encoder,
        num_layers=args.num_layers,
        dropout=args.dropout
    )
    
    print(f"\nModel Architecture:")
    print(f"  Encoder: {args.encoder.upper()}")
    print(f"  Input dim: {graph_data.x.size(1)}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Embedding dim: {args.embed_dim}")
    print(f"  Layers: {args.num_layers}")
    
    # 创建训练器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    trainer = CollaborationTrainer(model, graph_data, device)
    
    # 分割数据
    trainer.split_edges(
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio
    )
    
    # 训练
    trainer.train(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        loss_type='classification'
    )
    
    # 测试
    test_results = trainer.test(loss_type='classification')
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 保存模型
    model_path = output_dir / 'best_model.pt'
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    # 保存训练历史
    history_df = pd.DataFrame(trainer.train_history)
    history_path = output_dir / 'training_history.csv'
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to: {history_path}")
    
    # 保存测试结果
    results = {
        'test_auc': test_results['auc'],
        'test_ap': test_results['ap'],
        'test_loss': test_results['loss'],
        'model_params': {
            'encoder': args.encoder,
            'hidden_dim': args.hidden_dim,
            'embed_dim': args.embed_dim,
            'num_layers': args.num_layers
        }
    }
    
    results_path = output_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Test results saved to: {results_path}")
    
    return model


def recommend_pipeline(args, model, graph_data, artist_to_idx, 
                       idx_to_artist, artists_df):
    """推荐流程"""
    print("\n" + "="*60)
    print("RECOMMENDATION PIPELINE")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建推荐器
    recommender = CollaborationRecommender(
        model, graph_data, artist_to_idx, idx_to_artist, device
    )
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. 为特定艺人推荐
    if args.target_artist:
        print(f"\nGenerating recommendations for: {args.target_artist}")
        
        recommendations = recommender.recommend_for_artist(
            args.target_artist, 
            top_k=args.top_k
        )
        
        if recommendations:
            # 保存推荐结果
            rec_df = pd.DataFrame(recommendations)
            rec_path = output_dir / f'recommendations_{args.target_artist.replace(" ", "_")}.csv'
            rec_df.to_csv(rec_path, index=False)
            print(f"Recommendations saved to: {rec_path}")
            
            # 生成报告
            report = generate_collaboration_report(
                recommender, args.target_artist, artists_df, args.top_k
            )
            
            report_path = output_dir / f'report_{args.target_artist.replace(" ", "_")}.txt'
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to: {report_path}")
            
            print(report)
    
    # 2. 生成全局推荐
    print("\nGenerating global recommendations...")
    all_recommendations = recommender.recommend_all_pairs(
        top_k=100,
        min_score=args.min_score
    )
    
    if len(all_recommendations) > 0:
        all_rec_path = output_dir / 'all_recommendations.csv'
        all_recommendations.to_csv(all_rec_path, index=False)
        print(f"All recommendations saved to: {all_rec_path}")
        
        # 打印top 10
        print("\nTop 10 Global Recommendations:")
        print(all_recommendations.head(10).to_string(index=False))
        
        # 可视化
        try:
            recommender.visualize_top_recommendations(
                all_recommendations,
                save_path=str(output_dir / 'recommendations_viz.png')
            )
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    return recommender


def main():
    """主函数"""
    args = setup_args()
    
    print("="*60)
    print("ARTIST COLLABORATION PREDICTION WITH GNN")
    print("="*60)
    
    # 1. 加载和预处理数据
    print("\n[Step 1] Loading and preprocessing data...")
    loader = MusicDataLoader(args.data_path)
    songs_df, artists_df, collabs_df = loader.process_all()
    artist_to_idx, idx_to_artist = loader.get_artist_mapping()
    
    # 保存处理后的数据
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    artists_df.to_csv(output_dir / 'artists_features.csv', index=False)
    collabs_df.to_csv(output_dir / 'collaborations.csv', index=False)
    
    # 2. 构建图
    print("\n[Step 2] Building graph...")
    builder = MusicGraphBuilder(artists_df, collabs_df, artist_to_idx)
    graph_data = builder.build_graph_data(include_negative=True)
    
    # 构建NetworkX图用于网络分析
    nx_graph = builder.build_networkx_graph()
    network_features = builder.compute_network_features(nx_graph)
    network_features.to_csv(output_dir / 'network_features.csv', index=False)
    
    # 3. 训练或加载模型
    if args.mode in ['train', 'full']:
        model = train_pipeline(args, loader, builder, graph_data)
    elif args.mode == 'recommend':
        if args.model_path is None:
            raise ValueError("model_path required for recommend mode")
        
        model = CollaborationGNN(
            in_channels=graph_data.x.size(1),
            hidden_channels=args.hidden_dim,
            out_channels=args.embed_dim,
            encoder_type=args.encoder,
            num_layers=args.num_layers,
            dropout=args.dropout
        )
        model.load_state_dict(torch.load(args.model_path))
        print(f"\nLoaded model from: {args.model_path}")
    
    # 4. 生成推荐
    if args.mode in ['recommend', 'full']:
        recommend_pipeline(
            args, model, graph_data, 
            artist_to_idx, idx_to_artist, artists_df
        )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)


if __name__ == "__main__":
    main()