import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from typing import Dict, Tuple
import time
from tqdm import tqdm


class CollaborationTrainer:
    """艺人合作预测模型训练器"""
    
    def __init__(self, model, data, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Args:
            model: GNN模型
            data: PyTorch Geometric Data对象
            device: 训练设备
        """
        self.model = model.to(device)
        self.data = data.to(device)
        self.device = device
        
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_ap': []
        }
        
    def split_edges(self, val_ratio=0.1, test_ratio=0.1):
        """
        分割边为训练集、验证集和测试集
        
        Args:
            val_ratio: 验证集比例
            test_ratio: 测试集比例
        """
        print("\nSplitting edges into train/val/test...")
        
        num_edges = self.data.edge_index.size(1) // 2  # 无向图，除以2
        edge_indices = torch.arange(num_edges) * 2  # 只取一个方向的边
        
        # 随机打乱
        perm = torch.randperm(num_edges)
        
        # 分割
        num_val = int(num_edges * val_ratio)
        num_test = int(num_edges * test_ratio)
        num_train = num_edges - num_val - num_test
        
        train_idx = perm[:num_train]
        val_idx = perm[num_train:num_train + num_val]
        test_idx = perm[num_train + num_val:]
        
        # 提取边索引
        train_edge_idx = edge_indices[train_idx]
        val_edge_idx = edge_indices[val_idx]
        test_edge_idx = edge_indices[test_idx]
        
        # 构建训练边（双向）
        train_edges = []
        for idx in train_edge_idx:
            train_edges.append(self.data.edge_index[:, idx])
            train_edges.append(self.data.edge_index[:, idx + 1])
        self.train_edge_index = torch.stack(train_edges, dim=1).to(self.device)
        
        # 验证和测试边（单向）
        self.val_pos_edge = torch.stack(
            [self.data.edge_index[:, idx] for idx in val_edge_idx], dim=1
        ).to(self.device)
        
        self.test_pos_edge = torch.stack(
            [self.data.edge_index[:, idx] for idx in test_edge_idx], dim=1
        ).to(self.device)
        
        # 提取对应的边标签（流行度）
        self.train_edge_label = torch.cat(
            [self.data.edge_label[idx:idx+2] for idx in train_edge_idx]
        ).to(self.device)
        
        self.val_edge_label = torch.stack(
            [self.data.edge_label[idx] for idx in val_edge_idx]
        ).to(self.device)
        
        self.test_edge_label = torch.stack(
            [self.data.edge_label[idx] for idx in test_edge_idx]
        ).to(self.device)
        
        print(f"Train edges: {self.train_edge_index.size(1)}")
        print(f"Val edges: {self.val_pos_edge.size(1)}")
        print(f"Test edges: {self.test_pos_edge.size(1)}")
        
    def train_epoch(self, optimizer, loss_type='classification'):
        """
        训练一个epoch
        
        Args:
            optimizer: 优化器
            loss_type: 'classification' 或 'regression'
            
        Returns:
            平均损失
        """
        self.model.train()
        
        optimizer.zero_grad()
        
        # 编码所有节点
        z = self.model.encode(self.data.x, self.train_edge_index)
        
        if loss_type == 'classification':
            # 链接预测任务（二分类）
            
            # 正样本预测
            pos_pred = self.model.decode(z, self.val_pos_edge)
            
            # 负采样
            neg_edge = negative_sampling(
                edge_index=self.train_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=self.val_pos_edge.size(1)
            ).to(self.device)
            
            neg_pred = self.model.decode(z, neg_edge)
            
            # 计算损失
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_pred, torch.ones_like(pos_pred)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_pred, torch.zeros_like(neg_pred)
            )
            loss = pos_loss + neg_loss
            
        else:  # regression
            # 流行度预测任务（回归）
            
            # 只对训练边进行预测
            # 注意：需要根据边索引提取对应的标签
            train_edge_subset = self.train_edge_index[:, :self.val_pos_edge.size(1)]
            train_label_subset = self.train_edge_label[:self.val_pos_edge.size(1)]
            
            pred = self.model.decode(z, train_edge_subset)
            
            # 标准化标签到[0, 1]范围
            normalized_label = (train_label_subset - train_label_subset.min()) / \
                             (train_label_subset.max() - train_label_subset.min() + 1e-8)
            
            loss = F.mse_loss(pred, normalized_label)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self, edge_index, edge_label=None, eval_type='classification'):
        """
        评估模型
        
        Args:
            edge_index: 要评估的边
            edge_label: 边标签（回归任务需要）
            eval_type: 'classification' 或 'regression'
            
        Returns:
            评估指标字典
        """
        self.model.eval()
        
        # 编码节点
        z = self.model.encode(self.data.x, self.train_edge_index)
        
        # 正样本预测
        pos_pred = self.model.decode(z, edge_index)
        
        if eval_type == 'classification':
            # 负采样
            neg_edge = negative_sampling(
                edge_index=self.train_edge_index,
                num_nodes=self.data.num_nodes,
                num_neg_samples=edge_index.size(1)
            ).to(self.device)
            
            neg_pred = self.model.decode(z, neg_edge)
            
            # 计算AUC和AP
            pred_all = torch.cat([pos_pred, neg_pred]).cpu().numpy()
            label_all = np.concatenate([
                np.ones(pos_pred.size(0)),
                np.zeros(neg_pred.size(0))
            ])
            
            auc = roc_auc_score(label_all, pred_all)
            ap = average_precision_score(label_all, pred_all)
            
            # 计算损失
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_pred, torch.ones_like(pos_pred)
            )
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_pred, torch.zeros_like(neg_pred)
            )
            loss = (pos_loss + neg_loss).item()
            
            return {
                'loss': loss,
                'auc': auc,
                'ap': ap
            }
        
        else:  # regression
            if edge_label is None:
                raise ValueError("edge_label required for regression evaluation")
            
            # 标准化
            normalized_label = (edge_label - edge_label.min()) / \
                             (edge_label.max() - edge_label.min() + 1e-8)
            
            loss = F.mse_loss(pos_pred, normalized_label).item()
            mae = F.l1_loss(pos_pred, normalized_label).item()
            
            return {
                'loss': loss,
                'mae': mae,
                'rmse': np.sqrt(loss)
            }
    
    def train(self, epochs=200, lr=0.01, weight_decay=5e-4,
              patience=20, loss_type='classification'):
        """
        完整训练流程
        
        Args:
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            loss_type: 损失类型
        """
        print("\n" + "="*50)
        print("Starting training...")
        print("="*50)
        
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        best_val_metric = 0 if loss_type == 'classification' else float('inf')
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            
            # 训练
            train_loss = self.train_epoch(optimizer, loss_type)
            
            # 验证
            val_metrics = self.evaluate(
                self.val_pos_edge,
                self.val_edge_label,
                loss_type
            )
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_metrics['loss'])
            
            if loss_type == 'classification':
                self.train_history['val_auc'].append(val_metrics['auc'])
                self.train_history['val_ap'].append(val_metrics['ap'])
                
                current_metric = val_metrics['auc']
                improved = current_metric > best_val_metric
            else:
                current_metric = val_metrics['loss']
                improved = current_metric < best_val_metric
            
            # 早停检查
            if improved:
                best_val_metric = current_metric
                patience_counter = 0
                # 保存最佳模型
                torch.save(self.model.state_dict(), 'best_model.pt')
            else:
                patience_counter += 1
            
            # 打印进度
            if epoch % 10 == 0 or epoch == 1:
                elapsed = time.time() - start_time
                print(f"Epoch {epoch:03d} | "
                      f"Train Loss: {train_loss:.4f} | "
                      f"Val Loss: {val_metrics['loss']:.4f} | ", end="")
                
                if loss_type == 'classification':
                    print(f"AUC: {val_metrics['auc']:.4f} | "
                          f"AP: {val_metrics['ap']:.4f} | ", end="")
                else:
                    print(f"MAE: {val_metrics['mae']:.4f} | "
                          f"RMSE: {val_metrics['rmse']:.4f} | ", end="")
                
                print(f"Time: {elapsed:.2f}s")
            
            # 早停
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pt'))
        
        print("\n" + "="*50)
        print("Training completed!")
        print("="*50)
    
    def test(self, loss_type='classification'):
        """测试模型"""
        print("\nEvaluating on test set...")
        
        test_metrics = self.evaluate(
            self.test_pos_edge,
            self.test_edge_label,
            loss_type
        )
        
        print("\nTest Results:")
        for key, value in test_metrics.items():
            print(f"  {key.upper()}: {value:.4f}")
        
        return test_metrics


if __name__ == "__main__":
    # 测试代码
    from model import CollaborationGNN
    from data_loader import MusicDataLoader
    from graph_builder import MusicGraphBuilder
    
    # 加载数据
    loader = MusicDataLoader("your_data.csv")
    songs, artists, collabs = loader.process_all()
    artist_to_idx, idx_to_artist = loader.get_artist_mapping()
    
    # 构建图
    builder = MusicGraphBuilder(artists, collabs, artist_to_idx)
    graph_data = builder.build_graph_data()
    
    # 创建模型
    model = CollaborationGNN(
        in_channels=graph_data.x.size(1),
        hidden_channels=128,
        out_channels=64,
        encoder_type='sage'
    )
    
    # 训练
    trainer = CollaborationTrainer(model, graph_data)
    trainer.split_edges(val_ratio=0.1, test_ratio=0.1)
    trainer.train(epochs=100, lr=0.01)
    trainer.test()